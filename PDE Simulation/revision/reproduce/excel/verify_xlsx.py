"""Verify the exported workbooks against the completed CSV and NPZ data."""

import argparse
import csv
import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from openpyxl import load_workbook

from export_excel import (
    DEFAULT_COORDINATE, FIELDS, SETS, SETUPS, TREATMENT, archived_coordinates, coordinate_order,
)


def verify_sensitivity(root, source, archived):
    sensitivity = root / "sensitivity"
    paths = list((sensitivity / "samples_xlsx").glob("sample_*.xlsx"))
    assert {int(p.stem.split("_")[1]) for p in paths} == set(range(1, 401))
    assert len(paths) == 400
    archive = archived_coordinates(archived)
    assert hashlib.sha256(archived.read_bytes()).digest() == hashlib.sha256(
        (sensitivity / "coordinate.xlsx").read_bytes()
    ).digest()
    with (source / "design.csv").open(newline="") as f:
        design = {int(r["sample_id"]): r for r in csv.DictReader(f)}
    for sid in (1, 100, 101, 200, 201, 300, 301, 400):
        book = load_workbook(sensitivity / "samples_xlsx" / f"sample_{sid}.xlsx",
                             read_only=True, data_only=True)
        try:
            assert book.sheetnames == ["variable", *FIELDS]
            variable = dict(list(book["variable"].values)[1:])
            for name, value in variable.items():
                np.testing.assert_allclose(value, float(design[sid][name]), rtol=1e-14)
            with np.load(source / "samples" / f"sample_{sid:03d}.npz", allow_pickle=False) as npz:
                for sheet, key in FIELDS.items():
                    rows = list(book[sheet].values)
                    assert tuple(map(str, rows[0])) == tuple(map(str, range(15)))
                    region = key.split("_")[1]
                    order = coordinate_order(archive[region], npz[f"{region}_xy"], sheet)
                    np.testing.assert_allclose(np.array(rows[1:], dtype=float), npz[key][:, order].T,
                                               atol=1e-25, rtol=1e-14)
        finally:
            book.close()
    with (source / "endpoints.csv").open(newline="") as f:
        endpoints = list(csv.reader(f))
    book = load_workbook(sensitivity / "parameter.xlsx", read_only=True, data_only=True)
    try:
        assert book.sheetnames == ["Basic Setting", "Parameters", "Endpoints", "Design ranges"]
        rows = list(book["Endpoints"].values)
        assert len(rows) == len(endpoints) == 401
        assert list(rows[0]) == endpoints[0]
        np.testing.assert_allclose(np.array(rows[1:], dtype=float), np.array(endpoints[1:], dtype=float),
                                   rtol=1e-14, atol=1e-25)
    finally:
        book.close()
    assert len(list(sensitivity.rglob("*.xlsx"))) == 402
    print("400 sample files, the parameter workbook, and the coordinate workbook verified")


def verify_treatment(root, source, sets):
    for branch in SETUPS:
        for sid in sets:
            path = Path("treatment") / branch / "data" / f"{sid}_new"
            with (source / path.with_suffix(".csv")).open(newline="") as f:
                expected = [[float(row[k]) for k in TREATMENT] for row in csv.DictReader(f)]
            book = load_workbook(root / path.with_suffix(".xlsx"), read_only=True, data_only=True)
            try:
                assert book.sheetnames == ["Sheet"]
                header, *actual = book.active.values
                assert tuple(header) == TREATMENT
                assert len(actual) == len(expected) == 170
                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-25)
            finally:
                book.close()
    print(f"{2 * len(sets)} treatment workbooks verified")


def verify_zip_files(root):
    for file in root.rglob("*.xlsx"):
        with ZipFile(file) as z:
            assert z.testzip() is None, file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scope", choices=("all", "sensitivity", "treatment"), default="all")
    parser.add_argument("--sensitivity-branch", type=Path)
    parser.add_argument("--treatment-sets", nargs="+", type=int, choices=SETS, default=SETS)
    parser.add_argument("--coordinate", type=Path, default=DEFAULT_COORDINATE)
    args = parser.parse_args()
    if args.scope != "treatment":
        verify_sensitivity(args.root, args.sensitivity_branch or args.source / "sensitivity", args.coordinate)
    if args.scope != "sensitivity":
        verify_treatment(args.root, args.source, args.treatment_sets)
    verify_zip_files(args.root)
