"""Convert flux CSVs to single-tab workbooks for merge_flux.py."""

import argparse
import csv
from pathlib import Path

import numpy as np
from openpyxl import Workbook

from compute_flux import COLUMNS, TIME


def write_sheet(source, target):
    with source.open(newline="") as stream:
        reader = csv.reader(stream)
        header = tuple(next(reader))
        data = np.array(list(reader), dtype=float)
    if (header != COLUMNS or data.shape != (len(TIME), len(COLUMNS))
            or not np.isfinite(data).all() or not np.allclose(data[:, 0], TIME)):
        raise ValueError(f"{source}: invalid flux table")
    book = Workbook()
    tab = book.active
    tab.title = "flux"
    for i, title in enumerate(header):
        tab.column_dimensions[chr(65 + i)].width = 14 if i == 0 else 21
    tab.append(header)
    for row in data.tolist():
        tab.append(row)
    target.parent.mkdir(exist_ok=True, parents=True)
    book.save(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ids", type=int, nargs="+", default=range(1, 401))
    args = parser.parse_args()
    for sid in args.ids:
        write_sheet(args.raw / f"sample_{sid}.csv", args.output / f"sample_{sid}.xlsx")
