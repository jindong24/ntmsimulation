"""Export completed original-mesh tau results to new, archive-compatible XLSX files."""

import argparse
import csv
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from openpyxl import load_workbook
from write_workbook import write_workbook


HERE = Path(__file__).resolve().parent
DEFAULT_COORDINATE = HERE.parents[1] / "inputs/sensitivity/coordinate.xlsx"
VARIABLES = (
    "DB1", "DB2", "DM", "KB", "KBi", "dB", "dM", "M0", "alpha0", "alpha1",
    "tauM", "tauB", "lambdaB", "dMb", "eta", "danti",
)
REGIONS = {"Mucus": 838, "Tissue": 842}
FIELDS = {
    "B_MUCUS": "bacteria_mucus",
    "B_TISSUE": "bacteria_tissue",
    "M_MUCUS": "macrophage_mucus",
    "M_TISSUE": "macrophage_tissue",
}
CUMULATIVE = (
    "B_CLEARANCE", "M_CLEARANCE", "B_M_TO_T", "M_M_TO_T",
    "B_T_TO_M", "M_T_TO_M", "B_NET_TO_T", "M_NET_TO_T",
)
TREATMENT = (*FIELDS, "eta", "danti", "B0", "M0")
SETUPS = ("legacy_calibrated", "recalibrated")
SETS = (48, 83, 182, 188, 335, 586, 646, 756, 808, 843)
DAYS = np.arange(15)


def numeric(value, label):
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{label}: expected a number, got {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"{label}: nonfinite value {value!r}")
    return result


def excel_cell(value):
    if value == "":
        return None
    try:
        float(value)
    except ValueError:
        return value
    return numeric(value, "CSV value")


def csv_table(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        table = csv.reader(stream)
        header = next(table, [])
        if not header or "" in header or len(header) != len(set(header)):
            raise ValueError(f"{path}: missing, blank, or duplicate headers")
        if missing := set(required) - set(header):
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = []
        for line, row in enumerate(table, 2):
            if len(row) != len(header):
                raise ValueError(f"{path}:{line}: expected {len(header)} columns")
            rows.append(dict(zip(header, row)))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return header, rows


def indexed(rows, count, label):
    result = {}
    for row in rows:
        sid = int(row["sample_id"])
        if sid in result:
            raise ValueError(f"{label}: duplicate sample ID {sid}")
        result[sid] = row
    if set(result) != set(range(1, count + 1)):
        raise ValueError(f"{label}: expected sample IDs 1..{count}")
    return result


def sheet(name, rows):
    return {"name": name, "rows": rows}


def csv_sheet(name, header, rows):
    return sheet(name, [header, *[[excel_cell(row[k]) for k in header] for row in rows]])


def basic_setting(design, ranges, min_key, max_key):
    fixed = (
        ("beta0", r"\beta_0"), ("beta1", r"\beta_1"), ("m0", "m_0"),
        ("Dm", "D_m"), ("dm", "d_m"), ("delta", r"\delta"),
    )
    varied = (
        ("lambdaB", r"\lambda_B"), ("dB", "d_B"), ("KB", "K_B"), ("KBi", "K_Bi"),
        ("DB1", "DB_1"), ("DB2", "DB_2"), ("DM", "D_M"), ("dMb", "d_Mb"),
        ("dM", "d_M"), ("alpha0", r"\alpha_0"), ("alpha1", r"\alpha_1"),
        ("M0", "M_0"), ("tauB", r"\tau_B"), ("tauM", r"\tau_M"),
    )
    fixed_values = []
    for key, label in fixed:
        values = {numeric(r[key], key) for r in design}
        if len(values) != 1:
            raise ValueError(f"{key}: not fixed across the sensitivity design")
        fixed_values.append((label, values.pop()))
    by_name = {r["parameter"]: r for r in ranges}
    if len(by_name) != len(ranges):
        raise ValueError("ranges.csv: duplicate parameters")
    rows = [["Fixed Parameters", None, None, "Variable Parameter", None, "min", "max"]]
    for i, (key, label) in enumerate(varied):
        lo = numeric(by_name[key][min_key], f"{key} lower bound")
        hi = numeric(by_name[key][max_key], f"{key} upper bound")
        if lo > hi:
            raise ValueError(f"{key}: reversed design bounds")
        a, b = fixed_values[i] if i < len(fixed_values) else (None, None)
        rows.append([a, b, None, label, None, lo, hi])
    return sheet("Basic Setting", rows)


def archived_coordinates(path):
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        if book.sheetnames != list(REGIONS):
            raise ValueError(f"{path}: expected the original Mucus and Tissue tabs")
        result = {}
        for region, size in REGIONS.items():
            rows = iter(book[region].values)
            if next(rows, None) != ("x", "y"):
                raise ValueError(f"{path}:{region}: expected x, y")
            xy = np.asarray(list(rows), dtype=float)
            if xy.shape != (size, 2) or not np.isfinite(xy).all():
                raise ValueError(f"{path}:{region}: expected {size} finite x,y rows")
            result[region.lower()] = xy
        return result
    finally:
        book.close()


def coordinate_order(archive, native, label):
    if native.shape != archive.shape or not np.isfinite(native).all():
        raise ValueError(f"{label}: expected coordinate shape {archive.shape} with finite values")
    # Match each archived point to exactly one native vertex and vice versa.
    hits = np.max(np.abs(archive[:, None] - native[None, :]), axis=2) <= 1e-12
    if not (np.all(hits.sum(axis=0) == 1) and np.all(hits.sum(axis=1) == 1)):
        raise ValueError(f"{label}: no unambiguous coordinate bijection within 1e-12")
    return np.argmax(hits, axis=1)


def array(npz, key, shape):
    data = npz[key]
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError(f"{key}: expected shape {shape} and finite numbers")
    return data


def sample_sheets(npz, design, endpoint, archive, geometry_cache, label):
    values = [(k, numeric(design[k], k)) for k in VARIABLES]
    if "zeta" in design and not math.isclose(
        numeric(design["zeta"], "zeta"), values[-1][1], rel_tol=1e-12
    ):
        raise ValueError("danti and zeta disagree on the sensitivity multiplier")
    sheets = [sheet("variable", [["Variable", "Value"], *values])]
    daily_time = array(npz, "daily_time", (len(DAYS),))
    if not np.allclose(daily_time, DAYS, atol=1e-10, rtol=0):
        raise ValueError("daily_time: expected days 0..14")
    time = npz["time"]
    if (time.ndim != 1 or not time.size or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0)
            or not np.allclose(time[[0, -1]], [0, 14], atol=1e-10, rtol=0)):
        raise ValueError("time: expected strictly increasing times spanning days 0..14")
    means = array(npz, "concentrations", (len(time), len(FIELDS)))[-1]
    flux = array(npz, "cumulative", (len(time), len(CUMULATIVE)))[-1]
    expected = dict(zip(FIELDS, means)) | dict(zip(CUMULATIVE, flux)) | {
        "time_day": time[-1], "B_TOTAL": means[0] + means[1], "M_TOTAL": means[2] + means[3],
    }
    for key, value in expected.items():
        if key in endpoint and not math.isclose(
            numeric(endpoint[key], key), value, rel_tol=1e-10, abs_tol=1e-15
        ):
            raise ValueError(f"endpoints.csv {key} disagrees with the NPZ at day 14")
    orders = {}
    for region, original in archive.items():
        native = array(npz, f"{region}_xy", original.shape)
        if region not in geometry_cache or not np.array_equal(geometry_cache[region][0], native):
            order = coordinate_order(original, native, f"{label}:{region}")
            geometry_cache[region] = (native, order)
        orders[region] = geometry_cache[region][1]
    for name, key in FIELDS.items():
        region = key.split("_")[1]
        order = orders[region]
        spatial = array(npz, key, (len(DAYS), len(order)))[:, order].T
        sheets.append(sheet(name, [[str(d) for d in DAYS], *spatial.tolist()]))
    return sheets


def sensitivity_jobs(source, coordinates, count, part="all", sample_ids=None):
    header, design_rows = csv_table(
        source / "design.csv", ("sample_id", *VARIABLES, "danti_effective")
    )
    endpoint_header, endpoint_rows = csv_table(
        source / "endpoints.csv", ("sample_id", "time_day", *FIELDS)
    )
    if part != "samples":
        ranges_path = source / "ranges.csv"
        if ranges_path.is_file():
            range_header, ranges = csv_table(ranges_path, ("parameter",))
        else:
            range_header = ("parameter", "min", "max")
            ranges = [
                {"parameter": name, "min": min(float(r[name]) for r in design_rows),
                 "max": max(float(r[name]) for r in design_rows)}
                for name in VARIABLES
            ]
            # Retain the design bounds, rather than the observed extrema, for the rank map.
            mapping_path = source / "tauB_mapping_range.csv"
            if mapping_path.is_file():
                _, mapping = csv_table(mapping_path, ("design_minimum", "design_maximum"))
                row = next(r for r in ranges if r["parameter"] == "tauB")
                row.update(min=mapping[0]["design_minimum"], max=mapping[0]["design_maximum"])
    else:
        range_header, ranges = (), []
    min_key, max_key = (
        ("design_minimum", "design_maximum") if "design_minimum" in range_header else ("min", "max")
    )
    if any(k.lower().startswith("kappa") for k in [*header, *(r["parameter"] for r in ranges)]):
        raise ValueError("sensitivity: expected tauB and tauM with no kappa fields")
    design = indexed(design_rows, count, "design.csv")
    endpoints = indexed(endpoint_rows, count, "endpoints.csv")
    archive = archived_coordinates(coordinates)
    cache = {}
    if part != "samples":
        settings = basic_setting(design_rows, ranges, min_key, max_key)
        yield "sensitivity/parameter.xlsx", [
            settings, csv_sheet("Parameters", header, design.values()),
            csv_sheet("Endpoints", endpoint_header, endpoints.values()),
            csv_sheet("Design ranges", range_header, ranges),
        ]
    if part == "summary":
        return
    for sid in sorted(design) if sample_ids is None else sample_ids:
        path = source / "samples" / f"sample_{sid:03d}.npz"
        try:
            with np.load(path, allow_pickle=False) as npz:
                sheets = sample_sheets(npz, design[sid], endpoints[sid], archive, cache, path.name)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        yield f"sensitivity/samples_xlsx/sample_{sid}.xlsx", sheets


def treatment_jobs(source, sets):
    for setup in SETUPS:
        for sid in sets:
            relative = f"treatment/{setup}/data/{sid}_new"
            path = source / f"{relative}.csv"
            header, rows = csv_table(path, TREATMENT)
            if len(rows) != 170:
                raise ValueError(f"{path}: expected 170 data rows, got {len(rows)}")
            seen = set()
            matrix = [TREATMENT]
            for i, row in enumerate(rows, 2):
                original = [numeric(row[k], f"{path}:{i}:{k}") for k in TREATMENT]
                key = (original[4], original[5])
                if key in seen:
                    raise ValueError(f"{path}:{i}: duplicate eta/danti pair {key}")
                seen.add(key)
                matrix.append(original)
            if (1.0, 0.0) not in seen:
                raise ValueError(f"{path}: missing eta=1, danti=0 reference row")
            yield f"{relative}.xlsx", [sheet("Sheet", matrix)]


def export(args):
    output = args.output.resolve()
    if output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Output already exists: {args.output}")
    if not output.parent.is_dir():
        raise ValueError(f"Output parent does not exist: {output.parent}")
    sensitivity = args.scope != "treatment"
    treatment = args.scope != "sensitivity"
    with TemporaryDirectory(prefix=".ntm-tau-xlsx-", dir=output.parent) as tmp:
        stage = Path(tmp) / "excel"
        stage.mkdir()
        jobs = []
        if sensitivity:
            sample_ids = range(args.sample_id_range[0], args.sample_id_range[1] + 1) if args.sample_id_range else None
            jobs.append(sensitivity_jobs(args.sensitivity_branch or args.source / "sensitivity",
                                         args.coordinate, args.sample_count,
                                         args.sensitivity_part, sample_ids))
        if treatment:
            jobs.append(treatment_jobs(args.source, args.treatment_sets))
        total = 0
        for group in jobs:
            for name, sheets in group:
                write_workbook(stage / name, sheets)
                total += 1
                if total % 25 == 0:
                    print(f"Staged {total} workbooks", flush=True)
        if sensitivity and args.sensitivity_part != "samples":
            shutil.copyfile(args.coordinate, stage / "sensitivity/coordinate.xlsx")
            source = args.sensitivity_branch or args.source / "sensitivity"
            shutil.copyfile(source / "design.csv", stage / "sensitivity/parameter_set_samples.csv")
            total += 1
        if sum(1 for _ in stage.rglob("*.xlsx")) != total:
            raise RuntimeError("Unexpected staged workbook count; no export was published")
        output.mkdir()  # Reserve the name without replacing a pre-existing directory.
        try:
            stage.rename(output)
        except OSError:
            if not any(output.iterdir()):
                output.rmdir()
            raise
    print(f"Exported {total} workbooks to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="completed result root")
    parser.add_argument("--output", type=Path, required=True, help="new export directory")
    parser.add_argument("--coordinate", type=Path, default=DEFAULT_COORDINATE)
    parser.add_argument("--sensitivity-branch", type=Path, help="branch containing design, endpoints and samples")
    parser.add_argument("--scope", choices=("all", "sensitivity", "treatment"), default="all")
    parser.add_argument("--sensitivity-part", choices=("all", "summary", "samples"), default="all")
    parser.add_argument("--sample-id-range", type=int, nargs=2, metavar=("FIRST", "LAST"))
    parser.add_argument("--sample-count", type=int, default=400, choices=range(1, 401),
                        metavar="1..400", help="override for matching smaller fixture tables")
    parser.add_argument("--treatment-sets", nargs="+", type=int, choices=SETS, default=SETS)
    args = parser.parse_args()
    if len(set(args.treatment_sets)) != len(args.treatment_sets):
        parser.error("treatment sets must be unique")
    if args.sample_id_range and not 1 <= args.sample_id_range[0] <= args.sample_id_range[1] <= args.sample_count:
        parser.error("invalid sample ID range")
    try:
        export(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")


if __name__ == "__main__":
    main()
