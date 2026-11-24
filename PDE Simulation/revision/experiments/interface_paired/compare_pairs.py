#!/usr/bin/env python3
"""Compare paired day-14 NTM endpoint CSVs without running any simulations."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t as student


PACKAGE = Path(__file__).resolve().parents[2] / "inputs/sensitivity"
PAIRED = Path("/tmp/ntm_interface_paired")
DEFAULT_ARCHIVE = Path("/tmp/ntm_pair_archive/archive_endpoints.csv")
FACTORS = ("DB1", "DB2", "lambdaB", "KB", "KBi", "dMb", "dB", "tauB",
           "DM", "dM", "tauM", "M0", "alpha0", "alpha1", "eta", "danti")
BURDENS = ("B_MUCUS", "B_TISSUE", "M_MUCUS", "M_TISSUE", "B_TOTAL", "M_TOTAL")
OUTCOMES = BURDENS + ("B_CLEARANCE", "M_CLEARANCE", "B_M_TO_T", "M_M_TO_T",
                      "B_T_TO_M", "M_T_TO_M", "B_NET_TO_T", "M_NET_TO_T")
MODELS = ("A", "B", "C")
CONTRASTS = (("B", "A"), ("C", "A"), ("C", "B"))
ORIGINAL_IDS = set(range(1, 401))
DAY = 14


def read_table(path, columns, *, endpoint=False, archive=False):
    rows = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        has_time = "time_day" in fields
        required = ("sample_id", *columns)
        if endpoint and (not archive or has_time):
            required += ("time_day",)
        if len(set(fields)) != len(fields):
            raise ValueError(f"{path}: duplicate column names")
        missing = sorted(set(required) - set(fields))
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        for row in reader:
            where = f"{path}:{reader.line_num}"
            if None in row or None in row.values():
                raise ValueError(f"{where}: row does not match the header width")
            try:
                sid = int(row["sample_id"])
            except ValueError as error:
                raise ValueError(f"{where}: invalid sample_id {row['sample_id']!r}") from error
            if sid in rows:
                raise ValueError(f"{where}: duplicate sample_id {sid}")
            values = []
            for name in required[1:]:
                try:
                    value = float(row[name])
                except ValueError as error:
                    raise ValueError(f"{where}: invalid {name} for sample_id {sid}") from error
                if not math.isfinite(value):
                    raise ValueError(f"{where}: non-finite {name} for sample_id {sid}")
                values.append(value)
            if endpoint and has_time:
                day = values.pop()
                if not math.isclose(day, DAY, rel_tol=0, abs_tol=1e-8):
                    raise ValueError(f"{where}: sample_id {sid} time_day={day:g}, expected {DAY}")
            rows[sid] = values
    if not rows:
        raise ValueError(f"{path}: no data rows")
    missing_ids = ORIGINAL_IDS - rows.keys()
    extra_ids = rows.keys() - ORIGINAL_IDS
    problems = []
    if missing_ids and not archive:
        problems.append(f"missing IDs ({len(missing_ids)}): {sorted(missing_ids)[:8]}")
    if extra_ids:
        problems.append(f"unexpected IDs ({len(extra_ids)}): {sorted(extra_ids)[:8]}")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return rows, has_time


def paired_metrics(reference, target):
    delta = target - reference
    nonzero = reference != 0
    relative = delta[nonzero] / np.abs(reference[nonzero])
    scale = np.abs(reference) + np.abs(target)
    symmetric = np.divide(2 * delta, scale, out=np.zeros_like(delta), where=scale != 0)
    ranks = rankdata(reference)
    target_ranks = rankdata(target)
    centered = ranks - ranks.mean()
    target_centered = target_ranks - target_ranks.mean()
    rank_norm = np.linalg.norm(centered) * np.linalg.norm(target_centered)
    rho = np.clip(centered @ target_centered / rank_norm, -1, 1) if rank_norm else None
    return {
        "n": len(delta),
        "mean_delta": np.mean(delta),
        "median_delta": np.median(delta),
        "mae": np.mean(np.abs(delta)),
        "rmse": np.sqrt(np.mean(delta**2)),
        "median_abs_delta": np.median(np.abs(delta)),
        "p90_abs_delta": np.quantile(np.abs(delta), 0.90),
        "n_relative": len(relative),
        "median_relative_delta": np.median(relative) if len(relative) else None,
        "median_abs_relative_delta": np.median(np.abs(relative)) if len(relative) else None,
        "p90_abs_relative_delta": np.quantile(np.abs(relative), 0.90) if len(relative) else None,
        "median_abs_symmetric_delta": np.median(np.abs(symmetric)),
        "p90_abs_symmetric_delta": np.quantile(np.abs(symmetric), 0.90),
        "spearman_rho": rho,
        "mean_abs_rank_shift": np.mean(np.abs(target_ranks - ranks)),
    }


def compute_prcc(design, matrices, alpha):
    x = rankdata(design, axis=0)
    y = rankdata(np.column_stack([matrices[model] for model in MODELS]), axis=0)
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    n, p = x.shape
    df = n - p - 1
    if np.linalg.matrix_rank(x) != p:
        raise ValueError("original design has constant or collinear ranked inputs; cannot fit 16-factor PRCC")
    outcomes = [(model, outcome) for model in MODELS for outcome in OUTCOMES]
    results = {key: {} for key in outcomes}
    # Centering handles the intercept; project both ranked variables off the other 15 inputs.
    tol = n * np.finfo(float).eps * np.linalg.norm(y, axis=0)
    for j, factor in enumerate(FACTORS):
        controls, _ = np.linalg.qr(np.delete(x, j, axis=1), mode="reduced")
        pair = np.column_stack((x[:, j], y))
        residual = pair - controls @ (controls.T @ pair)
        rx, ry = residual[:, 0], residual[:, 1:]
        y_norm = np.linalg.norm(ry, axis=0)
        for k, key in enumerate(outcomes):
            corr = raw = adjusted = significant = None
            if y_norm[k] > tol[k]:
                corr = float(np.clip(rx @ ry[:, k] / (np.linalg.norm(rx) * y_norm[k]), -1, 1))
                statistic = abs(corr) * math.sqrt(df / (1 - corr**2)) if abs(corr) < 1 else math.inf
                raw = float(2 * student.sf(statistic, df))
                adjusted = min(1.0, p * raw)
                significant = adjusted <= alpha
            results[key][factor] = {
                "model": key[0], "outcome": key[1], "parameter": factor, "n": n,
                "PRCC": corr, "p_raw": raw, "p_Bonferroni_16_per_outcome": adjusted,
                "significant": significant,
            }
    return results


def prcc_deltas(prcc):
    for target, reference in CONTRASTS:
        for outcome in OUTCOMES:
            for factor in FACTORS:
                old = prcc[reference, outcome][factor]
                new = prcc[target, outcome][factor]
                a, b = old["PRCC"], new["PRCC"]
                sa, sb = old["significant"], new["significant"]
                comparable = a is not None and b is not None
                transitions = {(True, True): "retained", (False, False): "neither",
                               (True, False): "lost", (False, True): "gained"}
                yield {
                    "comparison": f"{target}-{reference}", "outcome": outcome, "parameter": factor,
                    "n": old["n"], "reference_PRCC": a, "target_PRCC": b,
                    "delta_PRCC": b - a if comparable else None,
                    "sign_flip": (a < 0 < b or b < 0 < a) if comparable else None,
                    "reference_p_Bonferroni": old["p_Bonferroni_16_per_outcome"],
                    "target_p_Bonferroni": new["p_Bonferroni_16_per_outcome"],
                    "reference_significant": sa, "target_significant": sb,
                    "significance_change": transitions.get((sa, sb), "undefined"),
                }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(args, archive_status, metrics, changes):
    lines = [
        f"NTM paired day-{DAY} endpoints (400 original design IDs)",
        f"Design: {args.design}",
        f"A: legacy, matched mesh/initial loads/dt ({args.a})",
        f"B: matched conservative, kappaB=DB2*tauB and kappaM=DM*tauM ({args.b})",
        f"C: previous rank-map conservative ({args.c})",
        f"Actual archive: {archive_status}",
        "Mesh, initial loads, dt and permeability assignment are run provenance, not verified by endpoint CSVs.",
        "Contrasts are target minus reference. Relative delta divides by |reference|; zero references are omitted.",
        "Symmetric delta is 2*delta/(|reference|+|target|), with joint zeros set to zero.",
        f"PRCC uses the same 16 original ranked inputs (including tauB/tauM) in A/B/C; alpha={args.alpha:g},",
        "Bonferroni within each model/outcome across 16 factors. Changes in significance are descriptive.",
        "Blank CSV values are undefined (e.g. constant ranks or no nonzero references).",
    ]
    for contrast in dict.fromkeys(row["comparison"] for row in metrics):
        rows = [row for row in metrics if row["comparison"] == contrast]
        top = sorted(rows, key=lambda r: -r["median_abs_symmetric_delta"])[:args.top]
        magnitudes = "; ".join(
            f"{r['outcome']} {r['median_abs_symmetric_delta']:.1%} (MAE {r['mae']:.3g})" for r in top
        )
        ranks = sorted((r for r in rows if r["spearman_rho"] is not None), key=lambda r: r["spearman_rho"])
        rank_text = "; ".join(f"{r['outcome']} {r['spearman_rho']:.3f}" for r in ranks[:args.top])
        lines += ["", f"{contrast} (n={rows[0]['n']}): largest median |symmetric delta|: {magnitudes}",
                  f"  Lowest Spearman rho: {rank_text or 'undefined'}"]
        block = [r for r in changes if r["comparison"] == contrast]
        if block:
            shifts = sorted((r for r in block if r["delta_PRCC"] is not None), key=lambda r: -abs(r["delta_PRCC"]))
            shift_text = "; ".join(
                f"{r['outcome']}/{r['parameter']} {r['reference_PRCC']:+.3f} to {r['target_PRCC']:+.3f}"
                for r in shifts[:args.top]
            )
            gained = sum(r["significance_change"] == "gained" for r in block)
            lost = sum(r["significance_change"] == "lost" for r in block)
            undefined = sum(r["significance_change"] == "undefined" for r in block)
            lines += [f"  Largest |PRCC shifts|: {shift_text or 'undefined'}",
                      f"  Bonferroni significance: {gained} gained, {lost} lost, {undefined} undefined."]
    return "\n".join(lines) + "\n"


def run(args):
    inputs, _ = read_table(args.design, FACTORS)
    matrices = {}
    for model, path in zip(MODELS, (args.a, args.b, args.c)):
        rows, _ = read_table(path, OUTCOMES, endpoint=True)
        matrices[model] = np.array([rows[sid] for sid in sorted(inputs)])
    comparisons = [(target, reference, matrices[target], matrices[reference], OUTCOMES)
                   for target, reference in CONTRASTS]

    archive_path = args.archive if args.archive is not None else DEFAULT_ARCHIVE
    if args.no_archive:
        archive_status = "disabled"
    elif args.archive is None and not archive_path.exists():
        archive_status = f"not found, skipped ({archive_path})"
    else:
        rows, has_time = read_table(archive_path, BURDENS, endpoint=True, archive=True)
        ids = sorted(rows)
        archived = np.array([rows[sid] for sid in ids])
        indices = np.array(ids) - 1
        comparisons.extend((model, "archive", matrices[model][indices, :len(BURDENS)], archived, BURDENS)
                           for model in MODELS)
        time_note = f"verified day {DAY}" if has_time else f"no time_day column; assumed day {DAY}"
        archive_status = f"{len(ids)}/400 IDs, six burdens only, {time_note} ({archive_path})"

    metrics = [
        {"comparison": f"{target}-{reference}", "outcome": outcome, "time_day": DAY,
         **paired_metrics(before[:, j], after[:, j])}
        for target, reference, after, before, outcomes in comparisons
        for j, outcome in enumerate(outcomes)
    ]
    design = np.array([inputs[sid] for sid in sorted(inputs)])
    prcc = compute_prcc(design, matrices, args.alpha)
    changes = list(prcc_deltas(prcc))
    report = summary(args, archive_status, metrics, changes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "paired_metrics.csv", metrics)
    write_csv(args.out_dir / "prcc.csv", [row for factors in prcc.values() for row in factors.values()])
    write_csv(args.out_dir / "prcc_deltas.csv", changes)
    (args.out_dir / "summary.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"\nWrote reports to {args.out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--design", type=Path, default=PACKAGE / "legacy_design.csv", help="original input design")
    parser.add_argument("--a", type=Path, default=PAIRED / "legacy/endpoints.csv", help="matched legacy endpoints")
    parser.add_argument("--b", type=Path, default=PAIRED / "conservative/endpoints.csv",
                        help="matched conservative endpoints")
    parser.add_argument("--c", type=Path, default=PACKAGE / "endpoints.csv", help="previous rank-map endpoints")
    archive = parser.add_mutually_exclusive_group()
    archive.add_argument("--archive", type=Path, metavar="CSV",
                         help=f"explicit archive path (otherwise use {DEFAULT_ARCHIVE} if present)")
    archive.add_argument("--no-archive", action="store_true", help="skip the optional archive")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results",
                        help="report directory")
    parser.add_argument("--alpha", type=float, default=0.05, help="Bonferroni-adjusted significance threshold")
    parser.add_argument("--top", type=int, default=3, help="number of differences to highlight per contrast")
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    if args.top < 1:
        parser.error("--top must be positive")
    try:
        run(args)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
