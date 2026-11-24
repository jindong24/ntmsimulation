#!/usr/bin/env python3
"""Summarize velocity fields and matched or archived treatment responses."""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from replay_paired_experiments import p1_l2
from revision_runs import FIELDS, read_rows, write_rows
from revision_sensitivity import GEOMETRY, SPATIAL


OUTCOMES = (*FIELDS, "B_TOTAL", "M_TOTAL")


def treatments_from_archive(old, final):
    def load(path):
        out = {}
        for row in read_rows(path):
            key = (float(row["eta"]), float(row["danti"]))
            if key in out:
                raise ValueError(f"duplicate grid point {path}: {key}")
            c = {k: float(row[k]) for k in FIELDS}
            c["B_TOTAL"] = c["B_MUCUS"] + c["B_TISSUE"]
            c["M_TOTAL"] = c["M_MUCUS"] + c["M_TISSUE"]
            out[key] = c
        return out
    for path in sorted(final.glob("*_new.csv")):
        set_id = int(path.name.split("_")[0])
        a, b = load(old / path.name), load(path)
        if a.keys() != b.keys() or len(a) != 170:
            raise ValueError(f"treatment grids do not match: {set_id}")
        for x, d in sorted(a):
            yield {"set_id": set_id, "eta": x, "danti_effective": d, "time_day": 14.,
                   **{"legacy_" + k: v for k, v in a[(x, d)].items()},
                   **{"conservative_" + k: v for k, v in b[(x, d)].items()}}


def treatment_summary(rows, output):
    table = []
    slices = []
    for sid in sorted({int(r["set_id"]) for r in rows}):
        block = [r for r in rows if int(r["set_id"]) == sid]
        grid = {(float(r["eta"]), float(r["danti_effective"])): r for r in block}
        if len(grid) != len(block) or (1.0, 0.0) not in grid:
            raise ValueError(f"invalid treatment grid for {sid}")
        etas = sorted({x for x, d in grid if d > 0})
        doses = sorted({d for x, d in grid if d > 0})
        keys = [(x, d) for x in etas for d in doses]
        for name in OUTCOMES:
            a, b = [np.array([float(grid[k][tag + "_" + name]) for k in keys]).reshape(len(etas), len(doses))
                    for tag in ("legacy", "conservative")]
            base_a, base_b = (float(grid[1.0, 0.0][tag + "_" + name]) for tag in ("legacy", "conservative"))
            if min(a.min(), b.min(), base_a, base_b) <= 0.:
                raise ValueError(f"nonpositive output: {sid} {name}")
            rel = b / a - 1.0
            log_residual = np.log10((b / base_b) / (a / base_a))
            # Maximum dose relative to the minimum positive dose at fixed eta.
            dose_a, dose_b = a[:, -1] / a[:, 0], b[:, -1] / b[:, 0]
            eta_a, eta_b = a[-1, :] / a[0, :], b[-1, :] / b[0, :]
            table.append({
                "set_id": sid, "outcome": name, "n_positive": len(keys),
                "zero_dose_ratio_new_old": base_b / base_a,
                "median_signed_relative_delta": np.median(rel),
                "median_abs_relative_delta": np.median(np.abs(rel)),
                "p90_abs_relative_delta": np.quantile(np.abs(rel), 0.9),
                "spearman_rho": spearmanr(a.ravel(), b.ravel()).statistic,
                "median_abs_log10_reference_normalized_delta": np.median(np.abs(log_residual)),
                "median_legacy_high_to_low_dose_ratio": np.median(dose_a),
                "median_conservative_high_to_low_dose_ratio": np.median(dose_b),
                "median_legacy_high_to_low_eta_ratio": np.median(eta_a),
                "median_conservative_high_to_low_eta_ratio": np.median(eta_b),
            })
            for control, x, fa, fb in (("eta", etas, dose_a, dose_b),
                                       ("danti_effective", doses, eta_a, eta_b)):
                for value, old_ratio, new_ratio in zip(x, fa, fb):
                    slices.append({"set_id": sid, "outcome": name, "fixed_parameter": control,
                                   "fixed_value": value, "legacy_max_to_min_response": old_ratio,
                                   "conservative_max_to_min_response": new_ratio})
    write_rows(output / "summary_by_set.csv", table)
    write_rows(output / "response_slices.csv", slices)


def compare_legacy_replay(matched, archive, output):
    def key(row):
        return int(row["set_id"]), float(row["eta"]), float(row["danti_effective"])
    lookup = {key(r): r for r in archive}
    checks = []
    for outcome in OUTCOMES:
        reference = np.array([float(lookup[key(r)]["legacy_" + outcome]) for r in matched])
        replay = np.array([float(r["legacy_" + outcome]) for r in matched])
        relative = np.abs(replay / reference - 1.)
        checks.append({
            "outcome": outcome, "n": len(matched),
            "median_abs_relative_delta": np.median(relative),
            "p90_abs_relative_delta": np.quantile(relative, .9),
            "spearman_rho": spearmanr(reference, replay).statistic,
        })
    write_rows(output / "legacy_replay_vs_archive.csv", checks)


def velocity_recalibrated(root, previous):
    rows = []
    for p in sorted((root / "velocity" / "fields").glob("*_velocity.npz")):
        sid = int(p.name.split("_")[0])
        for mode in ("baseline", "velocity"):
            with np.load(p.with_name(f"{sid}_{mode}.npz")) as old, np.load(
                previous / f"set_{sid:03d}" / mode / "spatial.npz"
            ) as new:
                for k in (*GEOMETRY, "time"):
                    np.testing.assert_allclose(old[k], new[k], rtol=0., atol=1e-12)
                for i, day in enumerate(old["time"]):
                    for field, key in zip(FIELDS, SPATIAL):
                        region = key.split("_")[-1]
                        xy, tri = old[region + "_xy"], old[region + "_triangles"]
                        a, b = old["legacy_" + key][i], new[key][i]
                        l2 = lambda v: p1_l2(v, xy, tri)
                        rows.append({
                            "set_id": sid, "mode": mode, "day": day, "outcome": field,
                            "legacy_l2": l2(a), "recalibrated_conservative_l2": l2(b),
                            "difference_l2": l2(b - a),
                        })
    write_rows(root / "velocity" / "recalibrated_vs_legacy.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("reference_treatment", type=Path)
    parser.add_argument("revision_results", type=Path)
    parser.add_argument("--skip-matched", action="store_true")
    parser.add_argument("--compare-recalibrated-velocity", action="store_true")
    args = parser.parse_args()
    out = args.root / "treatment"
    rows = list(treatments_from_archive(args.reference_treatment, args.revision_results / "treatment" / "data"))
    if len(rows) != 1700:
        raise ValueError("expected all 1,700 archive treatment cases")
    write_rows(out / "archive_vs_recalibrated" / "paired_endpoints.csv", rows)
    treatment_summary(rows, out / "archive_vs_recalibrated")
    if not args.skip_matched:
        matched = read_rows(out / "matched_form_endpoints.csv")
        treatment_summary(matched, out / "matched_form")
        compare_legacy_replay(matched, rows, out / "matched_form")
    if args.compare_recalibrated_velocity:
        velocity_recalibrated(args.root, args.revision_results / "velocity")


if __name__ == "__main__":
    main()
