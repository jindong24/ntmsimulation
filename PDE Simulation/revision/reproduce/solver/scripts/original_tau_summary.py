#!/usr/bin/env python3
"""Summarize original-mesh calibration and build the matched sensitivity design."""

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_conservative import concentration_weights
from lung_pde.constants import SUPPORTED_SET_IDS
from revision_runs import FIELDS, legacy_parameters, load_best, make_model, read_rows, write_rows


def calculate_scores(p, obs):
    diff = (np.asarray(p) - obs) / obs
    msre = np.mean(diff**2, axis=0)
    return dict(B_MSRE=float(msre[0]), M_MSRE=float(msre[1]), joint_MSRE=float(msre.sum()),
                B_RRMSE_pct=float(100 * np.sqrt(msre[0])), M_RRMSE_pct=float(100 * np.sqrt(msre[1])))


def legacy_error(job):
    sid, params = job
    model = make_model(sid, "calibration", params, 32, 1 / 6, conservative=False)
    weights = concentration_weights(model)[[0, 2]]
    # No parameter optimization here: use the same original calibrated XML as the paper.
    model.reset()
    pred = []
    for t in model.abm.times:
        model.step(float(t))
        pred.append(weights @ model.w.vector().get_local())
    obs = np.column_stack((model.abm.bacteria, model.abm.macrophages))
    row = {"set_id": sid, **calculate_scores(pred, obs)}
    print(f"legacy error reference {sid:03d}: B={row['B_MSRE']:.5g}, M={row['M_MSRE']:.5g}", flush=True)
    return row


def legacy_table(root, old_root, workers, ids):
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(sid, legacy_parameters(old_root, sid, "param_final2")) for sid in ids]
    if workers == 1:
        old = list(map(legacy_error, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            old = list(pool.map(legacy_error, jobs))
    write_rows(root / "calibration" / "legacy_full_reference.csv", old)
    return old


def error_table(root, old_root, workers, ids):
    cached = root / "calibration" / "legacy_full_reference.csv"
    if cached.is_file():
        old = [{"set_id": int(r["set_id"]), **{k: float(v) for k, v in r.items() if k != "set_id"}}
               for r in read_rows(cached)]
        if [r["set_id"] for r in old] != list(ids):
            raise ValueError("legacy calibration references do not cover all original sets")
    else:
        old = legacy_table(root, old_root, workers, ids)
    by_id = {r["set_id"]: r for r in old}
    rows, parameters = [], {"full": [], "partial": []}
    for sid in ids:
        for mode in parameters:
            folder = root / "calibration" / mode / f"set_{sid:03d}"
            metrics = json.loads((folder / "metrics.json").read_text())
            traj = read_rows(folder / "trajectory.csv")
            t = np.array([float(r["day"]) for r in traj])
            obs = np.array([(float(r["B_ABM"]), float(r["M_ABM"])) for r in traj])
            pde = np.array([(float(r["B_mucus"]), float(r["M_mucus"])) for r in traj])
            is_training = np.array([r["used_for_fit"] == "True" for r in traj])
            cutoff = 83 if mode == "full" else 72
            if len(t) != 83 or not np.array_equal(is_training, np.arange(83) < cutoff):
                raise ValueError(f"incorrect calibration window for {sid}/{mode}")
            record = {"set_id": sid, "mode": mode, "training_points": cutoff, "holdout_points": len(t) - cutoff,
                      "optimizer_converged": metrics["optimizer_converged"],
                      "optimizer_nfev": metrics["optimizer_nfev"]}
            scores = {"legacy_full": {k: v for k, v in by_id[sid].items() if k != "set_id"},
                      "train": calculate_scores(pde[:cutoff], obs[:cutoff])}
            if mode == "partial":
                scores["holdout"] = calculate_scores(pde[cutoff:], obs[cutoff:])
                scores["persistence"] = calculate_scores(obs[cutoff - 1], obs[cutoff:])
            else:
                for name in ("holdout", "persistence"):
                    scores[name] = dict.fromkeys(scores["train"], "")
            record.update({f"{group}_{k}": v for group, result in scores.items() for k, v in result.items()})
            rows.append(record)
            parameters[mode].append({"set_id": sid, **metrics["fitted_parameters"], **metrics["fixed_parameters"]})
    write_rows(root / "calibration" / "error_comparison.csv", rows)
    for mode, table in parameters.items():
        write_rows(root / "calibration" / f"{mode}_parameters.csv", table)


def rank_map(values, lo, hi):
    if not 0. < lo < hi or not np.isfinite([lo, hi]).all():
        raise ValueError("expected a finite non-degenerate positive fitted tau interval")
    x = np.asarray(values)
    if len(np.unique(x)) != len(x):
        raise ValueError("archived sensitivity ranks must be distinct")
    out = np.empty(len(x))
    q = (np.arange(len(x)) + .5) / len(x)
    out[np.argsort(x, kind="stable")] = np.exp(np.log(lo) + q * (np.log(hi) - np.log(lo)))
    return out


def sensitivity_design(root, original, original_ranges):
    sources = read_rows(original)
    if [int(r["sample_id"]) for r in sources] != list(range(1, 401)):
        raise ValueError("expected 400 original sample IDs")
    params = load_best(root, SUPPORTED_SET_IDS)
    intervals, mappings = {}, {}
    for k in ("tauB", "tauM"):
        values = [p[k] for p in params.values()]
        intervals[k] = (min(values), max(values))
        mappings[k] = rank_map([float(r[k]) for r in sources], *intervals[k])
    mapped = [dict(r, **{k: v[i] for k, v in mappings.items()}) for i, r in enumerate(sources)]
    destination = root / "sensitivity"
    write_rows(destination / "design.csv", mapped)
    write_rows(destination / "legacy_design.csv", sources)
    template = {r["parameter"]: r for r in read_rows(original_ranges)}
    ranges = []
    for k in mapped[0]:
        if k == "sample_id":
            continue
        observed = [float(r[k]) for r in mapped]
        if k in mappings:
            lo, hi = intervals[k]
            row = {"parameter": k, "design_minimum": lo, "design_maximum": hi,
                   "sample_minimum": min(observed), "sample_maximum": max(observed),
                   "provenance": "Original tau ranks mapped to log-stratum centers of full refit range; cm/day"}
        else:
            row = template[k].copy()
        ranges.append(row)
    write_rows(destination / "ranges.csv", ranges)


def treatment_summary(root):
    treatment = root / "treatment"
    branches = [treatment / branch for branch in ("legacy_calibrated", "recalibrated")]
    dt = [json.loads((branch / "batch_metadata.json").read_text())["dt"] for branch in branches]
    rows = []
    for sid in SUPPORTED_SET_IDS:
        name = f"{sid}_new.csv"
        paths = [branch / "data" / name for branch in branches]
        before, after = [read_rows(p) for p in paths]
        coords = [(r["eta"], r["danti"]) for r in before]
        if coords != [(r["eta"], r["danti"]) for r in after]:
            raise ValueError(f"treatment coordinate mismatch for {sid}")
        for data, d in zip((before, after), dt):
            for r in data:
                r.setdefault("dt_day", d)
        if any(float(a["dt_day"]) != float(b["dt_day"]) for a, b in zip(before, after)):
            raise ValueError(f"treatment time-step mismatch for {sid}")
        for p, data in zip(paths, (before, after)):
            write_rows(p, data)
        positive = np.array([float(r[1]) > 0. for r in coords])
        old, new = [np.array([[float(r[k]) for k in FIELDS] for r in table])
                    for table in (before, after)]
        old = np.c_[old, old[:, 0] + old[:, 1], old[:, 2] + old[:, 3]]
        new = np.c_[new, new[:, 0] + new[:, 1], new[:, 2] + new[:, 3]]
        for j, k in enumerate((*FIELDS, "B_TOTAL", "M_TOTAL")):
            rows.append({
                "set_id": sid, "outcome": k, "n_positive_dose": int(positive.sum()),
                "original_calibration_zero_dose": old[~positive, j].item(),
                "new_calibration_zero_dose": new[~positive, j].item(),
                "original_calibration_median": np.median(old[positive, j]),
                "new_calibration_median": np.median(new[positive, j]),
                "median_paired_difference": np.median(new[positive, j] - old[positive, j]),
                "median_absolute_paired_difference": np.median(np.abs(new[positive, j] - old[positive, j])),
            })
    write_rows(treatment / "paired_setup_summary.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--original-design", type=Path)
    parser.add_argument("--original-ranges", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--treatment-summary", action="store_true")
    args = parser.parse_args()
    if args.original_design is not None:
        sensitivity_design(args.root, args.original_design, args.original_ranges)
    if args.legacy_root is not None:
        operation = legacy_table if args.legacy_only else error_table
        operation(args.root, args.legacy_root, args.workers, SUPPORTED_SET_IDS)
    if args.treatment_summary:
        treatment_summary(args.root)


if __name__ == "__main__":
    main()
