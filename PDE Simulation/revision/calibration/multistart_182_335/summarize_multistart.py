"""Validate and compare the full-observation refits against the current fits."""

import argparse
import csv
import json
from pathlib import Path
from shutil import copyfile

import numpy as np


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def evaluate(folder, sid, label, reference=None):
    report = json.loads((folder / "metrics.json").read_text())
    traj = read(folder / "trajectory.csv")
    assert report["set"] == sid and report["fit"] == "full" and report["train_count"] == len(traj) == 83
    assert report["mesh_backend"] == "original_mshr" and report["dt"] == 1 / 6
    assert all(r["used_for_fit"] == "True" for r in traj)
    t, b, m, bp, mp = np.array([[float(r[k]) for k in ("day", "B_ABM", "M_ABM", "B_mucus", "M_mucus")]
                               for r in traj]).T
    if reference is not None:
        target, obs = reference
        np.testing.assert_array_equal(np.stack((t, b, m)), obs)
        assert report["fixed_parameters"] == target["fixed_parameters"]
        assert all(v[1:] == target["seed_and_bounds"][k][1:] for k, v in report["seed_and_bounds"].items())
    err = np.stack((bp / b - 1, mp / m - 1))
    assert np.isfinite(err).all()
    b_msre, m_msre = np.mean(err ** 2, axis=1)
    b_rmse, m_rmse = 100 * np.sqrt([b_msre, m_msre])
    np.testing.assert_allclose([b_rmse, m_rmse], [report["training"][k]["relative_rmse_pct"]
                                                for k in ("bacteria", "macrophages")], rtol=1e-12)
    row = dict(set_id=sid, start=label, points=len(t), B_MSRE=b_msre, M_MSRE=m_msre,
               joint_MSRE=b_msre + m_msre, B_RRMSE_pct=b_rmse, M_RRMSE_pct=m_rmse,
               B_after_day12_RRMSE_in_sample_pct=100 * np.sqrt(np.mean(err[0, t > 12 + 1e-12] ** 2)),
               final_B_mucus=bp[-1], final_B_ABM=b[-1], optimizer_converged=report["optimizer_converged"],
               optimizer_nfev=report["optimizer_nfev"], max_nfev=report["max_nfev"],
               failed_pde_evaluations=report["failed_pde_evaluations"],
               optimizer_message=report["optimizer_message"], selected=False)
    return row, report, np.stack((t, b, m))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    root = args.output or study / "runs/multistart_182_335"
    all_rows, best_params, selections = [], [], []
    for sid in (182, 335):
        base = study / "calibration/full" / f"set_{sid:03d}"
        before, saved, observations = evaluate(base, sid, "previous_best")
        group = [(before, saved, base)]
        folder = root / "runs" / f"set_{sid:03d}"
        for label in ("near_current", "legacy_mucus_side", "low_tissue_diffusion"):
            trial = folder / label
            result, report, _ = evaluate(trial, sid, label, (saved, observations))
            group.append((result, report, trial))
        for trial in sorted(folder.glob("refine_*")):
            if (trial / "metrics.json").is_file():
                result, report, _ = evaluate(trial, sid, trial.name, (saved, observations))
                group.append((result, report, trial))
        best, report, path = min(group, key=lambda item: item[0]["joint_MSRE"])
        best["selected"] = True
        all_rows.extend(row for row, _, _ in group)
        params = dict(set_id=sid, start=best["start"], **report["fitted_parameters"], **report["fixed_parameters"])
        best_params.append(params)
        selections.append({"set_id": sid, "selected_start": best["start"],
                           "previous_joint_MSRE": before["joint_MSRE"], "selected_joint_MSRE": best["joint_MSRE"],
                           "relative_joint_MSRE_reduction_pct": 100 * (1 - best["joint_MSRE"] / before["joint_MSRE"]),
                           "previous_B_RRMSE_pct": before["B_RRMSE_pct"], "selected_B_RRMSE_pct": best["B_RRMSE_pct"],
                           "previous_M_RRMSE_pct": before["M_RRMSE_pct"], "selected_M_RRMSE_pct": best["M_RRMSE_pct"],
                           "selected_optimizer_converged": best["optimizer_converged"]})
        selected = root / "selected" / f"set_{sid:03d}"
        selected.mkdir(parents=True, exist_ok=True)
        for name in ("trajectory.csv", "metrics.json"):
            copyfile(path / name, selected / name)
        print(f"{sid}: {before['joint_MSRE']:.8g} -> {best['joint_MSRE']:.8g}; {best['start']}")
    write(root / "all_fits.csv", all_rows)
    write(root / "selected_comparison.csv", selections)
    write(root / "best_parameters.csv", best_params)


if __name__ == "__main__":
    main()
