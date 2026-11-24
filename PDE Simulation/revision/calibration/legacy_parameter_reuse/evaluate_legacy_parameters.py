"""Replay the original full-fit parameters using conservative exchange."""

import argparse
from pathlib import Path
import sys

import numpy as np


def evaluate(task):
    root, sid = task
    solver = root / "reproduce" / "solver"
    sys.path[:0] = [str(solver / "scripts"), str(solver / "src")]
    from calibrate_conservative import concentration_weights, predict
    from original_tau_summary import calculate_scores
    from revision_runs import legacy_parameters, make_model, map_legacy_exchange, read_rows

    params = legacy_parameters(root / "inputs" / "legacy_xml", sid, "param_final2")
    mapped = map_legacy_exchange(params)
    assert all(params[k] == mapped[k] for k in params if k not in ("tauB", "tauM"))
    assert mapped["tauB"] == params["DB2"] * params["tauB"]
    assert mapped["tauM"] == params["DM"] * params["tauM"]
    model = make_model(sid, "calibration", mapped, 32, 1 / 6, conservative=True)
    model.solver.parameters["relative_tolerance"] = 1e-8
    t = model.abm.times
    assert len(t) == 83
    np.testing.assert_allclose(t, np.arange(1, 84) / 6, atol=1e-12)
    obs = np.column_stack((model.abm.bacteria, model.abm.macrophages))
    ref = read_rows(root / "calibration" / "full" / f"set_{sid:03d}" / "trajectory.csv")
    np.testing.assert_allclose(t, [float(r["day"]) for r in ref], atol=1e-12)
    np.testing.assert_allclose(obs, [[float(r["B_ABM"]), float(r["M_ABM"])] for r in ref], rtol=1e-12)
    weights = concentration_weights(model)
    y = predict(model, mapped, t, weights)
    score = calculate_scores(y[:, (0, 2)], obs)
    trajectory = [dict(set_id=sid, day=float(ti), B_ABM=float(b), M_ABM=float(m),
                       B_mucus=float(bm), B_tissue=float(bt), M_mucus=float(mm), M_tissue=float(mt))
                  for ti, (b, m), (bm, bt, mm, mt) in zip(t, obs, y)]
    mapping = dict(set_id=sid, DB2=params["DB2"], DM=params["DM"], tauB_old=params["tauB"],
                   tauM_old=params["tauM"], tauB_applied=mapped["tauB"], tauM_applied=mapped["tauM"])
    print(f"{sid}: B={score['B_MSRE']:.6g}, M={score['M_MSRE']:.6g}", flush=True)
    return sid, score, trajectory, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    root = args.study.resolve()
    sys.path[:0] = [str(root / "reproduce/solver" / p) for p in ("scripts", "src")]
    from revision_runs import read_rows, write_rows

    refs = [r for r in read_rows(root / "calibration" / "error_comparison.csv") if r["mode"] == "full"]
    ids = [int(r["set_id"]) for r in refs]
    assert len(ids) == len(set(ids)) == 10
    results = [evaluate((root, sid)) for sid in ids]
    rows = []
    for ref, (sid, scores, _, _) in zip(refs, results):
        assert int(ref["set_id"]) == sid
        rows.append({"set_id": sid, "n_points": 83, "dt_day": 1 / 6, **{
            f"{prefix}_{k}": value for k in scores for prefix, value in (
                ("legacy_form_old_parameters", float(ref[f"legacy_full_{k}"])),
                ("conservative_form_old_parameters", scores[k]),
                ("conservative_form_refitted_parameters", float(ref[f"train_{k}"])),
            )
        }})
    out = Path(__file__).resolve().parent
    write_rows(out / "error_comparison.csv", rows)
    write_rows(out / "old_parameters_conservative_trajectories.csv", [r for item in results for r in item[2]])
    write_rows(out / "tau_mapping.csv", [item[3] for item in results])


if __name__ == "__main__":
    main()
