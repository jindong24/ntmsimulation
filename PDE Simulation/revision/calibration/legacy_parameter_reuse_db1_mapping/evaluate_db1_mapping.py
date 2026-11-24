"""Replay the original full-fit parameters with mucus-side tauB mapping."""

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
    tissue_mapped = map_legacy_exchange(params)
    mapped = dict(tissue_mapped, tauB=params["DB1"] * params["tauB"])
    assert all(params[k] == mapped[k] for k in params if k not in ("tauB", "tauM"))
    assert all(mapped[k] == tissue_mapped[k] for k in mapped if k != "tauB")
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
    mapping = dict(set_id=sid, DB1=params["DB1"], DB2=params["DB2"], DM=params["DM"],
                   tauB_old=params["tauB"], tauM_old=params["tauM"],
                   tauB_applied_DB1=mapped["tauB"], tauB_applied_DB2=tissue_mapped["tauB"],
                   tauM_applied=mapped["tauM"])
    print(f"{sid}: B={score['B_MSRE']:.6g}, M={score['M_MSRE']:.6g}", flush=True)
    return sid, score, trajectory, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    root = args.study.resolve()
    sys.path[:0] = [str(root / "reproduce/solver" / p) for p in ("scripts", "src")]
    from revision_runs import read_rows, write_rows

    baseline = root / "calibration" / "legacy_parameter_reuse"
    refs = read_rows(baseline / "error_comparison.csv")
    ref_mapping = {int(r["set_id"]): r for r in read_rows(baseline / "tau_mapping.csv")}
    ids = [int(r["set_id"]) for r in refs]
    assert len(ids) == len(set(ids)) == 10
    results = [evaluate((root, sid)) for sid in ids]
    rows = []
    for ref, (sid, scores, _, mapping) in zip(refs, results):
        assert int(ref["set_id"]) == sid
        assert int(ref["n_points"]) == 83 and float(ref["dt_day"]) == 1 / 6
        assert mapping["tauB_applied_DB2"] == float(ref_mapping[sid]["tauB_applied"])
        assert mapping["tauM_applied"] == float(ref_mapping[sid]["tauM_applied"])
        rows.append({"set_id": sid, "n_points": 83, "dt_day": 1 / 6, **{
            f"{prefix}_{k}": value for k in scores for prefix, value in (
                ("legacy_form", float(ref[f"legacy_form_old_parameters_{k}"])),
                ("conservative_DB2_mapping", float(ref[f"conservative_form_old_parameters_{k}"])),
                ("conservative_DB1_mapping", scores[k]),
            )
        }})
    out = Path(__file__).resolve().parent
    write_rows(out / "error_comparison.csv", rows)
    write_rows(out / "db1_mapping_trajectories.csv", [r for item in results for r in item[2]])
    write_rows(out / "tau_mapping.csv", [item[3] for item in results])


if __name__ == "__main__":
    main()
