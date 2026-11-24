"""Compute a bounded calibration-objective profile for DB2 in dataset 335."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

from run_identifiability import calibration, write_csv

GRID = (0.08, 0.15, 0.30, 0.60, 1.0)


def fixed_fit(job):
    study, fixed, budget = job
    solver = study / "reproduce" / "solver"
    sys.path[:0] = [str(solver / "scripts"), str(solver / "src")]
    os.environ.setdefault("NTM_ORIGINAL_MESH", str(solver.parent / "mesh" / "original_mshr.h5"))
    from dolfin import LogLevel, set_log_level

    set_log_level(LogLevel.WARNING)
    metrics, _, times, obs, names, q, bounds, _, tangent = calibration(study, 335)
    i = names.index("DB2")
    free = np.arange(len(q)) != i
    q[i] = np.log10(fixed)
    r, j = tangent.relative_objective(obs)

    def expand(x):
        z = q.copy()
        z[free] = x
        return z

    initial = r(q)
    result = least_squares(lambda x: r(expand(x)), q[free],
                           jac=lambda x: j(expand(x))[:, free],
                           bounds=(bounds[free, 0], bounds[free, 1]), x_scale="jac",
                           max_nfev=budget, ftol=1e-7, xtol=1e-7, gtol=1e-7)
    solution = expand(result.x)
    data, _ = tangent.predict(solution)
    relative = (data[:, (0, 2)] - obs) / obs
    b, m = np.mean(relative**2, axis=0)
    np.testing.assert_allclose(relative.ravel(), result.fun, atol=2e-9, rtol=1e-7)
    initial_loss = initial @ initial / len(times)
    assert b + m <= initial_loss + 1e-11
    params = tangent.unpack(solution)
    assert np.isclose(params["DB2"], fixed)
    final = dict(DB2=fixed, joint_MSRE_initial=initial_loss, joint_MSRE=b+m,
                 B_MSRE=b, M_MSRE=m, final_B_tissue=data[-1, 1], final_M_tissue=data[-1, 3],
                 last_day=times[-1], converged=result.success, status=result.status,
                 nfev=result.nfev, first_order_optimality=result.optimality,
                 termination=result.message)
    print(f"DB2={fixed:g}: MSRE={b+m:.8g}, nfev={result.nfev}, "
          f"converged={result.success}", flush=True)
    return final, params, times, data, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = [(args.study.resolve(), d, args.budget) for d in GRID]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fits = list(pool.map(fixed_fit, jobs))
    summary, parameters, trajectories = [], [], []
    for row, params, times, data, metrics in fits:
        summary.append(row)
        parameters.append(params)
        trajectories.extend(dict(DB2=row["DB2"], day=t, B_mucus=b_m, B_tissue=b_t,
                                 M_mucus=m_m, M_tissue=m_t)
                            for t, (b_m, b_t, m_m, m_t) in zip(times, data, strict=True))
    write_csv(args.output / "profile_summary.csv", summary)
    write_csv(args.output / "profile_parameters.csv", parameters)
    write_csv(args.output / "profile_trajectories.csv", trajectories)
    (args.output / "settings.json").write_text(json.dumps(
        dict(set_id=335, fixed_parameter="DB2", grid=GRID, evaluation_budget=args.budget,
             reference_checkpoint=metrics["fitted_parameters"],
             reference_training=metrics["training"], dt=metrics["dt"],
             n_training=metrics["train_count"]), indent=2) + "\n")


if __name__ == "__main__":
    main()
