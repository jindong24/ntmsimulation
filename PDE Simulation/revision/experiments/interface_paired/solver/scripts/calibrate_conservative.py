#!/usr/bin/env python3
"""Pilot calibration and temporal validation with conservative exchange."""

import argparse
import csv
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lung_pde.constants import CALIBRATION_DT, FIXED_KEYS, SUPPORTED_SET_IDS


# Common seeds and bounds for both fits. No fitted XML or ABM-derived
# end-of-run growth bounds are used for these fifteen parameters.
# Entries are (seed, lower bound, upper bound); all except alpha1 use log10.
SEARCH = {
    "DB1": (0.005, 1e-9, 1.0),
    "DB2": (0.005, 1e-9, 1.0),
    "lambdaB": (0.2, 1e-5, 10.0),
    "KB": (0.01, 1e-9, 1.0),
    "KBi": (1e-4, 1e-9, 1.0),
    "dMb": (1e-4, 1e-8, 10.0),
    "dB": (0.01, 1e-9, 1.0),
    "kappaB": (0.001, 1e-9, 1.0),
    "DM": (0.001, 1e-9, 1.0),
    "delta": (1e-4, 1e-9, 1.0),
    "dM": (0.1, 1e-9, 1.0),
    "kappaM": (0.01, 1e-9, 1.0),
    "M0": (0.01, 1e-4, 0.75),
    "alpha0": (0.001, 1e-9, 100.0),
    "alpha1": (0.1, 0.0, 100.0),
}


def fit_count(times, mode, training_days=11.0):
    if mode == "full":
        return len(times)
    if mode == "half":
        return (len(times) + 1) // 2
    if mode == "partial":
        if not np.isfinite(training_days):
            raise ValueError("training days must be finite")
        count = int(np.searchsorted(times, training_days + 1e-12, side="right"))
        if not 0 < count < len(times):
            raise ValueError("cutoff must leave both training and holdout observations")
        return count
    raise ValueError(f"unknown fit mode: {mode}")


def concentration_weights(model):
    from dolfin import TestFunctions, assemble
    from lung_pde.mesh import MUCUS_FLAG, TISSUE_FLAG

    tests = TestFunctions(model.w.function_space())
    regions = (MUCUS_FLAG, TISSUE_FLAG) * 2
    return 50.0 * np.stack([
        assemble(test * model.mesh.dx(region)).get_local()
        for test, region in zip(tests[:4], regions)
    ])


def predict(model, parameters, times, weights):
    model.reset()
    model.assign_parameters(parameters)
    result = []
    for time in times:
        model.step(time)
        result.append(weights @ model.w.vector().get_local())
    values = np.array(result)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("non-finite PDE trajectory")
    return values


def scores(predicted, observed):
    error = predicted - observed
    return {
        name: {
            "rmse": float(np.sqrt(np.mean(error[:, i] ** 2))),
            "relative_rmse_pct": float(
                100 * np.sqrt(np.mean((error[:, i] / observed[:, i]) ** 2))
            ),
            "relative_l2_pct": float(
                100 * np.linalg.norm(error[:, i]) / np.linalg.norm(observed[:, i])
            ),
        }
        for i, name in enumerate(("bacteria", "macrophages"))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=int, choices=SUPPORTED_SET_IDS, default=756)
    parser.add_argument("--fit", choices=("full", "partial", "half"), required=True)
    parser.add_argument("--training-days", type=float, default=11.0,
                        help="Partial fit cutoff and full fit comparison cutoff.")
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--max-nfev", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from dolfin import Form
    from lung_pde.model import build_model

    model = build_model(
        args.set, "calibration", resolution=args.resolution, mesh_backend="rectangle",
        interface_permeabilities=(SEARCH["kappaB"][0], SEARCH["kappaM"][0]),
    )
    model.solver.parameters["absolute_tolerance"] = 1e-13
    model.solver.parameters["relative_tolerance"] = 1e-8
    model.problem.residual = Form(model.form.residual)
    model.problem.jacobian = Form(model.form.jacobian)
    weights = concentration_weights(model)
    np.testing.assert_allclose(weights @ model.w.vector().get_local(),
                               model.concentrations()[:4], atol=1e-15)
    times = model.abm.times
    count = fit_count(times, args.fit, args.training_days)
    comparison_mode = "half" if args.fit == "half" else "partial"
    cutoff = fit_count(times, comparison_mode, args.training_days)
    train_times = times[:count]
    observations = np.column_stack((model.abm.bacteria, model.abm.macrophages))
    train = observations[:count]
    if np.any(train <= 0):
        raise ValueError("relative residuals require positive training observations")
    args.output.mkdir(parents=True, exist_ok=False)

    names = tuple(SEARCH)
    use_log = np.array([name != "alpha1" for name in names])
    start, lower, upper = np.array(list(SEARCH.values()), dtype=float).T.copy()
    for vector in (start, lower, upper):
        vector[use_log] = np.log10(vector[use_log])

    def unpack(vector):
        values = vector.copy()
        values[use_log] = 10 ** values[use_log]
        return dict(zip(names, map(float, values)))

    best_loss = np.inf
    best_parameters = None
    failed = 0
    evaluations = 0
    started = monotonic()
    print(f"set={args.set}, fit={args.fit}, train={count}, last_train_day={train_times[-1]:g}", flush=True)

    def residual(vector):
        nonlocal best_loss, best_parameters, failed, evaluations
        parameters = unpack(vector)
        evaluations += 1
        try:
            predicted = predict(model, parameters, train_times, weights)[:, (0, 2)]
        except RuntimeError:
            if best_parameters is None:
                raise
            failed += 1
            return np.full(train.size, 1000.0)
        relative = (predicted - train) / train
        loss = float(np.mean(relative ** 2))
        if loss < best_loss:
            best_loss, best_parameters = loss, parameters
        if evaluations == 1 or evaluations % 50 == 0:
            elapsed = monotonic() - started
            print(f"{args.fit} evaluations={evaluations}, best relative RMSE={100*np.sqrt(best_loss):.3f}%, elapsed={elapsed:.1f}s", flush=True)
        return relative.ravel()

    result = least_squares(
        residual, start, bounds=(lower, upper), x_scale="jac",
        diff_step=1e-4, max_nfev=args.max_nfev, ftol=1e-7, gtol=1e-7, xtol=1e-7,
    )
    prediction = predict(model, best_parameters, times, weights)
    mucus = prediction[:, (0, 2)]
    persistence = np.broadcast_to(observations[cutoff - 1], mucus[cutoff:].shape)
    report = {
        "set": args.set, "fit": args.fit, "interface_law": "conservative",
        "mesh_backend": "rectangle", "resolution": args.resolution,
        "dt": CALIBRATION_DT, "train_count": count, "evaluation_count": len(times) - count,
        "last_train_day": float(train_times[-1]), "last_observation_day": float(times[-1]),
        "comparison_cutoff_day": float(times[cutoff - 1]),
        "objective_evaluations": evaluations, "failed_pde_evaluations": failed,
        "optimizer_converged": bool(result.success), "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev), "max_nfev": args.max_nfev,
        "elapsed_seconds": monotonic() - started,
        "seed_and_bounds": SEARCH,
        "fixed_parameters": {name: model.parameter_values[name] for name in FIXED_KEYS},
        "fitted_parameters": best_parameters,
        "training": scores(mucus[:count], train),
        "through_cutoff": scores(mucus[:cutoff], observations[:cutoff]),
        "after_cutoff": {
            "kind": "in_sample" if args.fit == "full" else "holdout",
            "count": len(times) - cutoff,
            "first_day": float(times[cutoff]),
            "errors": scores(mucus[cutoff:], observations[cutoff:]),
            "persistence_errors": scores(persistence, observations[cutoff:]),
        },
    }
    with (args.output / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "day", "used_for_fit", "B_ABM", "M_ABM", "B_mucus", "B_tissue",
            "M_mucus", "M_tissue",
        ))
        for i, (t, obs, values) in enumerate(zip(times, observations, prediction)):
            writer.writerow((t, i < count, *obs, *values))
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.fit}: {result.message}\noutput={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
