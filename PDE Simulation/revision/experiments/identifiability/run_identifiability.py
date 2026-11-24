"""Diagnose identifiability at saved original-mesh calibration checkpoints."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from time import monotonic

import numpy as np

IDS = (48, 83, 182, 188, 335, 586, 646, 756, 808, 843)
RTOLS = (1e-4, 1e-6, 1e-8)


def diagnose(j):
    """Use the residual Jacobian rather than squaring its condition number."""
    _, sv, vt = np.linalg.svd(j, full_matrices=False)
    cutoff = sv[0] * RTOLS[-1]
    independent = np.empty_like(j)
    for i, column in enumerate(j.T):
        u, other_sv, _ = np.linalg.svd(np.delete(j, i, axis=1), full_matrices=False)
        basis = u[:, other_sv > cutoff]
        independent[:, i] = column - basis @ (basis.T @ column)
    return sv, vt, independent


def self_test():
    j = np.array([[2., 4., 0.], [0., 0., 3.], [0., 0., 0.]])
    sv, vt, independent = diagnose(j)
    np.testing.assert_allclose(sv, [np.sqrt(20), 3, 0], atol=1e-14)
    np.testing.assert_allclose(j @ vt[-1], 0, atol=1e-14)
    np.testing.assert_allclose(independent[:, :2], 0, atol=1e-14)
    np.testing.assert_allclose(independent[:, 2], j[:, 2], atol=1e-14)


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def calibration(study, sid):
    from dolfin import Form
    from calibration_tangent import CalibrationTangent
    from calibrate_conservative import SEARCH, concentration_weights
    from lung_pde.model import build_model

    path = study / "calibration" / "full" / f"set_{sid:03d}"
    metrics = json.loads((path / "metrics.json").read_text())
    assert metrics["set"] == sid and metrics["fit"] == "full"
    assert metrics["interface_law"] == "conservative"
    assert metrics["mesh_backend"] == "original_mshr"
    names = tuple(SEARCH)
    assert len(names) == 15
    assert all(tuple(metrics["seed_and_bounds"][k][1:]) == SEARCH[k][1:] for k in names)
    params = metrics["fitted_parameters"]
    model = build_model(sid, "calibration", resolution=metrics["resolution"],
                        mesh_backend="original_mshr",
                        original_mesh=Path(os.environ["NTM_ORIGINAL_MESH"]),
                        conservative_exchange=True)
    assert np.isclose(float(model.dt), metrics["dt"], rtol=0, atol=1e-15)
    assert model.mesh.mesh.num_vertices() == metrics["mesh_vertices"]
    assert model.mesh.mesh.num_cells() == metrics["mesh_cells"]
    model.solver.parameters["absolute_tolerance"] = 1e-13
    model.solver.parameters["relative_tolerance"] = 1e-8
    model.problem.residual = Form(model.form.residual)
    model.problem.jacobian = Form(model.form.jacobian)
    model.assign_parameters(metrics["fixed_parameters"])
    weights = concentration_weights(model)
    np.testing.assert_allclose(weights @ model.w.vector().get_local(),
                               model.concentrations()[:4], atol=1e-15)
    n = metrics["train_count"]
    times = model.abm.times[:n]
    assert len(model.abm.times) == n
    observed = np.column_stack((model.abm.bacteria[:n], model.abm.macrophages[:n]))
    assert n == 83 and observed.shape == (n, 2) and (observed > 0).all()
    log_cols = np.array([key != "alpha1" for key in names])
    q = np.array([params[key] for key in names])
    bounds = np.array([SEARCH[key][1:] for key in names], dtype=float)
    q[log_cols] = np.log10(q[log_cols])
    bounds[log_cols] = np.log10(bounds[log_cols])
    tangent = CalibrationTangent(model, names, times, weights)
    return metrics, model, times, observed, names, q, bounds, weights, tangent


def check_tangent(model, tangent, times, observed, names, q, bounds, r, j, weights):
    from calibrate_conservative import predict

    rows = []
    for name in ("DB2", "tauM"):
        i = names.index(name)
        h = 1e-3
        step = np.zeros_like(q)
        step[i] = h

        def relative(z):
            values = predict(model, tangent.unpack(z), times, weights)[:, (0, 2)]
            return ((values - observed) / observed).ravel()

        if q[i] - h >= bounds[i, 0] and q[i] + h <= bounds[i, 1]:
            fd = (relative(q + step) - relative(q - step)) / (2*h)
            method = "central"
        else:
            sign = 1 if q[i] + 2*h <= bounds[i, 1] else -1
            assert bounds[i, 0] <= q[i] + sign*2*h <= bounds[i, 1]
            fd = (-3*r + 4*relative(q + sign*step) - relative(q + sign*2*step)) / (sign*2*h)
            method = "forward" if sign == 1 else "backward"
        relerr = np.linalg.norm(fd - j[:, i]) / np.linalg.norm(j[:, i])
        rows.append(dict(set_id=335, parameter=name, method=method, step_log10=h,
                         tangent_norm=np.linalg.norm(j[:, i]),
                         finite_difference_norm=np.linalg.norm(fd),
                         relative_l2_error=relerr))
        assert relerr < 0.015, rows[-1]
    return rows


def run(study, output, ids):
    from dolfin import LogLevel, set_log_level

    set_log_level(LogLevel.WARNING)
    self_test()
    output.mkdir(parents=True, exist_ok=False)
    summary, spectrum, coordinates, directions, tangent_checks = [], [], [], [], []
    for sid in ids:
        t0 = monotonic()
        metrics, model, times, obs, names, q, bounds, weights, tangent = calibration(study, sid)
        residual, jacobian = tangent.relative_objective(obs)
        r, j = residual(q), jacobian(q)
        assert j.shape == (166, 15) and np.isfinite(j).all()
        msre = np.mean(r.reshape(-1, 2)**2, axis=0)
        saved = [(metrics["training"][k]["relative_rmse_pct"]/100)**2
                 for k in ("bacteria", "macrophages")]
        np.testing.assert_allclose(msre, saved, atol=1e-12, rtol=1e-6)
        sv, vt, independent = diagnose(j)
        ratios = sv / sv[0]
        ranks = [int(np.sum(ratios > rtol)) for rtol in RTOLS]
        near = (q[:, None] - bounds) / np.diff(bounds, axis=1)
        bound = np.where(abs(near[:, 0]) < 1e-5, "lower",
                         np.where(abs(near[:, 1]) < 1e-5, "upper", ""))
        np.savez_compressed(output / f"set_{sid:03d}.npz",
                            parameter_names=names, optimizer_vector=q,
                            optimizer_bounds=bounds, observations=obs, times=times,
                            residual=r, jacobian=j, information=j.T @ j,
                            singular_values=sv, right_singular_vectors=vt,
                            independent_components=independent)
        summary.append(dict(set_id=sid, B_MSRE=msre[0], M_MSRE=msre[1],
                            joint_MSRE=msre.sum(), fit_converged=metrics["optimizer_converged"],
                            rank_rtol_1e_minus_4=ranks[0], rank_rtol_1e_minus_6=ranks[1],
                            rank_rtol_1e_minus_8=ranks[2], sigma_max=sv[0],
                            sigma_min_over_max=ratios[-1], lambda_min_over_max=ratios[-1]**2,
                            near_bound=";".join(f"{k}:{v}" for k, v in zip(names, bound) if v)))
        for i, ratio in enumerate(ratios):
            spectrum.append(dict(set_id=sid, mode=i+1, sigma=sv[i],
                                 sigma_over_max=ratio, lambda_over_max=ratio**2))
        for i, k in enumerate(names):
            norm = np.linalg.norm(j[:, i])
            remaining = np.linalg.norm(independent[:, i])
            coordinates.append(dict(set_id=sid, parameter=k, fitted_value=metrics["fitted_parameters"][k],
                                    optimizer_coordinate="linear" if k == "alpha1" else "log10",
                                    near_bound=bound[i], sensitivity_l2=norm,
                                    independent_linf=np.max(np.abs(independent[:, i])),
                                    independent_l2=remaining,
                                    independent_fraction_l2=remaining/norm if norm else 0.0))
        for i in range(15):
            for k, coefficient in zip(names, vt[i], strict=True):
                directions.append(dict(set_id=sid, mode=i+1, sigma_over_max=ratios[i],
                                       parameter=k, coefficient=coefficient))
        if sid == 335:
            tangent_checks.extend(check_tangent(model, tangent, times, obs, names, q, bounds, r, j, weights))
        print(f"{sid}: joint MSRE={msre.sum():.6g}; ranks={ranks}; "
              f"sigma_min/sigma_max={ratios[-1]:.3g}; {monotonic()-t0:.1f} s", flush=True)
    for name, rows in (("summary", summary), ("singular_values", spectrum),
                       ("parameter_diagnostics", coordinates), ("parameter_directions", directions),
                       ("tangent_check", tangent_checks)):
        write_csv(output / f"{name}.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sets", type=int, nargs="+", choices=IDS, default=IDS)
    args = parser.parse_args()
    study = args.study.resolve()
    solver = study / "reproduce" / "solver"
    sys.path[:0] = [str(solver / "scripts"), str(solver / "src")]
    os.environ.setdefault("NTM_ORIGINAL_MESH", str(solver.parent / "mesh" / "original_mshr.h5"))
    run(study, args.output, args.sets)


if __name__ == "__main__":
    main()
