#!/usr/bin/env python3
"""Replay the archived 400-sample design with conservative interface exchange."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t as student

from revision_runs import FIELDS, PARAMETERS, make_model, pool_map, read_rows, write_rows
from calibrate_conservative import concentration_weights


INITIAL_LOADS = (1.4091e-6, 0.001974684)
TRANSFER = ("B_CLEARANCE", "M_CLEARANCE", "B_M_TO_T", "M_M_TO_T",
            "B_T_TO_M", "M_T_TO_M", "B_NET_TO_T", "M_NET_TO_T")
SPATIAL = ("bacteria_mucus", "bacteria_tissue", "macrophage_mucus", "macrophage_tissue")
GEOMETRY = ("mucus_xy", "mucus_triangles", "tissue_xy", "tissue_triangles")
VARIED = tuple(k for k in PARAMETERS if k not in ("delta", "Dm", "dm", "m0", "beta0", "beta1")) + ("eta", "danti")


def flux_forms(model):
    from dolfin import Form, split
    from lung_pde.conservative_interface import interface_fluxes
    from lung_pde.mesh import INTERFACE_FLAG, RIGHT_MUCUS_FLAG
    from ufl import max_value

    b, _, m, _, _ = split(model.w)
    j_b, j_m = interface_fluxes(model.w, model.parameters, model.eta)
    right, interface = model.mesh.ds(RIGHT_MUCUS_FLAG), model.mesh.dS(INTERFACE_FLAG)
    # 50 is the same region-volume scaling as concentrations().
    return tuple(Form(form) for form in (
        0.5 * b * right, 0.5 * m * right,
        *(50 * max_value(j, 0.) * interface for j in (j_b, j_m, -j_b, -j_m)),
    ))


def snapshot_indices(model):
    from dolfin import vertex_to_dof_map

    snapshot = model.snapshot(0.)
    space = model.w.function_space()
    maps = {}
    for i, name in enumerate(SPATIAL):
        region = name.split("_")[-1]
        submesh = getattr(model.mesh, region + "_mesh")
        parent_vertices = submesh.data().array("parent_vertex_indices", 0)
        scalar, parent_dofs = space.sub(i).collapse(collapsed_dofs=True)
        parent_dofs = np.array([parent_dofs[j] for j in range(scalar.dim())])
        maps[name] = parent_dofs[vertex_to_dof_map(scalar)[parent_vertices]]
        actual = model.w.vector().get_local()[maps[name]]
        np.testing.assert_allclose(actual, getattr(snapshot, name)[0], atol=1e-14, rtol=1e-12)
    return maps, {k: getattr(snapshot, k) for k in GEOMETRY}


def run_sample(model, weights, forms, time, day_stride, spatial, kernel=None):
    from dolfin import assemble

    model.reset()
    state = model.w.vector().get_local()
    if kernel is not None:
        kernel.factor = None
    concentrations = [weights @ state]
    integrated = np.zeros(6)
    cumulative = [np.zeros(8)]
    indices, geometry = spatial
    captured = {k: [state[idx]] for k, idx in indices.items()}
    dt = time[1]
    for i, moment in enumerate(time[1:], start=1):
        if kernel is None:
            model.step(float(moment))
            state = model.w.vector().get_local()
        else:
            state = kernel.step(float(moment), state, lag=True)
            vector = model.w.vector()
            vector.set_local(state)
            vector.apply("insert")
        concentrations.append(weights @ state)
        integrated += dt * np.array([assemble(form) for form in forms])
        cumulative.append(np.r_[integrated, integrated[2:4] - integrated[4:6]])
        if i % day_stride == 0:
            for k, idx in indices.items():
                captured[k].append(state[idx])
    results = {
        "time": time, "daily_time": time[::day_stride],
        "concentrations": np.array(concentrations), "cumulative": np.array(cumulative),
        **{k: np.array(v) for k, v in captured.items()},
        **geometry,
    }
    if not all(np.isfinite(v).all() for v in results.values()):
        raise RuntimeError("non-finite result")
    return results


def sample_chunk(job):
    records, root, resolution, dt, backend, mean_viscosity = job
    first = {k: float(records[0][k]) for k in PARAMETERS}
    model = make_model(756, "treatment", first, resolution, dt, initial_loads=INITIAL_LOADS,
                       interpolate_profile=True, polynomial_velocity=backend == "ufc",
                       mean_viscosity_velocity=mean_viscosity)
    kernel = None
    if backend == "ufc":
        from lung_pde.ufc_batch import BatchNewton

        kernel = BatchNewton(model, os.environ["NTM_UFC_LIBRARY"])
    forms = flux_forms(model)
    weights = concentration_weights(model)
    spatial = snapshot_indices(model)
    day_stride = round(1 / dt)
    time = np.arange(14 * day_stride + 1) * dt
    for record in records:
        sid = int(record["sample_id"])
        path = root / "samples" / f"sample_{sid:03d}.npz"
        if path.exists():
            continue
        model.assign_parameters({k: float(record[k]) for k in PARAMETERS})
        model.eta.assign(float(record["eta"]))
        model.danti.assign(float(record["danti"]))
        if not math.isclose(model.effective_danti, float(record["danti_effective"]), rel_tol=1e-12):
            raise ValueError(f"effective antibiotic dose mismatch for {sid}")
        try:
            values = run_sample(model, weights, forms, time, day_stride, spatial, kernel=kernel)
        except RuntimeError as error:
            from dolfin import NewtonSolver

            print(f"sample {sid}: retrying with direct solve: {error}", flush=True)
            old = model.solver
            model.solver = NewtonSolver()
            model.solver.parameters["report"] = False
            model.solver.parameters["linear_solver"] = "lu"
            model.solver.parameters["absolute_tolerance"] = 1e-13
            model.solver.parameters["relative_tolerance"] = 1e-7
            model.solver.parameters["maximum_iterations"] = 70
            try:
                values = run_sample(model, weights, forms, time, day_stride, spatial)
            finally:
                model.solver = old
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **values)
        tmp.replace(path)
    return f"sensitivity samples {records[0]['sample_id']}..{records[-1]['sample_id']}"


def prcc(design, endpoints):
    features = np.array([[float(r[k]) for k in VARIED] for r in design])
    x = rankdata(features, axis=0)
    n, p = x.shape
    results = []
    for name in tuple(endpoints[0])[2:]:
        y = rankdata([r[name] for r in endpoints])
        for j, variable in enumerate(VARIED):
            controls = np.c_[np.ones(n), np.delete(x, j, axis=1)]
            pair = np.column_stack((x[:, j], y))
            residuals = pair - controls @ np.linalg.lstsq(controls, pair, rcond=None)[0]
            corr = np.corrcoef(residuals.T)[0, 1]
            df = n - p - 1
            statistic = abs(corr) * np.sqrt(df / max(1 - corr * corr, 1e-15))
            p_value = float(2 * student.sf(statistic, df))
            results.append({"outcome": name, "parameter": variable, "n": n, "PRCC": corr,
                            "p_raw": p_value, "p_Bonferroni_16_per_outcome": min(1., p * p_value)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resolution", type=int, choices=(32,), default=32)
    parser.add_argument("--dt", type=float, default=.1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--solver", choices=("cold", "ufc"), default="cold")
    parser.add_argument("--velocity", choices=("mean", "local"), default="mean",
                        help="use a separate output directory for each closure")
    args = parser.parse_args()
    if not math.isclose(round(1 / args.dt) * args.dt, 1.):
        raise ValueError("dt must divide one day")
    design = read_rows(args.root / "design.csv")
    if [int(row["sample_id"]) for row in design] != list(range(1, 401)):
        raise ValueError("expected original sample IDs 1..400")
    if args.limit:
        design = design[:args.limit]
    (args.root / "samples").mkdir(exist_ok=True)
    jobs = [(design[i:i + 10], args.root, args.resolution, args.dt, args.solver, args.velocity == "mean")
            for i in range(0, len(design), 10)
            if any(not (args.root / "samples" / f"sample_{int(r['sample_id']):03d}.npz").is_file()
                   for r in design[i:i + 10])]
    pool_map(sample_chunk, jobs, args.workers)
    endpoints = []
    for record in design:
        sid = int(record["sample_id"])
        with np.load(args.root / "samples" / f"sample_{sid:03d}.npz", allow_pickle=False) as file:
            if not np.allclose(file["daily_time"], np.arange(15)) or not np.isclose(file["time"][-1], 14):
                raise ValueError(f"unexpected time grid for sample {sid}")
            b, bt, m, mt = file["concentrations"][-1]
            flux = file["cumulative"][-1]
            endpoints.append(dict(sample_id=sid, time_day=14.) | dict(zip(FIELDS, (b, bt, m, mt)))
                             | dict(B_TOTAL=b + bt, M_TOTAL=m + mt) | dict(zip(TRANSFER, flux)))
    write_rows(args.root / "endpoints.csv", endpoints)
    if len(design) == 400:
        write_rows(args.root / "prcc_day14.csv", prcc(design, endpoints))
    (args.root / "simulation_metadata.json").write_text(json.dumps({
        "resolution": args.resolution, "mesh": "original_mshr", "dt_day": args.dt,
        "initial_loads_B_M": INITIAL_LOADS, "sample_count": len(design),
        "interface_law": "shared viscosity-scaled tauB and DM*tauM; tauM is the original sample value",
        "velocity_law": ("u0(y,t) divided by mucus-domain mean of eta0*exp(m/m_ref-1)"
                         if args.velocity == "mean" else "u0(y,t) divided by local eta(m)"),
        "solver": args.solver,
        "concentrations_order": FIELDS, "cumulative_order": TRANSFER,
        "cumulative_time_rule": "backward Euler, right-endpoint quadrature",
        "positive_transfer": "pointwise positive or negative part integrated on Gamma1",
        "clearance": "Robin boundary term 0.01 * concentration integrated on right mucus edge",
        "spatial": "field values at mesh vertices, indexed by mucus/tissue XY and triangles",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
