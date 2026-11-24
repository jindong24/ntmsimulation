#!/usr/bin/env python3
"""Replay identical inputs with the original and conservative interface laws."""

import argparse
import math
from pathlib import Path

import numpy as np

from calibrate_conservative import concentration_weights
from revision_runs import FIELDS, make_model, pool_map, read_rows, write_rows
from revision_sensitivity import INITIAL_LOADS, TRANSFER, flux_forms
from lung_pde.constants import REQUIRED_PARAMETER_KEYS


def mapped_parameters(original):
    return {
        **{k: v for k, v in original.items() if k not in ("tauB", "tauM")},
        "kappaB": original["DB2"] * original["tauB"],
        "kappaM": original["DM"] * original["tauM"],
    }


def original_model(resolution, dt, mode="treatment", initial_loads=INITIAL_LOADS, set_id=756):
    from dolfin import Form, LogLevel, set_log_level
    from lung_pde.model import build_model

    model = build_model(
        set_id, mode, resolution=resolution, mesh_backend="rectangle",
        parameter_source="initial", dt=dt, initial_loads=initial_loads,
    )
    model.problem.residual = Form(model.form.residual)
    model.problem.jacobian = Form(model.form.jacobian)
    model.solver.parameters["absolute_tolerance"] = 1e-13
    model.solver.parameters["relative_tolerance"] = 1e-7
    model.solver.parameters["maximum_iterations"] = 35
    set_log_level(LogLevel.ERROR)
    return model


def original_flux_forms(model):
    from dolfin import Form, as_vector, avg, dot, exp, grad, split
    from ufl import max_value
    from lung_pde.mesh import INTERFACE_FLAG, RIGHT_MUCUS_FLAG

    bm, bt, mm, mt, mucin = split(model.w)
    p = model.parameters
    normal = model.mesh.normal
    tissue_outward = as_vector((0.0, 1.0))
    scale = exp(1.0 - avg(mucin) / p["m0"]) / model.eta
    dn_bt = sum(dot(grad(bt(side)), normal(side)) *
                (1.0 + dot(normal(side), tissue_outward)) / 2.0
                for side in ("+", "-"))
    # Positive exchange enters the tissue; the two legacy fluxes need not balance.
    jb = scale * p["DB2"] * p["tauB"] * (avg(bm) - avg(bt))
    jm = scale * p["DM"] * p["tauM"] * (avg(mm) - avg(mt)) - p["delta"] * dn_bt * avg(mt)
    right = model.mesh.ds(RIGHT_MUCUS_FLAG)
    interface = model.mesh.dS(INTERFACE_FLAG)
    return tuple(Form(form) for form in (
        0.5 * bm * right, 0.5 * mm * right,
        *(50 * max_value(j, 0.0) * interface for j in (jb, jm, -jb, -jm)),
    ))


def trajectory(model, time, weights, fluxes):
    from dolfin import assemble

    model.reset()
    concentration = np.empty((len(time), 4))
    cumulative = np.zeros((len(time), 8))
    concentration[0] = (weights @ model.w.vector().get_local())[:4]
    for i, moment in enumerate(time[1:], 1):
        model.step(float(moment))
        concentration[i] = (weights @ model.w.vector().get_local())[:4]
        cumulative[i, :6] = cumulative[i - 1, :6] + float(time[1]) * np.array(
            [assemble(form) for form in fluxes]
        )
        cumulative[i, 6:] = cumulative[i, 2:4] - cumulative[i, 4:6]
    if not (np.isfinite(concentration).all() and np.isfinite(cumulative).all()):
        raise RuntimeError("non-finite trajectory")
    return concentration, cumulative


def with_retry(model, time, weights, fluxes, label):
    from dolfin import NewtonSolver

    try:
        return (*trajectory(model, time, weights, fluxes), False)
    except RuntimeError as exc:
        print(f"{label}: direct linear solver retry: {exc}", flush=True)
        old_solver = model.solver
        model.solver = NewtonSolver()
        model.solver.parameters["report"] = False
        model.solver.parameters["linear_solver"] = "lu"
        model.solver.parameters["absolute_tolerance"] = 1e-13
        model.solver.parameters["relative_tolerance"] = 1e-7
        model.solver.parameters["maximum_iterations"] = 70
        try:
            return (*trajectory(model, time, weights, fluxes), True)
        finally:
            model.solver = old_solver


def run_chunk(job):
    records, root, resolution, dt = job
    time = np.arange(round(14 / dt) + 1) * dt
    a = original_model(resolution, dt)
    old = {k: float(records[0][k]) for k in REQUIRED_PARAMETER_KEYS}
    b = make_model(756, "treatment", mapped_parameters(old), resolution, dt, initial_loads=INITIAL_LOADS)
    structures = [
        (a, concentration_weights(a), original_flux_forms(a)),
        (b, concentration_weights(b), flux_forms(b)),
    ]
    for row in records:
        sid = int(row["sample_id"])
        path = root / "pairs" / f"sample_{sid:03d}.npz"
        if path.exists():
            continue
        original = {k: float(row[k]) for k in REQUIRED_PARAMETER_KEYS}
        data = {"time": time}
        for tag, (model, weights, fluxes), params in zip(
            ("legacy", "conservative"), structures, (original, mapped_parameters(original))
        ):
            model.assign_parameters(params)
            model.eta.assign(float(row["eta"]))
            model.danti.assign(float(row["danti"]))
            if not math.isclose(model.effective_danti, float(row["danti_effective"]), rel_tol=1e-12):
                raise ValueError(f"sample {sid}: effective danti mismatch")
            c, q, fallback = with_retry(model, time, weights, fluxes, f"{sid}/{tag}")
            data[f"{tag}_c"], data[f"{tag}_q"], data[f"{tag}_fallback"] = c, q, fallback
        np.testing.assert_allclose(data["legacy_c"][0], data["conservative_c"][0], rtol=1e-12, atol=1e-18)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as out:
            np.savez_compressed(out, **data)
        tmp.replace(path)
    return f"paired samples {records[0]['sample_id']}..{records[-1]['sample_id']}"


def export(root, records):
    for tag in ("legacy", "conservative"):
        rows = []
        for record in records:
            sid = int(record["sample_id"])
            with np.load(root / "pairs" / f"sample_{sid:03d}.npz") as pair:
                c = pair[f"{tag}_c"][-1]
                q = pair[f"{tag}_q"][-1]
                rows.append({
                    "sample_id": sid, "time_day": pair["time"][-1],
                    **dict(zip(FIELDS, c)),
                    "B_TOTAL": c[0] + c[1], "M_TOTAL": c[2] + c[3],
                    **dict(zip(TRANSFER, q)),
                })
        write_rows(root / tag / "endpoints.csv", rows)
    write_rows(root / "matched_inputs.csv", [
        {**r, "kappaB": float(r["DB2"]) * float(r["tauB"]),
         "kappaM": float(r["DM"]) * float(r["tauM"])}
        for r in records
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("design", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()
    records = read_rows(args.design)[:args.limit]
    ids = [int(r["sample_id"]) for r in records]
    if len(set(ids)) != len(ids) or not ids or not math.isclose(14 / args.dt, round(14 / args.dt)):
        raise ValueError("invalid IDs or time step")
    (args.root / "pairs").mkdir(parents=True, exist_ok=True)
    jobs = [(records[i:i + args.chunk_size], args.root, args.resolution, args.dt)
            for i in range(0, len(records), args.chunk_size)
            if any(not (args.root / "pairs" / f"sample_{int(r['sample_id']):03d}.npz").exists()
                   for r in records[i:i + args.chunk_size])]
    if jobs:
        pool_map(run_chunk, jobs, args.workers)
    export(args.root, records)


if __name__ == "__main__":
    main()
