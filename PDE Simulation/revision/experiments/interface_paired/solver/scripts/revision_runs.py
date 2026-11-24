#!/usr/bin/env python3
"""Run the conservative-interface revision experiments without overwriting earlier runs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import numpy as np

from calibrate_conservative import ROOT, SEARCH, concentration_weights
from lung_pde.constants import FIXED_KEYS, SUPPORTED_SET_IDS


FIELDS = ("B_MUCUS", "B_TISSUE", "M_MUCUS", "M_TISSUE")
PARAMETERS = (*SEARCH, *FIXED_KEYS)


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(".tmp")
    with scratch.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scratch.replace(path)


def load_best():
    source = ROOT / "outputs/conservative/latest_best/best_full_parameters.csv"
    rows = read_rows(source)
    result = {int(r["set_id"]): {k: float(r[k]) for k in PARAMETERS} for r in rows}
    if set(result) != set(SUPPORTED_SET_IDS):
        raise ValueError("full-calibration set IDs do not match")
    return result


def make_model(sid, mode, parameters, resolution, dt, *, initial_loads=None):
    from dolfin import Form, LogLevel, set_log_level
    from lung_pde.model import build_model

    model = build_model(
        sid, mode, resolution=resolution, mesh_backend="rectangle", parameter_source="initial",
        dt=dt, interface_permeabilities=(parameters["kappaB"], parameters["kappaM"]),
        initial_loads=initial_loads,
    )
    model.assign_parameters(parameters)
    model.problem.residual = Form(model.form.residual)
    model.problem.jacobian = Form(model.form.jacobian)
    model.solver.parameters["absolute_tolerance"] = 1e-13
    model.solver.parameters["relative_tolerance"] = 1e-7
    model.solver.parameters["maximum_iterations"] = 35
    set_log_level(LogLevel.ERROR)
    return model


def final_or_series(model, time, weights, series=False):
    model.reset()
    records = [weights @ model.w.vector().get_local()] if series else []
    for t in time[1:]:
        model.step(float(t))
        if series:
            records.append(weights @ model.w.vector().get_local())
    out = np.array(records) if series else weights @ model.w.vector().get_local()
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite output concentration")
    return out


def solve(model, time, weights, series=False):
    try:
        return final_or_series(model, time, weights, series)
    except RuntimeError:
        # Same equations and time grid, with a direct linear solve as fallback.
        from dolfin import NewtonSolver

        previous = model.solver
        model.solver = NewtonSolver()
        model.solver.parameters["report"] = False
        model.solver.parameters["linear_solver"] = "lu"
        model.solver.parameters["absolute_tolerance"] = 1e-13
        model.solver.parameters["relative_tolerance"] = 1e-7
        model.solver.parameters["maximum_iterations"] = 60
        try:
            return final_or_series(model, time, weights, series)
        finally:
            model.solver = previous


def velocity_task(job):
    from lung_pde.results import save_simulation_result
    from lung_pde.simulation import run_simulation

    sid, mode, params, out, resolution = job
    model = make_model(sid, mode, params, resolution, 0.01)
    out = out / "velocity" / f"set_{sid:03d}" / mode
    config = replace(model.config, final_time=2., snapshot_times=(0., 1., 2.))
    result = run_simulation(model, config)
    result.metadata.update(interface_law="conservative", parameters=params)
    save_simulation_result(result, out)
    keys = ("time", "bacteria_mucus", "bacteria_tissue", "macrophage_mucus", "macrophage_tissue")
    write_rows(out / "concentrations.csv",
               [dict(zip(keys, row)) for row in zip(*(getattr(result, key) for key in keys))])
    return f"velocity set={sid} mode={mode}"


def treatment_task(job):
    sid, number, cases, params, out, resolution, dt = job
    path = out / "treatment" / "chunks" / f"{sid}_{number:02d}.csv"
    if path.exists():
        return f"resume set={sid} chunk={number}"
    model = make_model(sid, "treatment", params, resolution, dt)
    weights = concentration_weights(model)
    time = np.arange(round(14 / dt) + 1) * dt
    b0, m0 = model.abm.initial_bacteria, model.abm.initial_macrophages
    dose_factor = m0 * params["dB"]
    if not math.isfinite(dose_factor) or dose_factor <= 0.0:
        raise ValueError("effective danti requires positive M_initial*dB")
    rows = []
    for eta, dose in cases:
        model.eta.assign(eta)
        multiplier = dose / dose_factor
        model.danti.assign(multiplier)
        final = solve(model, time, weights)
        rows.append(dict(zip(FIELDS, final)) | {
            "eta": eta, "danti": dose, "B0": b0, "M0": m0,
            "danti_input": multiplier, "dB": params["dB"], "time_day": 14.,
        })
    write_rows(path, rows)
    return f"treatment set={sid} chunk={number} n={len(rows)}"


def treatment_jobs(args, best):
    jobs, designs = [], {}
    for sid in args.sets:
        raw = read_rows(args.reference / f"{sid}_new.csv")
        cases = sorted({(float(row["eta"]), float(row["danti"])) for row in raw})
        if len(cases) != 170 or len(raw) != 170:
            raise ValueError(f"expected 169 positive grid points and one reference for {sid}")
        if args.pilot:
            cases = [min(cases), max(cases), (1.0, 0.0),
                     max(pair for pair in cases if pair[0] == min(cases)[0])]
        designs[sid] = cases
        for i in range(0, len(cases), 17):
            jobs.append((sid, i // 17, cases[i:i + 17], best[sid],
                         args.output, args.resolution, args.dt))
    return jobs, designs


def pool_map(func, jobs, workers):
    if workers == 1:
        for job in jobs:
            print(func(job), flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(func, jobs):
            print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("velocity", "treatment"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--sets", type=int, nargs="+", choices=SUPPORTED_SET_IDS, default=SUPPORTED_SET_IDS)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--dt", type=float, default=.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    best = load_best()
    if args.experiment == "velocity":
        jobs = [(sid, mode, best[sid], args.output, args.resolution)
                for sid in args.sets for mode in ("baseline", "velocity")]
        pool_map(velocity_task, jobs, args.workers)
        sources = read_rows(ROOT / "outputs/conservative/latest_best/best_fit_summary.csv")
        for row in sources:
            sid = int(row["set_id"])
            if sid in args.sets:
                write_rows(args.output / "velocity" / f"set_{sid:03d}/calibration.csv",
                           read_rows(ROOT / row["full_trajectory"]))
    else:
        if args.reference is None:
            parser.error("--reference is required for the treatment dose grid")
        jobs, designs = treatment_jobs(args, best)
        pool_map(treatment_task, jobs, args.workers)
        for sid, design in designs.items():
            records = []
            for path in sorted((args.output / "treatment/chunks").glob(f"{sid}_*.csv")):
                records.extend(read_rows(path))
            obtained = sorted((float(r["eta"]), float(r["danti"])) for r in records)
            if obtained != sorted(design):
                raise ValueError(f"output treatment grid differs for {sid}")
            write_rows(args.output / "treatment/data" / f"{sid}_new.csv", records)
    metadata = {"resolution": args.resolution, "mesh": "rectangle",
                "interface_law": "conservative", "sets": list(args.sets),
                "dt": 0.01 if args.experiment == "velocity" else args.dt}
    (args.output / args.experiment / "batch_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
