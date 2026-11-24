#!/usr/bin/env python3
"""Run the conservative-interface revision experiments without overwriting earlier runs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
import json
import math
import os
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


def load_best(root, ids):
    result = {}
    for sid in ids:
        report = json.loads((root / "calibration" / "full" / f"set_{sid:03d}" / "metrics.json").read_text())
        if report["set"] != sid or report["mesh_backend"] != "original_mshr" or report["train_count"] != 83:
            raise ValueError(f"expected original mesh full fit for {sid}")
        result[sid] = report["fixed_parameters"] | report["fitted_parameters"]
    return result


def legacy_parameters(root, sid, folder):
    from lung_pde.parameters import read_parameter_xml

    xml = read_parameter_xml(root / folder / f"parameters_{sid}.xml")
    return {name: float(xml[name]) for name in PARAMETERS}


def map_legacy_exchange(parameters):
    # Match the old tissue-side diffusive exchange, not the incompatible mucus-side coefficient.
    return dict(parameters, tauB=parameters["DB2"] * parameters["tauB"],
                tauM=parameters["DM"] * parameters["tauM"])


def make_model(sid, mode, parameters, resolution, dt, *, initial_loads=None,
               interpolate_profile=False, conservative=True, polynomial_velocity=False,
               mean_viscosity_velocity=False, macrophage_diffusivity_in_flux=True):
    from dolfin import Form, LogLevel, set_log_level
    from lung_pde.model import build_model

    model = build_model(
        sid, mode, resolution=resolution, mesh_backend="original_mshr", parameter_source="initial",
        dt=dt, conservative_exchange=conservative, original_mesh=Path(os.environ["NTM_ORIGINAL_MESH"]),
        initial_loads=initial_loads, interpolate_profile=interpolate_profile,
        polynomial_velocity=polynomial_velocity,
        mean_viscosity_velocity=mean_viscosity_velocity,
        macrophage_diffusivity_in_flux=macrophage_diffusivity_in_flux,
    )
    model.assign_parameters(parameters)
    model.problem.residual = Form(model.form.residual)
    model.problem.jacobian = Form(model.form.jacobian)
    model.solver.parameters["absolute_tolerance"] = 1e-13
    model.solver.parameters["relative_tolerance"] = 1e-7
    model.solver.parameters["maximum_iterations"] = 35
    if mode in {"velocity", "treatment"}:
        model.solver.parameters["krylov_solver"]["maximum_iterations"] = 300
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
    out = out / "velocity" / f"set_{sid:03d}" / mode
    if all((out / file).is_file() for file in ("concentrations.csv", "spatial.npz", "metadata.json")):
        return f"velocity set={sid} mode={mode}: already available"
    model = make_model(sid, mode, params, resolution, 0.01)
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
            "zeta": multiplier, "danti_input": multiplier, "dB": params["dB"], "time_day": 14.,
        })
    write_rows(path, rows)
    return f"treatment set={sid} chunk={number} n={len(rows)}"


def treatment_slice(job):
    from treatment_continuation import dose_trajectory

    sid, number, cases, params, root, resolution, dt, backend = job
    path = root / "slices" / f"{sid}_{number:02d}.csv"
    if path.exists():
        return sid, f"treatment set={sid} slice={number}: already available"
    eta = cases[0][0]
    model = make_model(sid, "treatment", params, resolution, dt, polynomial_velocity=backend == "ufc")
    model.eta.assign(eta)
    weights = concentration_weights(model)
    time = np.arange(round(14 / dt) + 1) * dt
    b0, m0 = model.abm.initial_bacteria, model.abm.initial_macrophages
    dose_factor = m0 * params["dB"]
    if not math.isfinite(dose_factor) or dose_factor <= 0.:
        raise ValueError(f"invalid danti conversion for {sid}")
    multipliers = [dose / dose_factor for _, dose in cases]
    values = []
    used = backend
    if backend == "ufc":
        from lung_pde.ufc_batch import BatchNewton

        kernel = BatchNewton(model, os.environ["NTM_UFC_LIBRARY"])
        try:
            values = kernel.slice_endpoints(time, weights, multipliers)
            if not np.isfinite(values).all():
                raise RuntimeError("non-finite compiled slice")
        except RuntimeError:
            values = []
            used = "dolfin_warm_fallback"
    if len(values) == 0:
        guesses = None
        for multiplier in multipliers:
            model.danti.assign(multiplier)
            try:
                end, guesses = dose_trajectory(model, time, weights, guesses)
            except RuntimeError:
                end = solve(model, time, weights)
                guesses = None
            values.append(end)
    rows = [
        dict(zip(FIELDS, final)) | {
            "eta": eta, "danti": dose, "B0": b0, "M0": m0,
            "zeta": z, "danti_input": z, "dB": params["dB"], "time_day": 14.,
        }
        for (_, dose), z, final in zip(cases, multipliers, values)
    ]
    write_rows(path, rows)
    return sid, f"treatment set={sid} eta={eta:g} n={len(rows)} ({used})"


def available_treatment(root, sid):
    rows = {}
    for path in (*sorted((root / "treatment/chunks").glob(f"{sid}_*.csv")),
                 *sorted((root / "slices").glob(f"{sid}_*.csv"))):
        for r in read_rows(path):
            key = (float(r["eta"]), float(r["danti"]))
            if key in rows:
                raise ValueError(f"duplicated completed treatment point {sid}: {key}")
            rows[key] = r
    return rows


def finish_treatment_set(root, sid, design):
    present = available_treatment(root, sid)
    if not set(present) <= set(design):
        raise ValueError(f"unexpected completed treatment point for {sid}")
    if len(present) < len(design):
        return False
    write_rows(root / "data" / f"{sid}_new.csv", [present[key] for key in design])
    return True


def treatment_slice_batch(args, best, root):
    from concurrent.futures import as_completed

    _, designs = treatment_jobs(args, best, root)
    jobs = []
    complete = set()
    for sid, design in designs.items():
        present = available_treatment(root, sid)
        if finish_treatment_set(root, sid, design):
            complete.add(sid)
            continue
        for i, eta in enumerate(sorted({x for x, _ in design})):
            cases = sorted(key for key in design if key[0] == eta and key not in present)
            if cases:
                jobs.append((sid, i, cases, best[sid], root, args.resolution, args.dt, args.solver))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(treatment_slice, job) for job in jobs]
        for future in as_completed(futures):
            sid, message = future.result()
            print(message, flush=True)
            if sid not in complete and finish_treatment_set(root, sid, designs[sid]):
                complete.add(sid)
                print(f"treatment set {sid} complete", flush=True)
    if complete != set(designs):
        raise ValueError("missing treatment sets after the sweep")


def treatment_jobs(args, best, root):
    jobs, designs = [], {}
    for sid in args.sets:
        raw = read_rows(args.reference / f"{sid}_new.csv")
        cases = [(float(row["eta"]), float(row["danti"])) for row in raw]
        if len(set(cases)) != 170 or len(raw) != 170:
            raise ValueError(f"expected 169 positive grid points and one reference for {sid}")
        if args.pilot:
            cases = [min(cases), max(cases), (1.0, 0.0),
                     max(pair for pair in cases if pair[0] == min(cases)[0])]
        designs[sid] = cases
        for i in range(0, len(cases), 17):
            jobs.append((sid, i // 17, cases[i:i + 17], best[sid],
                         root, args.resolution, args.dt))
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
    parser.add_argument("--resolution", type=int, choices=(32,), default=32)
    parser.add_argument("--dt", type=float, default=.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--calibration-root", type=Path,
                        help="Study root containing calibration/full; defaults to output")
    parser.add_argument("--setup", choices=("legacy_calibrated", "recalibrated"), default="recalibrated")
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--solver", choices=("cold", "warm", "ufc"), default="cold")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.experiment == "velocity":
        fits = args.calibration_root or args.output
        best = load_best(fits, args.sets)
        jobs = [(sid, mode, best[sid], args.output, args.resolution)
                for sid in args.sets for mode in ("baseline", "velocity")]
        pool_map(velocity_task, jobs, args.workers)
        for sid in args.sets:
            source = fits / "calibration" / "full" / f"set_{sid:03d}" / "trajectory.csv"
            if source.is_file():
                write_rows(args.output / "velocity" / f"set_{sid:03d}/calibration.csv", read_rows(source))
        output = args.output / "velocity"
    else:
        if args.reference is None:
            parser.error("--reference is required for the treatment dose grid")
        if args.legacy_root is None:
            parser.error("--legacy-root is required for the original fixed treatment coefficients")
        raw = {s: legacy_parameters(args.legacy_root, s, "param_final-m") for s in args.sets}
        if args.setup == "legacy_calibrated":
            best = {s: map_legacy_exchange(p) for s, p in raw.items()}
        else:
            best = load_best(args.output, args.sets)
            original_fixed = {
                s: {k: best[s][k] for k in FIXED_KEYS} for s in args.sets
            }
            for s, p in best.items():
                p.update({k: raw[s][k] for k in FIXED_KEYS})
        root = args.output / "treatment" / args.setup
        if args.solver == "cold":
            jobs, designs = treatment_jobs(args, best, root)
            pool_map(treatment_task, jobs, args.workers)
            for sid, design in designs.items():
                if not finish_treatment_set(root, sid, design):
                    raise ValueError(f"treatment grid is incomplete for {sid}")
        else:
            treatment_slice_batch(args, best, root)
        write_rows(root / "applied_parameters.csv", [{"set_id": s, **p} for s, p in best.items()])
        if args.setup == "legacy_calibrated":
            write_rows(root / "original_parameters_before_tau_mapping.csv",
                       [{"set_id": s, **p} for s, p in raw.items()])
        else:
            write_rows(root / "calibration_fixed_parameters_before_treatment.csv",
                       [{"set_id": s, **p} for s, p in original_fixed.items()])
        output = root
    metadata = {"resolution": args.resolution, "mesh": "original_mshr",
                "interface_law": "conservative", "sets": list(args.sets),
                "dt": 0.01 if args.experiment == "velocity" else args.dt,
                "solver": args.solver}
    (output / "batch_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
