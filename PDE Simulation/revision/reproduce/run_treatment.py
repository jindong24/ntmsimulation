"""Run matched treatment grids with the new mucus-mean advection closure."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
from pathlib import Path
from shutil import copy2
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "solver/scripts"))

import numpy as np
from calibrate_conservative import concentration_weights
from revision_runs import FIELDS, PARAMETERS, make_model, read_rows, write_rows


BRANCHES = ("legacy_calibrated", "recalibrated")
SETS = (756, 48, 83, 182, 188, 335, 586, 646, 808, 843)


def solve_slice(sid, eta, doses, parameters, dt, mean_viscosity=True):
    from lung_pde.ufc_batch import BatchNewton

    model = make_model(sid, "treatment", parameters, 32, dt, polynomial_velocity=True,
                       mean_viscosity_velocity=mean_viscosity, macrophage_diffusivity_in_flux=False)
    model.eta.assign(eta)
    b0, m0 = model.abm.initial_bacteria, model.abm.initial_macrophages
    factor = m0 * parameters["dB"]
    if not math.isfinite(factor) or factor <= 0.:
        raise ValueError(f"invalid effective dose conversion for set {sid}")
    multipliers = [d / factor for d in doses]
    time = np.arange(round(14. / dt) + 1) * dt
    kernel = BatchNewton(model, os.environ["NTM_UFC_LIBRARY"])
    result = kernel.slice_endpoints(time, concentration_weights(model), multipliers)
    if result.shape != (len(doses), 4) or not np.isfinite(result).all():
        raise RuntimeError("Nonfinite treatment result")
    return [dict(zip(FIELDS, row)) | {
        "eta": eta, "danti": dose, "B0": b0, "M0": m0,
        "zeta": z, "danti_input": z, "dB": parameters["dB"], "time_day": 14., "dt_day": dt,
    } for row, dose, z in zip(result, doses, multipliers, strict=True)]


def paired_slice(job):
    sid, index, eta, doses, parameters, out, mean_viscosity = job
    path = out / "paired_slices" / f"{sid:03d}_{index:02d}.csv"
    if path.is_file():
        return sid, path, "reused"
    errors = []
    for dt in (.1, .05, .025, .0125):
        rows = []
        try:
            for branch in BRANCHES:
                result = solve_slice(sid, eta, doses, parameters[branch][sid], dt, mean_viscosity)
                rows.extend(dict(row, setup=branch) for row in result)
        except RuntimeError as exc:
            print(f"set {sid}, eta={eta:g}, dt={dt:g}: {str(exc)[:360]}; retrying both branches",
                  flush=True)
            errors.append(str(exc))
            continue
        write_rows(path, rows)
        return sid, path, f"eta={eta:g}, {len(doses)} points per branch, dt={dt:g}"
    raise RuntimeError(f"Set {sid}, eta={eta:g} failed at all time steps: {errors[-1]}")


def prepare(source, output):
    parameters = {}
    designs = {}
    for branch in BRANCHES:
        before, after = source / branch, output / branch
        after.mkdir(parents=True, exist_ok=True)
        archive = before / "applied_parameters.csv"
        target = after / "applied_parameters.csv"
        if not target.exists():
            copy2(archive, target)
        assert archive.read_bytes() == target.read_bytes()
        parameters[branch] = {
            int(r["set_id"]): {k: float(r[k]) for k in PARAMETERS} for r in read_rows(archive)
        }
        designs[branch] = {}
        for sid in SETS:
            rows = read_rows(before / "data" / f"{sid}_new.csv")
            pairs = [(float(r["eta"]), float(r["danti"])) for r in rows]
            assert len(pairs) == len(set(pairs)) == 170
            assert (1., 0.) in pairs and sum(dose > 0 for _, dose in pairs) == 169
            designs[branch][sid] = pairs
    for sid in SETS:
        assert designs[BRANCHES[0]][sid] == designs[BRANCHES[1]][sid]
    return parameters, designs[BRANCHES[0]]


def merge(output, sid, order, paths):
    results = {b: {} for b in BRANCHES}
    for path in paths:
        for row in read_rows(path):
            setup = row.pop("setup")
            key = float(row["eta"]), float(row["danti"])
            assert key not in results[setup]
            results[setup][key] = row
    for branch in BRANCHES:
        assert set(order) == set(results[branch]), (branch, sid)
        write_rows(output / branch / "data" / f"{sid}_new.csv", [results[branch][key] for key in order])
    assert all(results[BRANCHES[0]][k]["dt_day"] == results[BRANCHES[1]][k]["dt_day"] for k in order)


def main(source, output, workers, sets=SETS, mean_viscosity=True):
    (output / "paired_slices").mkdir(parents=True, exist_ok=True)
    parameters, designs = prepare(source, output)
    jobs, counts = [], {}
    for sid in sets:
        pairs = designs[sid]
        groups = sorted({x for x, _ in pairs})
        counts[sid] = len(groups)
        for i, eta in enumerate(groups):
            doses = sorted(d for x, d in pairs if x == eta)
            jobs.append((sid, i, eta, doses, parameters, output, mean_viscosity))
    paths = {sid: [] for sid in designs}
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(paired_slice, j): (j[0], j[2]) for j in jobs}
        for future in as_completed(futures):
            try:
                sid, path, msg = future.result()
            except RuntimeError as exc:
                failed.append((futures[future], str(exc)))
                print(f"Treatment failure: {failed[-1]}", flush=True)
                continue
            paths[sid].append(path)
            print(f"treatment set {sid}: {msg}", flush=True)
            if len(paths[sid]) == counts[sid]:
                merge(output, sid, designs[sid], paths[sid])
                print(f"Treatment set {sid} complete in both branches.", flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} treatment slices are incomplete: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sets", nargs="+", type=int, choices=SETS, default=SETS)
    parser.add_argument("--velocity", choices=("mean", "local"), default="mean",
                        help="use a separate output directory for each closure")
    args = parser.parse_args()
    main(args.source, args.output, args.workers, tuple(dict.fromkeys(args.sets)),
         args.velocity == "mean")
