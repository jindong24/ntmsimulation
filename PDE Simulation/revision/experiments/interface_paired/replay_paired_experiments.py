#!/usr/bin/env python3
"""Isolate the interface-law effect on velocity and treatment experiments."""

import argparse
from pathlib import Path

import numpy as np

from calibrate_conservative import concentration_weights
from lung_pde.constants import REQUIRED_PARAMETER_KEYS, SUPPORTED_SET_IDS
from lung_pde.parameters import read_parameter_xml
from replay_paired_interface import mapped_parameters, original_model
from revision_runs import FIELDS, final_or_series, make_model, pool_map, read_rows, solve, write_rows
from revision_sensitivity import GEOMETRY, SPATIAL


def models(set_id, mode, parameters, resolution, dt):
    old = original_model(resolution, dt, mode=mode, initial_loads=None, set_id=set_id)
    old.assign_parameters(parameters)
    new = make_model(set_id, mode, mapped_parameters(parameters), resolution, dt)
    return old, new


def velocity_job(job):
    set_id, parameters, root, resolution = job
    dt = 0.01
    time = np.arange(201) * dt
    for mode in ("baseline", "velocity"):
        path = root / "velocity" / "fields" / f"{set_id}_{mode}.npz"
        if path.exists():
            continue
        result = {"time": np.arange(3, dtype=float)}
        for tag, model in zip(("legacy", "conservative"), models(set_id, mode, parameters, resolution, dt)):
            weights = concentration_weights(model)
            model.reset()
            concentrations = [(weights @ model.w.vector().get_local())[:4]]
            snapshots = [model.snapshot(0.0)]
            for i, t in enumerate(time[1:], 1):
                model.step(float(t))
                if i % 100 == 0:
                    snapshots.append(model.snapshot(float(t)))
                    concentrations.append((weights @ model.w.vector().get_local())[:4])
            if tag == "legacy":
                result.update({k: getattr(snapshots[0], k) for k in GEOMETRY})
            else:
                for k in GEOMETRY:
                    np.testing.assert_array_equal(result[k], getattr(snapshots[0], k))
            result[tag + "_means"] = np.array(concentrations)
            result.update({tag + "_" + k: np.concatenate([getattr(s, k) for s in snapshots]) for k in SPATIAL})
        for k in SPATIAL:
            np.testing.assert_allclose(result["legacy_" + k][0], result["conservative_" + k][0],
                                       rtol=1e-12, atol=1e-18)
        if not all(np.isfinite(a).all() for a in result.values()):
            raise RuntimeError(f"non-finite velocity result: {set_id} {mode}")
        with path.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(stream, **result)
        path.with_suffix(".tmp").replace(path)
    return f"velocity set {set_id}"


def p1_l2(values, xy, triangles):
    xyz = xy[triangles]
    ab, ac = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
    area = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]) / 2.
    local = values[triangles]
    return np.sqrt(np.sum(area / 12. * (np.sum(local**2, axis=1) + np.sum(local, axis=1)**2)))


def velocity_metrics(root, ids):
    results = []
    for set_id in ids:
        with np.load(root / "velocity" / "fields" / f"{set_id}_baseline.npz") as base, \
                np.load(root / "velocity" / "fields" / f"{set_id}_velocity.npz") as adv:
            for mode_data in (base, adv):
                for k in GEOMETRY:
                    np.testing.assert_array_equal(base[k], mode_data[k])
            for i, day in enumerate(base["time"]):
                for j, name in enumerate(SPATIAL):
                    region = name.split("_")[-1]
                    xy, tri = base[region + "_xy"], base[region + "_triangles"]
                    a0, a1 = base["legacy_" + name][i], adv["legacy_" + name][i]
                    b0, b1 = base["conservative_" + name][i], adv["conservative_" + name][i]
                    l2 = lambda z: p1_l2(z, xy, tri)
                    norm = l2(a0)
                    results.append({
                        "set_id": set_id, "day": day, "field": FIELDS[j],
                        **{f"{tag}_{mode}_mean": mode_data[tag + "_means"][i, j]
                           for tag in ("legacy", "conservative")
                           for mode, mode_data in (("no_velocity", base), ("with_velocity", adv))},
                        "interface_l2_no_velocity": l2(b0 - a0),
                        "interface_l2_with_velocity": l2(b1 - a1),
                        "legacy_velocity_l2": l2(a1 - a0),
                        "conservative_velocity_l2": l2(b1 - b0),
                        "difference_of_velocity_effect_l2": l2((b1 - b0) - (a1 - a0)),
                        "legacy_no_velocity_l2": norm,
                    })
    write_rows(root / "velocity" / "paired_metrics.csv", results)


def treatment_job(job):
    set_id, parameters, cases, root, resolution, number = job
    path = root / "treatment" / "chunks" / f"{set_id}_{number:03d}.csv"
    if path.exists():
        return f"treatment set {set_id} part {number}: already done"
    dt = 1 / 6
    a, b = models(set_id, "treatment", parameters, resolution, dt)
    time = np.arange(85) * dt
    weights = [concentration_weights(m) for m in (a, b)]
    rows = []
    for eta, dose in cases:
        rec = {"set_id": set_id, "eta": eta, "danti_effective": dose, "time_day": 14.}
        for tag, m, w in zip(("legacy", "conservative"), (a, b), weights):
            m.eta.assign(eta)
            m.danti.assign(dose / (m.abm.initial_macrophages * parameters["dB"]))
            final = solve(m, time, w)[:4]
            rec.update(dict(zip((tag + "_" + k for k in FIELDS), final)))
            rec[tag + "_B_TOTAL"] = final[0] + final[1]
            rec[tag + "_M_TOTAL"] = final[2] + final[3]
        rows.append(rec)
    write_rows(path, rows)
    return f"treatment set {set_id} part {number}: {len(cases)} pairs"


def treatment_cases(reference, set_id, axis_points):
    raw = read_rows(reference / f"{set_id}_new.csv")
    all_cases = {(float(r["eta"]), float(r["danti"])) for r in raw}
    positive = {(x, d) for x, d in all_cases if d > 0.0}
    eta, dose = (sorted({p[i] for p in positive}) for i in (0, 1))
    def picks(axis):
        return [axis[i] for i in np.rint(np.linspace(0, len(axis) - 1, axis_points)).astype(int)]
    chosen = {(x, d) for x in picks(eta) for d in picks(dose)}
    if len(positive) != 169 or len(all_cases) != 170 or not chosen <= positive:
        raise ValueError(f"unexpected treatment design: {set_id}")
    return sorted(chosen | (all_cases - positive))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("velocity", "treatment"))
    parser.add_argument("legacy_root", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--sets", nargs="+", type=int, default=SUPPORTED_SET_IDS)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--axis-points", type=int, choices=(3, 5, 7, 13), default=5)
    args = parser.parse_args()
    mode = args.experiment
    folder = "param_final2" if mode == "velocity" else "param_final-m"
    params = {}
    for s in args.sets:
        path = args.legacy_root / folder / f"parameters_{s}.xml"
        xml = read_parameter_xml(path)
        params[s] = {k: float(xml[k]) for k in REQUIRED_PARAMETER_KEYS}
    write_rows(args.root / mode / "legacy_and_mapped_inputs.csv",
               [{"set_id": s, **p, "kappaB": p["DB2"] * p["tauB"], "kappaM": p["DM"] * p["tauM"]}
                for s, p in params.items()])
    if mode == "velocity":
        (args.root / "velocity" / "fields").mkdir(parents=True, exist_ok=True)
        pool_map(velocity_job, [(s, params[s], args.root, args.resolution) for s in args.sets], args.workers)
        velocity_metrics(args.root, args.sets)
    else:
        jobs = []
        for s in args.sets:
            cases = treatment_cases(args.reference, s, args.axis_points)
            for i in range(0, len(cases), 9):
                jobs.append((s, params[s], cases[i:i + 9], args.root, args.resolution, i // 9))
        pool_map(treatment_job, jobs, args.workers)
        rows = []
        for s, _, _, _, _, number in jobs:
            rows.extend(read_rows(args.root / "treatment" / "chunks" / f"{s}_{number:03d}.csv"))
        write_rows(args.root / "treatment" / "matched_form_endpoints.csv", rows)


if __name__ == "__main__":
    main()
