"""Apply the same bounded directional perturbation test to all ten fits."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

import numpy as np

from run_identifiability import IDS, calibration, write_csv

STEPS = (0.10, 0.25)
CHANNELS = ("B_mucus", "B_tissue", "M_mucus", "M_tissue")


def select_directions(j, q, bounds):
    # Use one common subspace for every mode and sign at a given fit.
    interior = np.min(np.abs(bounds - q[:, None]), axis=1) >= max(STEPS)
    assert np.all(q >= bounds[:, 0] - 1e-12) and np.all(q <= bounds[:, 1] + 1e-12)
    _, sv, vt = np.linalg.svd(j[:, interior], full_matrices=False)
    assert len(sv) >= 3
    indices = (0, (len(sv)-1)//2, len(sv)-1)
    directions = {}
    for mode, index in zip(("strong", "middle", "weak"), indices, strict=True):
        v = np.zeros_like(q)
        v[interior] = vt[index]
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        np.testing.assert_allclose(np.linalg.norm(v), 1., atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(j @ v), sv[index], atol=1e-12, rtol=1e-8)
        directions[mode] = (index+1, sv[index], v)
    return interior, directions


def symmetric_pct(a, b):
    return 200 * np.linalg.norm(a-b) / (np.linalg.norm(a) + np.linalg.norm(b))


def test_set(job):
    study, local, out, sid = job
    solver = study / "reproduce" / "solver"
    sys.path[:0] = [str(solver / "scripts"), str(solver / "src")]
    os.environ.setdefault("NTM_ORIGINAL_MESH", str(solver.parent / "mesh" / "original_mshr.h5"))
    from dolfin import LogLevel, set_log_level
    from calibrate_conservative import predict

    set_log_level(LogLevel.WARNING)
    metrics, model, times, obs, names, q, bounds, weights, tangent = calibration(study, sid)
    with np.load(local / f"set_{sid:03d}.npz") as archived:
        assert tuple(archived["parameter_names"]) == names
        np.testing.assert_array_equal(archived["optimizer_vector"], q)
        np.testing.assert_array_equal(archived["optimizer_bounds"], bounds)
        np.testing.assert_array_equal(archived["times"], times)
        np.testing.assert_array_equal(archived["observations"], obs)
        j, saved_r = archived["jacobian"], archived["residual"]
    base = predict(model, tangent.unpack(q), times, weights)
    base_r = ((base[:, (0, 2)] - obs)/obs).ravel()
    np.testing.assert_allclose(base_r, saved_r, atol=1e-10, rtol=1e-7)
    base_loss = base_r @ base_r / len(times)
    interior, directions = select_directions(j, q, bounds)
    rows, vectors = [], []
    variants, labels = [base], ["baseline"]
    for mode, (index, sigma, v) in directions.items():
        for name, free, coefficient in zip(names, interior, v, strict=True):
            vectors.append(dict(set_id=sid, mode=mode, parameter=name,
                                eligible=bool(free), coefficient=coefficient,
                                singular_index=index, singular_value=sigma))
        for step in STEPS:
            for sign in (-1, 1):
                perturbed = q + sign*step*v
                assert np.all(perturbed >= bounds[:, 0]-1e-12)
                assert np.all(perturbed <= bounds[:, 1]+1e-12)
                # Replay the nonlinear PDE; do not substitute the tangent prediction.
                data = predict(model, tangent.unpack(perturbed), times, weights)
                r = ((data[:, (0, 2)] - obs)/obs).ravel()
                loss = r @ r / len(times)
                rows.append(dict(set_id=sid, mode=mode, step=step, sign=sign,
                                 eligible_count=int(interior.sum()), singular_index=index,
                                 singular_value=sigma, baseline_MSRE=base_loss,
                                 joint_MSRE=loss, MSRE_change_pct=100*(loss/base_loss-1),
                                 mucus_change_pct=100*np.linalg.norm(r-base_r)/np.sqrt(r.size),
                                 linear_mucus_change_pct=100*step*sigma/np.sqrt(r.size),
                                 **{f"{key}_L2_change_pct": symmetric_pct(base[:, i], data[:, i])
                                    for i, key in enumerate(CHANNELS)}))
                labels.append(f"{mode}:{step}:{sign}")
                variants.append(data)
    directory = out / f"set_{sid:03d}"
    directory.mkdir()
    write_csv(directory / "outcomes.csv", rows)
    write_csv(directory / "directions.csv", vectors)
    np.savez_compressed(directory / "trajectories.npz", time=times, channel=CHANNELS,
                        variants=labels, values=variants)
    print(f"{sid}: {interior.sum()} interior parameters, {len(rows)} verified nonlinear replays",
          flush=True)
    return sid, rows, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("local", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = [(args.study.resolve(), args.local.resolve(), args.output.resolve(), sid) for sid in IDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(test_set, job) for job in jobs]
        results = sorted(future.result() for future in as_completed(futures))
    write_csv(args.output / "outcomes.csv", [r for _, rows, _ in results for r in rows])
    write_csv(args.output / "directions.csv", [r for _, _, rows in results for r in rows])
    (args.output / "settings.json").write_text(json.dumps(
        dict(sets=IDS, steps=STEPS, signs=(-1, 1), selection="same feasible interior SVD rule",
             mode_indices="largest, lower median, smallest singular value"), indent=2) + "\n")


if __name__ == "__main__":
    main()
