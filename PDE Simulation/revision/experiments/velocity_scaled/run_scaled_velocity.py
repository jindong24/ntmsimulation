"""Scale only the ordinary mucus velocity on the original mshr mesh."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys


SETS = (48, 83, 182, 188, 335, 586, 646, 756, 808, 843)
STUDY = Path(os.environ.get("NTM_RUNS", Path(__file__).resolve().parents[2] / "runs"))


def run(job):
    study, output, set_id, multiplier = job
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reproduce/solver/scripts"))
    import numpy as np
    from revision_runs import make_model
    from lung_pde.results import save_simulation_result
    from lung_pde.simulation import run_simulation

    archived = study / "velocity" / f"set_{set_id:03d}" / "velocity"
    reference = json.loads((archived / "metadata.json").read_text())
    destination = output / f"set_{set_id:03d}"
    if destination.exists():
        raise FileExistsError(destination)
    model = make_model(set_id, "velocity", reference["parameters"],
                       reference["resolution"], reference["dt"])
    model.velocity.scale = multiplier
    peak = multiplier * (0.2 - 0.1) ** 2
    for day, expected in ((0, 0.0), (1, peak), (2, 0.0)):
        model.velocity.t = day
        np.testing.assert_allclose(model.velocity(0.1, 0.2), [expected, 0.0],
                                   rtol=1e-14, atol=1e-15)
    config = replace(model.config, final_time=2., snapshot_times=(0., 1., 2.))
    result = run_simulation(model, config)
    result.metadata.update(interface_law="conservative", parameters=reference["parameters"],
                           velocity_multiplier=multiplier, peak_velocity_cm_per_day=peak)
    save_simulation_result(result, destination)

    with np.load(archived / "spatial.npz") as old, np.load(destination / "spatial.npz") as new:
        for key in old.files:
            if key == "time" or key.endswith(("_xy", "_triangles")):
                np.testing.assert_array_equal(new[key], old[key])
            else:
                atol = 1e-13 if key.startswith("bacteria") else 1e-11
                np.testing.assert_allclose(new[key][0], old[key][0], rtol=0., atol=atol)
                if multiplier == 1.0:
                    np.testing.assert_allclose(new[key], old[key], rtol=1e-7, atol=atol)
    return f"Set {set_id}: {multiplier:g}x, t = 0, 1, 2, peak {peak:.3g} cm/day"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--sets", nargs="+", type=int, choices=SETS, default=SETS)
    parser.add_argument("--multiplier", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not math.isfinite(args.multiplier) or args.multiplier <= 0:
        parser.error("multiplier must be finite and positive")
    jobs = [(args.study, args.output, sid, args.multiplier) for sid in args.sets]
    if args.workers == 1:
        for job in jobs:
            print(run(job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for message in executor.map(run, jobs):
                print(message, flush=True)


if __name__ == "__main__":
    main()
