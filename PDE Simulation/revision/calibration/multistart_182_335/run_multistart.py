"""Refit 182 and 335 from three starts without changing the calibration model."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys

import numpy as np


def nearby(baseline, bounds):
    out = {}
    for i, (k, v) in enumerate(baseline.items()):
        _, lo, hi = bounds[k]
        if k == "alpha1":
            out[k] = float(np.clip(v + .05, lo, hi))
            continue
        factor = 1.2 if i % 2 == 0 else 1 / 1.2
        if not lo < v * factor < hi:
            factor = 1 / factor
        out[k] = float(np.clip(v * factor, lo, hi))
    return out


def build_starts(root, study, legacy_root, ids, bounds, legacy_parameters):
    for sid in ids:
        report = json.loads((study / "calibration/full" / f"set_{sid:03d}" / "metrics.json").read_text())
        base = report["fitted_parameters"]
        assert report["set"] == sid and report["train_count"] == 83
        assert all(tuple(report["seed_and_bounds"][k][1:]) == tuple(bounds[k][1:]) for k in bounds)
        old = legacy_parameters(legacy_root, sid, "param_final2")
        original_mucus = dict(old, tauB=old["DB1"] * old["tauB"], tauM=old["DM"] * old["tauM"])
        low_tissue = dict(base, DB1=.002, DB2=.0006, lambdaB=.39, KB=.1,
                          KBi=5e-6, dMb=.00025, dB=.2, tauB=.035)
        starts = {"near_current": nearby(base, bounds), "legacy_mucus_side": original_mucus,
                  "low_tissue_diffusion": low_tissue}
        for label, parameters in starts.items():
            values = {k: float(np.clip(parameters[k], lo, hi)) for k, (_, lo, hi) in bounds.items()}
            assert all(math.isfinite(v) for v in values.values()) and values != base
            path = root / "seeds" / f"set_{sid:03d}_{label}.json"
            record = {"set": sid, "fit": "full", "start_label": label,
                      "fitted_parameters": values}
            if path.exists():
                assert json.loads(path.read_text()) == record
            else:
                path.write_text(json.dumps(record, indent=2) + "\n")
            yield sid, label, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget", type=int, default=28)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    study = args.study.resolve()
    revision = Path(__file__).resolve().parents[2]
    root = args.output or study / "runs/multistart_182_335"
    scripts = revision / "reproduce/solver/scripts"
    sys.path.insert(0, str(scripts))
    from calibrate_conservative import SEARCH
    from revision_runs import legacy_parameters
    from run_original_calibrations import run

    (root / "seeds").mkdir(exist_ok=True, parents=True)
    jobs = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for sid, label, seed in build_starts(root, study, revision / "inputs/legacy_xml",
                                            (182, 335), SEARCH, legacy_parameters):
            target = root / "runs" / f"set_{sid:03d}" / label
            future = pool.submit(run, (sid, "full", 12., seed, target, args.budget, True))
            jobs[future] = (sid, label)
        failed = []
        for future in as_completed(jobs):
            sid, label = jobs[future]
            try:
                print(f"{sid}/{label}: {future.result()}", flush=True)
            except Exception as error:
                failed.append((sid, label))
                print(f"{sid}/{label}: {error}", flush=True)
    if failed:
        raise SystemExit(f"Runs failed: {failed}")


if __name__ == "__main__":
    main()
