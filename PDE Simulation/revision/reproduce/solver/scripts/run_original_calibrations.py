#!/usr/bin/env python3
"""Fit the conservative tau model on the original mshr mesh."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from pathlib import Path
import subprocess
import sys

from lung_pde.constants import SUPPORTED_SET_IDS


def run(job):
    sid, mode, cutoff, seed, target, budget, tangent = job
    if (target / "metrics.json").is_file():
        return f"{sid:03d}/{mode}: already available"
    log = target.with_suffix(".log")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"remove or rename incomplete run before retry: {target}")
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("calibrate_conservative.py")),
        "--set", str(sid), "--fit", mode, "--training-days", str(cutoff),
        "--resolution", "32", "--max-nfev", str(budget),
        "--seed", str(seed), "--output", str(target),
    ]
    if tangent:
        command.append("--tangent")
    with log.open("w") as stream:
        code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(f"calibration failed for {sid:03d}/{mode}: {log}")
    return f"{sid:03d}/{mode}: saved"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeds", type=Path, help="previous PDE Simulation directory")
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--tangent", action="store_true")
    parser.add_argument("--sets", type=int, nargs="+", default=SUPPORTED_SET_IDS)
    parser.add_argument("--modes", choices=("full", "partial"), nargs="+", default=("full", "partial"))
    args = parser.parse_args()
    index = args.seeds / "outputs/conservative/latest_best/best_fit_summary.csv"
    with index.open(newline="") as f:
        previous = {int(row["set_id"]): row for row in csv.DictReader(f)}
    jobs = []
    for mode in args.modes:
        for sid in args.sets:
            seed = (args.seeds / previous[sid]["full_checkpoint"] if mode == "full" else
                    args.seeds / f"outputs/conservative/partial_day12/set_{sid:03d}/metrics.json")
            destination = args.root / "calibration" / mode / f"set_{sid:03d}"
            if not seed.is_file():
                raise FileNotFoundError(seed)
            jobs.append((sid, mode, 12., seed, destination, args.budget, args.tangent))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        errors = []
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                errors.append(str(exc))
                print(str(exc), flush=True)
    if errors:
        raise RuntimeError("\n".join(errors))


if __name__ == "__main__":
    main()
