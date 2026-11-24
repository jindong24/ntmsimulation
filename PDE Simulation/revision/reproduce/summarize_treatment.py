"""Check the treatment grids, compare velocities, and recreate the paper figures."""

import argparse
import csv
import math
from pathlib import Path
from shutil import copy2
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "figures"), str(HERE / "solver/scripts")]
from treatment import BRANCHES, render


SETS = (48, 83, 182, 188, 335, 586, 646, 756, 808, 843)
OUTCOMES = ("B_MUCUS", "B_TISSUE", "M_MUCUS", "M_TISSUE", "B_TOTAL", "M_TOTAL")


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write(path, columns, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def key(row):
    return float(row["eta"]), float(row["danti"])


def numeric(rows):
    result = {k: np.array([float(r[k]) for r in rows]) for k in OUTCOMES[:4]}
    for species in ("B", "M"):
        result[f"{species}_TOTAL"] = result[f"{species}_MUCUS"] + result[f"{species}_TISSUE"]
    assert all(np.isfinite(data).all() for data in result.values())
    return result


def endpoint_direction(rows, values):
    coordinates = np.array([key(r) for r in rows])
    eta, dose = coordinates.T
    low = dose == dose[dose > 0].min()
    m = values["M_TOTAL"]
    return float(m[low & (eta == eta.min())][0] - m[low & (eta == eta.max())][0])


def main(previous, runs, output, overlay):
    changes, steps, scenarios = [], [], []
    titles = {}
    for branch in BRANCHES:
        src, dst = runs / branch, output / branch
        (dst / "data").mkdir(parents=True, exist_ok=True)
        original = previous / branch
        copy2(src / "applied_parameters.csv", dst / "applied_parameters.csv")
        assert (dst / "applied_parameters.csv").read_bytes() == (original / "applied_parameters.csv").read_bytes()
        directions = {}
        for sid in SETS:
            name = f"{sid}_new.csv"
            old, new = (read(directory / "data" / name) for directory in (original, src))
            coordinates = [key(r) for r in new]
            assert len(coordinates) == len(set(coordinates)) == 170
            assert coordinates == [key(r) for r in old]
            for row, ref in zip(new, old, strict=True):
                for k in ("B0", "M0", "dB", "zeta"):
                    assert math.isclose(float(row[k]), float(ref[k]), rel_tol=1e-13, abs_tol=0)
                assert math.isclose(float(row["danti"]), float(row["zeta"])*float(row["dB"])*float(row["M0"]),
                                    rel_tol=1e-14)
                assert float(row["time_day"]) == 14
                if float(row["dt_day"]) != .1:
                    steps.append(dict(branch=branch, set_id=sid, eta=row["eta"], danti=row["danti"],
                                      dt_day=row["dt_day"]))
            positive = np.array([dose > 0 for _, dose in coordinates])
            a, b = (numeric(rows) for rows in (old, new))
            for k in OUTCOMES:
                previous_value, new_value = a[k][positive], b[k][positive]
                denom = np.abs(previous_value) + np.abs(new_value)
                rel = np.divide(200*np.abs(new_value-previous_value), denom, where=denom>0,
                                out=np.zeros_like(denom))
                changes.append(dict(branch=branch, set_id=sid, outcome=k, n=169,
                                    local_median=np.median(previous_value), mean_median=np.median(new_value),
                                    median_absolute_difference=np.median(np.abs(new_value - previous_value)),
                                    median_symmetric_change_pct=np.median(rel)))
            old_dir, new_dir = endpoint_direction(old, a), endpoint_direction(new, b)
            directions[sid] = new_dir
            scenarios.append(dict(branch=branch, set_id=sid, local_M_low_minus_high_eta=old_dir,
                                  mean_M_low_minus_high_eta=new_dir))
            copy2(src / "data" / name, dst / "data" / name)
        if directions[843] < 0 < directions[586]:
            titles[branch] = ("Scenario I", "Scenario II")
    for sid in SETS:
        t1, t2 = [read(output / b / "data" / f"{sid}_new.csv") for b in BRANCHES]
        assert [r["dt_day"] for r in t1] == [r["dt_day"] for r in t2]
    for name, rows in (("velocity_comparison.csv", changes), ("scenario_directions.csv", scenarios)):
        write(output / name, list(rows[0]), rows)
    write(output / "matched_timestep_overrides.csv", ("branch", "set_id", "eta", "danti", "dt_day"), steps)
    print(f"Treatment: 3,400 completed matched cases, {len(steps)} refined time steps.")
    print(f"Scenario headings retained for {', '.join(titles) or 'neither branch'}.")

    render(output, output / "figures", tuple(f"{sid:03d}" for sid in SETS),
           HERE.parent / "inputs/treatment_grid", ("paper", "shared"), maintext_titles=titles)
    overlay.mkdir(parents=True, exist_ok=True)
    images = output / "figures/recalibrated/paper"
    for sid in SETS:
        copy2(images / f"{sid}_normalized.png", overlay)
    copy2(images / "maintext_843_586_normalized.png", overlay / "maintext_treatment_heatmap_figure.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("previous", "runs", "output", "overlay"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    main(args.previous, args.runs, args.output, args.overlay)
