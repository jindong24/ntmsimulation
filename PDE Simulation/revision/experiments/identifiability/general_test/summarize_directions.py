"""Summarize the matched nonlinear directional perturbations."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_identifiability import IDS, write_csv

MODES = ("strong", "middle", "weak")
STEPS = (0.10, 0.25)
KEYS = ("mucus_change_pct", "B_tissue_L2_change_pct", "M_tissue_L2_change_pct")
LABELS = ("Fitted mucus trajectories", "Tissue bacteria", "Tissue macrophages")
COLORS = ("#b7472d", "#287799", "#3a7859")


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def aggregate(root):
    trials = read(root / "outcomes.csv")
    rows = {(int(r["set_id"]), float(r["step"]), r["mode"], int(r["sign"])): r
            for r in trials}
    assert len(rows) == len(trials) == 120
    dataset, pooled = [], []
    for h in STEPS:
        for mode in MODES:
            metrics = []
            for sid in IDS:
                pair = [rows[sid, h, mode, sign] for sign in (-1, 1)]
                values = [max(float(r[k]) for r in pair) for k in KEYS]
                losses = [float(r["MSRE_change_pct"]) for r in pair]
                metrics.append(values)
                dataset.append(dict(step=h, mode=mode, set_id=sid,
                                    eligible_count=pair[0]["eligible_count"],
                                    **dict(zip(KEYS, values)),
                                    joint_MSRE_change_min_pct=min(losses),
                                    joint_MSRE_change_max_pct=max(losses)))
            for k, values in zip(KEYS, np.array(metrics).T, strict=True):
                pooled.append(dict(step=h, mode=mode, outcome=k,
                                   minimum_pct=values.min(), median_pct=np.median(values),
                                   maximum_pct=values.max(),
                                   maximum_set_id=IDS[int(values.argmax())]))
    write_csv(root / "by_dataset_max_over_signs.csv", dataset)
    write_csv(root / "summary_across_datasets.csv", pooled)
    return dataset


def plot(data, figure):
    by_key = {(r["step"], r["mode"], r["set_id"]): r for r in data}
    fig, axes = plt.subplots(2, 3, sharex=True, sharey="col", figsize=(11.4, 5.8),
                             layout="constrained")
    x = np.arange(len(IDS))
    for i, h in enumerate(STEPS):
        for j, (key, label) in enumerate(zip(KEYS, LABELS, strict=True)):
            ax = axes[i, j]
            for mode, color, marker, offset in zip(
                    MODES, COLORS, ("o", "s", "D"), (-.20, 0., .20), strict=True):
                values = [by_key[h, mode, sid][key] for sid in IDS]
                assert all(value > 0 for value in values), (h, mode, key)
                ax.scatter(x+offset, values, s=22, c=color, marker=marker,
                           label=mode.capitalize(), linewidths=0.25, edgecolors="white")
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=.22, which="major")
            ax.set_ylabel(f"Change (%)\n$h={h:.2f}$" if j == 0 else "Change (%)")
            if i == 0:
                ax.set_title(label, fontsize=11)
            else:
                ax.set_xticks(x, IDS, rotation=45, ha="right", fontsize=8)
                ax.set_xlabel("Dataset")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3,
               fontsize=9, frameon=False)
    fig.savefig(figure, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    plot(aggregate(args.results), args.figure)


if __name__ == "__main__":
    main()
