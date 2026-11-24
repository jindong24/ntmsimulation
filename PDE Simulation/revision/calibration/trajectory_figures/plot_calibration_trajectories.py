"""Plot the archived full and 12-day conservative calibration trajectories."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
from matplotlib import font_manager, pyplot as plt, ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SETS = (48, 83, 182, 188, 335, 586, 646, 756, 808, 843)
CUTOFF = 12.
SPECIES = (("B", "Bacteria"), ("M", "Macrophages"))
INK, BLUE, ORANGE, HOLDOUT = "#252a2e", "#245b91", "#b64825", "#eaeef1"
DASHES = (0, (4.6, 2.25))


def read_run(root, sid, mode):
    with (root / "calibration" / mode / f"set_{sid:03d}" / "trajectory.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    cols = ("day", "B_ABM", "M_ABM", "B_mucus", "M_mucus")
    values = np.array([[float(r[k]) for k in cols] for r in rows])
    used = np.array([r["used_for_fit"] == "True" for r in rows])
    assert values.shape == (83, len(cols)) and np.isfinite(values).all()
    np.testing.assert_allclose(values[:, 0], np.arange(1, 84) / 6, rtol=0, atol=1e-12)
    return values, used


def trajectories(root, sid):
    full, fit_full = read_run(root, sid, "full")
    partial, fit_partial = read_run(root, sid, "partial")
    np.testing.assert_array_equal(full[:, :3], partial[:, :3])
    np.testing.assert_array_equal(fit_full, np.ones(83, dtype=bool))
    np.testing.assert_array_equal(fit_partial, np.arange(83) < 72)
    np.testing.assert_allclose(full[fit_partial][-1, 0], CUTOFF, rtol=0, atol=1e-12)
    return {"time": full[:, 0], "obs": full[:, 1:3], "full": full[:, 3:5],
            "partial": partial[:, 3:5], "used": fit_partial}


def display_limits(block, species):
    values = np.concatenate([r[k][:, species] for r in block for k in ("obs", "full", "partial")])
    lo, hi = values.min(), values.max()
    exp = int(np.floor(np.log10(max(abs(lo), abs(hi)))))
    scale = 10. ** exp
    span = max(hi - lo, hi * .10)
    return scale, exp, (max(0, lo - .09 * span) / scale, (hi + .10 * span) / scale)


def panel(ax, r, species, title, y, bottom):
    scale, exponent, limits = y
    t, used = r["time"], r["used"]
    obs = r["obs"][:, species] / scale
    ax.axvspan(CUTOFF, 14, color=HOLDOUT, linewidth=0, zorder=0)
    ax.axvline(CUTOFF, color="#6c747b", linewidth=.85, linestyle=(0, (1.5, 2.2)), zorder=2)
    ax.plot(t, r["full"][:, species] / scale, color=BLUE, linewidth=1.85, zorder=3)
    ax.plot(t, r["partial"][:, species] / scale, color=ORANGE, linewidth=1.65, linestyle=DASHES, zorder=4)
    ax.plot(t[used], obs[used], linestyle="none", marker="o", ms=2.25, color=INK, zorder=5)
    ax.plot(t[~used], obs[~used], linestyle="none", marker="o", ms=3.6, color=INK,
            markerfacecolor="white", markeredgewidth=.8, zorder=6)
    ax.set(xlim=(0, 14), ylim=limits, xticks=(0, 2, 4, 6, 8, 10, 12, 14))
    ax.set_title(title, loc="left", pad=7, fontweight="bold", fontsize=12)
    ax.set_ylabel(rf"Concentration ($10^{{{exponent}}}$ g cm$^{{-3}}$)", labelpad=7)
    if bottom:
        ax.set_xlabel("Time (days)", labelpad=4)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.tick_params(direction="out", width=.7, length=3.0, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#677076")
        spine.set_linewidth(.65)


def legend(fig, combined):
    entries = [
        Line2D([], [], color=INK, marker="o", ms=3, linestyle="none", label="ABM: through day 12"),
        Line2D([], [], color=INK, marker="o", ms=4, markerfacecolor="white", linestyle="none",
               label="ABM: after day 12"),
        Line2D([], [], color=BLUE, linewidth=1.85, label="PDE: full fit"),
        Line2D([], [], color=ORANGE, linewidth=1.65, linestyle=DASHES, label="PDE: 12-day fit"),
        Patch(facecolor=HOLDOUT, edgecolor="none", label="Held-out window"),
    ]
    fig.legend(handles=entries, loc="upper center", bbox_to_anchor=(.5, .928 if combined else .905),
               ncol=5, frameon=True, framealpha=1, edgecolor="#b0b7bd", fancybox=False,
               fontsize=9.25, handlelength=1.8, columnspacing=1.28, borderpad=.42, handletextpad=.45)


def figure(ids, data):
    combined = len(ids) > 1
    fig, axes = plt.subplots(len(ids), 2, squeeze=False, figsize=(8.65, 6.2 if combined else 3.65))
    fig.subplots_adjust(left=.112, right=.975, top=.83 if combined else .735,
                        bottom=.10 if combined else .17, wspace=.30, hspace=.47)
    fig.suptitle("Full and 12-day calibration" if combined else f"Parameter set {ids[0]}",
                 fontsize=14.5, fontweight="bold", y=.977)
    legend(fig, combined)
    block = [data[s] for s in ids]
    limits = [display_limits(block, i) for i in range(2)]
    for row, sid in enumerate(ids):
        for col, (_, name) in enumerate(SPECIES):
            label = chr(ord("a") + 2 * row + col)
            title = f"({label}) Set {sid}: {name}" if combined else f"({label}) {name}"
            panel(axes[row, col], data[sid], col, title, limits[col], row == len(ids) - 1)
    return fig


def save(fig, base):
    fig.savefig(base.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = {sid: trajectories(args.study, sid) for sid in SETS}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    font = font_manager.findfont("Times New Roman", fallback_to_default=True)
    style = {"font.family": "Times New Roman" if "Times New Roman" in Path(font).name else "DejaVu Serif",
             "font.size": 11, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
             "ytick.color": INK, "mathtext.fontset": "stix", "pdf.fonttype": 42}
    with mpl.rc_context(style):
        with PdfPages(output / "all_sets_full_vs_partial.pdf") as pdf:
            for sid in SETS:
                fig = figure((sid,), data)
                save(fig, output / f"set_{sid:03d}_full_vs_partial")
                pdf.savefig(fig)
                plt.close(fig)
        fig = figure((182, 335), data)
        save(fig, output / "sets_182_335_full_vs_partial")
        plt.close(fig)
    (output / "plot_metadata.json").write_text(json.dumps({
        "source": str(args.study.resolve() / "calibration"), "sets": SETS, "observations": 83,
        "days_used_in_partial_fit": 12, "used_in_partial_fit": 72, "held_out_in_partial_fit": 11,
        "compartment": "mucus", "scale": "linear, units scaled by labeled powers of ten",
        "included_multistart_candidates": False,
    }, indent=2) + "\n")
    print(f"Verified observations and training masks for {len(SETS)} sets; saved figures to {output}")


if __name__ == "__main__":
    main()
