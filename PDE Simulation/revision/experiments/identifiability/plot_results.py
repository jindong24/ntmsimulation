"""Summarize the SVD and bounded profile in supplemental figures."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_identifiability import write_csv

LABELS = (r"$D_{B1}$", r"$D_{B2}$", r"$\lambda_B$", r"$K_B$", r"$K_{Bi}$",
          r"$d_{Mb}$", r"$d_B$", r"$\tau_B$", r"$D_M$", r"$\delta$",
          r"$d_M$", r"$\tau_M$", r"$M_0$", r"$\alpha_0$", r"$\alpha_1$")
COLORS = {335: "#b94227", 182: "#296f8f", 756: "#367756"}


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def local_plot(root, figures):
    files = sorted(root.glob("set_*.npz"))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 3.9),
                                     width_ratios=(1, 1.55), layout="constrained")
    matrix, ids, pairs = [], [], []
    for file in files:
        sid = int(file.stem.split("_")[1])
        with np.load(file) as data:
            sv = data["singular_values"]
            strength = np.max(np.abs(data["independent_components"]), axis=0)
            matrix.append(strength)
            ids.append(sid)
            names = data["parameter_names"].tolist()
            d, b = data["jacobian"][:, [names.index("DB2"), names.index("tauB")]].T
            beta = d @ b / (b @ b)
            pairs.append(dict(set_id=sid,
                              sensitivity_cosine=d @ b / (np.linalg.norm(d) * np.linalg.norm(b)),
                              compensating_dlog10tauB_per_dlog10DB2=-beta,
                              remaining_fraction_l2=np.linalg.norm(d - beta*b)/np.linalg.norm(d)))
        color = COLORS.get(sid)
        left.semilogy(np.arange(1, 16), sv/sv[0], marker="o" if color else None,
                      ms=2.8, lw=1.45 if color else .85, color=color or "#b5bec3",
                      zorder=3 if color else 2, label=f"No. {sid}" if color else None)
    write_csv(root / "DB2_tauB_compensation.csv", pairs)
    left.axhline(1e-8, color="#777777", ls="--", lw=.8, label=r"$10^{-8}$ cutoff")
    left.set(xlabel="Singular-value index", ylabel=r"$\sigma_i/\sigma_1$",
             xticks=(1, 3, 6, 9, 12, 15), title="(a) Weighted sensitivity spectra")
    left.legend(loc="lower left", fontsize=8, frameon=False)
    right.set_title("(b) Independent sensitivity by parameter")
    heat = right.imshow(np.log10(np.maximum(matrix, 1e-18)), aspect="auto",
                        cmap="cividis", interpolation="none")
    right.set(xticks=np.arange(len(LABELS)), yticks=np.arange(len(ids)), ylabel="Dataset")
    right.set_xticklabels(LABELS, rotation=45, ha="right")
    right.set_yticklabels(ids)
    bar = fig.colorbar(heat, ax=right, shrink=.94)
    bar.set_label(r"$\log_{10}\|(I-P_{-i})J_i\|_\infty$")
    fig.savefig(figures / "local_identifiability.png", dpi=240)
    plt.close(fig)


def profile_plot(profile, figures):
    summary = read(profile / "profile_summary.csv")
    trajectories = read(profile / "profile_trajectories.csv")
    x = np.array([float(row["DB2"]) for row in summary])
    loss = np.array([float(row["joint_MSRE"]) for row in summary])
    converged = np.array([row["converged"] == "True" for row in summary])
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), layout="constrained")
    delta = 100 * (loss - loss.min()) / loss.min()
    axes[0].plot(x, delta, color="#374b5d", lw=1.4)
    axes[0].scatter(x[converged], delta[converged], color="#374b5d", s=20, zorder=3)
    if not converged.all():
        axes[0].scatter(x[~converged], delta[~converged], facecolor="white",
                        edgecolor="#374b5d", s=25, label="Evaluation limit", zorder=3)
        axes[0].legend(fontsize=8, frameon=False)
    axes[0].set(xlabel=r"Fixed $D_{B2}$", ylabel="Increase from best joint MSRE (%)",
                title="(a) Calibration objective", xscale="log", xticks=x,
                xticklabels=[f"{v:g}" for v in x])
    axes[0].minorticks_off()
    for d, color in zip(x, plt.get_cmap("viridis")(np.linspace(.08, .86, len(x))), strict=True):
        rows = [r for r in trajectories if float(r["DB2"]) == d]
        t = [float(row["day"]) for row in rows]
        for a, key, factor in zip(axes[1:], ("B_tissue", "M_tissue"), (1e6, 1), strict=True):
            a.plot(t, [factor*float(row[key]) for row in rows], lw=1.2, color=color,
                   label=f"{d:g}")
    axes[1].set(title="(b) Tissue bacteria", xlabel="Time (days)",
                ylabel=r"Concentration ($\times10^{-6}$)")
    axes[2].set(title="(c) Tissue macrophages", xlabel="Time (days)", ylabel="Concentration")
    axes[2].legend(title=r"Fixed $D_{B2}$", fontsize=7.5, title_fontsize=8, frameon=False)
    fig.savefig(figures / "profile_DB2_set335.png", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local", type=Path)
    parser.add_argument("figures", type=Path)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    args.figures.mkdir(exist_ok=True)
    local_plot(args.local, args.figures)
    if args.profile:
        profile_plot(args.profile, args.figures)


if __name__ == "__main__":
    main()
