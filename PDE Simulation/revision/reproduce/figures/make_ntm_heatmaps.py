"""
NTM PDE heatmap figures (normalized).

Layout follows NTM_PDE_heatmap_figure_sketch.pdf:
  - Per parameter set (n): 3 rows  x  2 columns
        rows    = Total, Mucus, Tissue
        columns = Bacteria (Bact),  Macrophage (Mphi)
  - Two shared colorbars per figure: one for the whole Bacteria column,
    one for the whole Macrophage column  ("same variable -> same colorbar").

Colormap: jet.   Font: Times New Roman (text and math).
The treatment adapter supplies the normalized panels and species limits.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ------------------------------------------------------------------ fonts
# Times New Roman everywhere: regular text AND math ($...$) expressions.
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
}

# ------------------------------------------------------------------ config
CMAP = "jet"
NLEV = 101                          # contour levels

# axis labels
XLABEL = r"viscosity $\eta_0$"
YLABEL = r"$\log_{10}(d_{\mathrm{anti}})$"

ROW_NAMES = ["Total", "Mucus", "Tissue"]


def draw_category(fig, axes_col_bact, axes_col_mphi, cax_bact, cax_mphi, dat):
    """Draw one 3x2 category block into the provided axes + colorbar axes."""
    eta, danti = dat["eta"], dat["danti"]

    # unified colorbar range PER VARIABLE (across the 3 rows of that column)
    bmin, bmax = dat["limits"]["bact"]
    mmin, mmax = dat["limits"]["mphi"]
    lev_b = np.linspace(bmin, bmax, NLEV)
    lev_m = np.linspace(mmin, mmax, NLEV)
    norm_b = Normalize(vmin=bmin, vmax=bmax)
    norm_m = Normalize(vmin=mmin, vmax=mmax)

    for r in range(3):
        axb = axes_col_bact[r]
        axb.tricontourf(eta, danti, dat["bact"][r], levels=lev_b,
                        cmap=CMAP, norm=norm_b, extend="both")
        axm = axes_col_mphi[r]
        axm.tricontourf(eta, danti, dat["mphi"][r], levels=lev_m,
                        cmap=CMAP, norm=norm_m, extend="both")

        for ax in (axb, axm):
            ax.set_xlabel(XLABEL)
        axb.set_ylabel(f"{ROW_NAMES[r]}\n" + YLABEL)
        axm.set_ylabel(YLABEL)

        if r == 0:
            axb.set_title("Bacteria")
            axm.set_title("Macrophage")

    # shared colorbars, one per column (bottom of the block)
    sm_b = ScalarMappable(norm=norm_b, cmap=CMAP)
    sm_m = ScalarMappable(norm=norm_m, cmap=CMAP)
    cb_b = fig.colorbar(sm_b, cax=cax_bact, orientation="horizontal")
    cb_m = fig.colorbar(sm_m, cax=cax_mphi, orientation="horizontal")
    cb_b.set_label("Bacteria")
    cb_m.set_label("Macrophage")


def make_supplement_figure(n, dat, out_dir):
    """One 3x2 normalized figure for a single parameter set n."""
    fig = plt.figure(figsize=(11, 15))
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        height_ratios=[1, 1, 1, 0.06],
        hspace=0.32, wspace=0.28,
        left=0.09, right=0.97, top=0.93, bottom=0.07,
    )
    axes_b = [fig.add_subplot(gs[r, 0]) for r in range(3)]
    axes_m = [fig.add_subplot(gs[r, 1]) for r in range(3)]
    cax_b = fig.add_subplot(gs[3, 0])
    cax_m = fig.add_subplot(gs[3, 1])

    draw_category(fig, axes_b, axes_m, cax_b, cax_m, dat)

    fig.suptitle(f"No. {n}", fontsize=14)
    out = out_dir / f"{n}_normalized.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def make_maintext_figure(first, second, out_dir, titles=None):
    """Preserve the original two-set main-text layout."""
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(
        nrows=4, ncols=4, height_ratios=[1, 1, 1, .06],
        hspace=.32, wspace=.30, left=.06, right=.98, top=.92, bottom=.07,
    )
    for index, (set_id, data) in enumerate((first, second)):
        j = 2 * index
        axes_b = [fig.add_subplot(gs[r, j]) for r in range(3)]
        axes_m = [fig.add_subplot(gs[r, j + 1]) for r in range(3)]
        draw_category(fig, axes_b, axes_m, fig.add_subplot(gs[3, j]),
                      fig.add_subplot(gs[3, j + 1]), data)
        center = (axes_b[0].get_position().x0 + axes_m[0].get_position().x1) / 2
        title = titles[index] if titles is not None else f"No. {set_id}"
        fig.text(center, .945, title, ha="center", fontsize=24 if titles else 15,
                 fontweight="bold")
    out = out_dir / f"maintext_{first[0]}_{second[0]}_normalized.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out
