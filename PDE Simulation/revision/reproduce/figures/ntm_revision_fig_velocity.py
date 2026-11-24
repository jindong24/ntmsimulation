#!/usr/bin/env python3
"""Build the manuscript's (a)-(g) velocity comparison from saved PDE results.

Primary inputs: ROOT/velocity/set_NNN/calibration.csv and each of
baseline/spatial.npz and velocity/spatial.npz. snapshots.npz is an alias for
spatial.npz. Without calibration.csv, use baseline/concentrations.npz (or CSV
with the same field names) and the original repository ABM observations.
Requires only NumPy and Matplotlib; no simulation is run.
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt, ticker, tri
from matplotlib.colors import Normalize


ABM_DIR = Path(__file__).resolve().parents[1] / "solver/data/abm"
DAYS = (0, 1, 2)
MODES = ("baseline", "velocity")
REGIONS = ("mucus", "tissue")
SPECIES = (
    ("B", "Bacteria", "bacteria", "viridis"),
    ("M", "Macrophages", "macrophage", "magma"),
)
FIELDS = tuple(f"{name}_{region}" for _, _, name, _ in SPECIES for region in REGIONS)
GEOMETRY = tuple(f"{region}_{key}" for region in REGIONS for key in ("xy", "triangles"))


def numeric(values, label):
    values = np.asarray(values)
    if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
        raise ValueError(f"{label}: expected finite real numbers")
    return values


def time_vector(values, path, *, ordered=True):
    values = numeric(values, f"{path}: time")
    if values.ndim != 1 or not values.size or np.any(values < 0):
        raise ValueError(f"{path}: expected a nonempty vector of nonnegative days")
    if ordered and np.any(values[1:] <= values[:-1]):
        raise ValueError(f"{path}: days must be strictly increasing")
    return values


def read_npz(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        missing = set(keys) - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing fields {', '.join(sorted(missing))}")
        return {key: numeric(archive[key], f"{path}: {key}") for key in keys}


def read_csv(path, keys, *, flags=False):
    expected = (*keys, "used_for_fit") if flags else keys
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(expected) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")
        columns = {key: [] for key in expected}
        for row in reader:
            try:
                for key in keys:
                    columns[key].append(float(row[key]))
                if flags:
                    columns["used_for_fit"].append(
                        {"true": True, "false": False, "1": True, "0": False}[
                            row["used_for_fit"].strip().lower()
                        ]
                    )
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise ValueError(f"{path}:{reader.line_num}: invalid CSV value") from exc
    return {
        key: np.asarray(values, dtype=bool) if key == "used_for_fit" else
        numeric(np.asarray(values, dtype=float), f"{path}: {key}")
        for key, values in columns.items()
    }


def first_file(directory, *names):
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"{directory}: expected {' or '.join(names)}")


def load_calibration(case, abm_dir, set_id):
    path = case / "calibration.csv"
    if path.is_file():
        keys = ("day", "B_ABM", "M_ABM", "B_mucus", "B_tissue", "M_mucus", "M_tissue")
        data = read_csv(path, keys, flags=True)
        days = time_vector(data["day"], path)
        pde = {
            f"{name}_{region}": data[f"{symbol}_{region}"]
            for symbol, _, name, _ in SPECIES for region in REGIONS
        }
        return days, pde, days, {s: data[f"{s}_ABM"] for s in "BM"}, data["used_for_fit"]

    path = first_file(case / "baseline", "concentrations.npz", "concentrations.csv")
    keys = ("time", *FIELDS)
    pde = read_npz(path, keys) if path.suffix == ".npz" else read_csv(path, keys)
    days = time_vector(pde["time"], path)
    for key in FIELDS:
        if pde[key].shape != days.shape:
            raise ValueError(f"{path}: {key} must have shape {days.shape}")

    path = abm_dir / f"data_{set_id}.npz"
    abm = read_npz(path, ("t", "B", "M"))
    obs_days = time_vector(abm["t"], path)
    for symbol in "BM":
        if abm[symbol].shape != (obs_days.size + 1,):
            raise ValueError(f"{path}: {symbol} must contain the initial value and observations")
        abm[symbol] = abm[symbol][1:]
    return days, pde, obs_days, abm, np.ones(obs_days.size, dtype=bool)


def load_spatial(directory):
    path = first_file(directory, "spatial.npz", "snapshots.npz")
    data = read_npz(path, ("time", *GEOMETRY, *FIELDS))
    days = time_vector(data["time"], path, ordered=False)
    indices = []
    for day in DAYS:
        matches = np.flatnonzero(np.isclose(days, day, rtol=0, atol=1e-6))
        if matches.size != 1:
            raise ValueError(
                f"{path}: expected exactly one snapshot at t={day} d; "
                f"found {matches.size} (saved days: {days.tolist()})"
            )
        indices.append(matches[0])

    meshes = {}
    for region in REGIONS:
        xy, cells = (data[f"{region}_{key}"] for key in ("xy", "triangles"))
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 3:
            raise ValueError(f"{path}: invalid {region}_xy shape {xy.shape}")
        if (cells.ndim != 2 or cells.shape[1] != 3 or not cells.size
                or np.any(cells != np.floor(cells))
                or cells.min() < 0 or cells.max() >= len(xy)):
            raise ValueError(f"{path}: invalid {region}_triangles")
        meshes[region] = tri.Triangulation(*xy.T, cells.astype(np.intp))
        for _, _, species, _ in SPECIES:
            key = f"{species}_{region}"
            if data[key].shape != (days.size, len(xy)):
                raise ValueError(f"{path}: {key} must have shape {(days.size, len(xy))}")
            data[key] = data[key][indices]
    return meshes, data


def species_norm(spatial, species):
    values = [data[f"{species}_{region}"] for _, data in spatial for region in REGIONS]
    low = min(0.0, *(a.min() for a in values))
    high = max(0.0, *(a.max() for a in values))
    return Normalize(low, high if high > low else 1.0)


def plot_calibration(fig, calibration):
    days, pde, obs_days, abm, fit = calibration
    grid = fig.add_gridspec(1, 2, left=.105, right=.967, bottom=.689, top=.867, wspace=.26)
    for i, (symbol, title, species, _) in enumerate(SPECIES):
        ax = fig.add_subplot(grid[0, i])
        mucus, tissue = (pde[f"{species}_{region}"] for region in REGIONS)
        all_values = np.concatenate((mucus, tissue, abm[symbol]))
        log = all_values.min() >= 0 and all_values.max() > 0
        if log:
            ax.set_yscale("log")

        def visible(a):
            return np.where(a > 0, a, np.nan) if log else a

        for mask, face, label in (
            (fit, "#1765AC", "ABM data-Mucus" if fit.all() else "ABM data-Mucus (fit)"),
            (~fit, "none", "ABM data-Mucus (not fitted)"),
        ):
            if mask.any():
                ax.scatter(obs_days[mask], visible(abm[symbol][mask]), s=5,
                           facecolors=face, edgecolors="#1765AC", linewidths=.45,
                           label=label, zorder=3)
        ax.plot(days, visible(mucus), color="#B3252A", lw=1.4, ls="--", zorder=4,
                label="PDE simulation-Mucus")
        ax.plot(days, visible(tissue), color="#317649", lw=1.35,
                label="PDE simulation-Tissue")
        ax.set(title=f"{title} ({symbol})", xlabel="Time (days)",
               xlim=(0, max(days[-1], obs_days[-1]) * 1.012))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=7))
        ax.grid(axis="y", linewidth=.4, color="#dfdfdf")
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_ylabel("Concentration")
            handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(.535, .597),
               frameon=False, ncol=len(labels), handlelength=2.0, columnspacing=1.65)


def plot_spatial(fig, spatial):
    norms = {key: species_norm(spatial, key) for _, _, key, _ in SPECIES}
    x = np.concatenate([mesh.x for meshes, _ in spatial for mesh in meshes.values()])
    y = np.concatenate([mesh.y for meshes, _ in spatial for mesh in meshes.values()])
    extent = [(a.min(), a.max()) for a in (x, y)]
    grid = fig.add_gridspec(2, 3, left=.105, right=.967, bottom=.136, top=.515,
                            wspace=.21, hspace=.52)
    for row, (meshes, data) in enumerate(spatial):
        for col, day in enumerate(DAYS):
            group = grid[row, col]
            bbox = group.get_position(fig)
            letter = "bcdefg"[3 * row + col]
            fig.text((bbox.x0 + bbox.x1) / 2, bbox.y1 + .033, f"({letter})  $t={day}$ d",
                     ha="center", va="center", fontsize=10)
            if col == 0:
                mode = ("Without\nvelocity", "With\nvelocity")[row]
                fig.text(.045, (bbox.y0 + bbox.y1) / 2, mode,
                         ha="center", va="center", fontsize=9)
            pair = group.subgridspec(1, 2, wspace=.14)
            for j, (symbol, _, species, cmap) in enumerate(SPECIES):
                ax = fig.add_subplot(pair[0, j])
                for region in REGIONS:
                    ax.tripcolor(meshes[region], data[f"{species}_{region}"][col],
                                 shading="gouraud", norm=norms[species], cmap=cmap,
                                 rasterized=True)
                ax.set(xlim=extent[0], ylim=extent[1], aspect="equal", anchor="N")
                ax.set_xticks(np.linspace(*extent[0], 3))
                ax.set_yticks(np.linspace(*extent[1], 3))
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
                ax.tick_params(labelsize=6.5, length=2.2, pad=1.5, labelleft=(j == 0))
                ax.set_title(symbol, fontsize=8.5, pad=2)

    for i, (symbol, title, species, cmap) in enumerate(SPECIES):
        norm = norms[species]
        power = int(np.floor(np.log10(max(abs(norm.vmin), abs(norm.vmax)))))
        scale = 10.0 ** power
        cax = fig.add_axes((.179 + .40 * i, .060, .310, .018))
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                           orientation="horizontal", ticks=ticker.MaxNLocator(nbins=5),
                           format=ticker.FuncFormatter(
                               lambda value, _, scale=scale: f"{value / scale:.3g}"
                           ))
        suffix = rf" ($\times 10^{{{power}}}$)" if power else ""
        bar.ax.set_title(f"{title} ({symbol}){suffix}", fontsize=9, pad=4)
        bar.ax.tick_params(labelsize=8, length=2.5, pad=2)


def make_figure(set_id, calibration, spatial):
    fig = plt.figure(figsize=(8.0, 5.7))
    fig.suptitle(f"Parameter set No. {set_id}", x=.535, y=.969, va="center", fontsize=14)
    fig.text(.535, .925, "(a)  Concentration fitting", ha="center", va="center", fontsize=11)
    plot_calibration(fig, calibration)
    plot_spatial(fig, spatial)
    return fig


def write_latex(path, sets):
    lines = [r"% Requires \usepackage{graphicx}. Override the path relative to the main .tex.",
             r"\providecommand{\velocityfigurepath}{.}", ""]
    for set_id in sets:
        stem = f"velocity_set_{set_id:03d}"
        lines.extend((
            r"\begin{figure*}[p]",
            r"  \centering",
            rf"  \includegraphics[width=\textwidth]{{\velocityfigurepath/{stem}.pdf}}",
            rf"  \caption{{Comparison with and without mucus velocity for parameter set No.~{set_id}.",
            r"  (a) Baseline PDE mucus and tissue concentrations and ABM mucus observations.",
            r"  (b--d) Spatial fields without velocity at $t=0$, $1$, and $2$ days.",
            r"  (e--g) Spatial fields with velocity at the same times.",
            r"  Each spatial panel shows bacteria (B, left) and macrophages (M, right).",
            r"  Each species uses one spatial color scale across both rows and all three times.}",
            rf"  \label{{fig:{stem}}}",
            r"\end{figure*}",
            "",
        ))
    path.write_text("\n".join(lines), encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="parent of velocity/set_NNN")
    parser.add_argument("--sets", "--set", nargs="+", type=int, action="extend",
                        help="set IDs; default: all set_NNN directories under ROOT/velocity")
    parser.add_argument("--out-dir", type=Path, help="default: ROOT/velocity/figures")
    parser.add_argument("--abm-dir", type=Path, default=ABM_DIR,
                        help="ABM archive directory (only used without calibration.csv)")
    parser.add_argument("--dpi", type=int, default=300, help="PNG and PDF field raster resolution")
    parser.add_argument("--latex", action="store_true", help="also write velocity_figures.tex")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    source = args.root / "velocity"
    sets = args.sets or sorted(
        int(path.name[4:]) for path in source.glob("set_*")
        if path.is_dir() and re.fullmatch(r"set_[0-9]{3,}", path.name)
    )
    sets = list(dict.fromkeys(sets))
    if not sets or any(i < 0 for i in sets):
        parser.error(f"expected nonnegative set IDs under {source}")
    destination = args.out_dir or source / "figures"

    style = {
        "font.family": "serif", "font.size": 9.5, "mathtext.fontset": "stix",
        "axes.linewidth": .65, "axes.titlesize": 10.5, "pdf.fonttype": 42,
    }
    try:
        with plt.rc_context(style):
            for set_id in sets:
                case = source / f"set_{set_id:03d}"
                calibration = load_calibration(case, args.abm_dir, set_id)
                spatial = [load_spatial(case / mode) for mode in MODES]
                fig = make_figure(set_id, calibration, spatial)
                try:
                    destination.mkdir(parents=True, exist_ok=True)
                    for ext in ("png", "pdf"):
                        path = destination / f"velocity_set_{set_id:03d}.{ext}"
                        fig.savefig(path, dpi=args.dpi, facecolor="white")
                        print(path)
                finally:
                    plt.close(fig)
        if args.latex:
            path = destination / "velocity_figures.tex"
            write_latex(path, sets)
            print(path)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
