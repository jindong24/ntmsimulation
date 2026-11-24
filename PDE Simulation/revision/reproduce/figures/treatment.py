"""Read treatment CSVs on the exact paper grid and normalize the six panels."""

import csv

import numpy as np
import pandas as pd

import make_ntm_heatmaps as template


BRANCHES = ("legacy_calibrated", "recalibrated")
COORDS = ("eta", "danti")
FIELDS = ("B_MUCUS", "B_TISSUE", "M_MUCUS", "M_TISSUE")
SPECIES = ("bact", "mphi")


def read_table(path, columns):
    frame = pd.read_csv(path, usecols=columns)
    data = frame.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: nonfinite treatment values")
    return data


def load_grid(path, reference):
    original = read_table(reference, COORDS)
    source = read_table(path, (*COORDS, *FIELDS))
    coords = source[:, :2]
    ordered_ref = np.lexsort(original.T[::-1])
    ordered_source = np.lexsort(coords.T[::-1])
    # Accept CSV round-off only, not a rescaled effective dose or changed viscosity.
    if (original.shape != coords.shape
            or len(np.unique(coords, axis=0)) != len(coords)
            or not np.allclose(coords[ordered_source], original[ordered_ref],
                               rtol=8 * np.finfo(float).eps, atol=0)):
        raise ValueError(f"{path}: eta and effective danti must match the original paper grid")

    positive = original[:, 1] > 0
    zero = original[~positive]
    if zero.shape != (1, 2) or tuple(zero[0]) != (1., 0.):
        raise ValueError(f"{reference}: expected a single (eta=1, danti=0) reference row")

    # Preserve the template's Delaunay point order even if the solver sorts by eta.
    row_order = np.empty(len(original), dtype=int)
    row_order[ordered_ref] = ordered_source
    bm, bt, mm, mt = source[row_order[positive], 2:].T
    return {
        "eta": original[positive, 0],
        "danti": np.log10(original[positive, 1]),
        "bact": np.stack((bm + bt, bm, bt)),
        "mphi": np.stack((mm + mt, mm, mt)),
    }


def normalize(grids, mode):
    """Return normalized panels and (mean, population std) for each species/row."""
    figures = [dict(grid, limits={}) for grid in grids]
    stats = [{} for _ in grids]
    groups = [range(len(grids))] if mode == "shared" else [(i,) for i in range(len(grids))]
    for group in groups:
        for species in SPECIES:
            reference = np.concatenate([grids[i][species] for i in group], axis=1)
            mean = reference.mean(axis=1)
            std = reference.std(axis=1)
            std[np.ptp(reference, axis=1) == 0] = 0
            lo, hi = np.inf, -np.inf
            for i in group:
                centered = grids[i][species] - mean[:, None]
                z = np.divide(centered, std[:, None], out=np.zeros_like(centered),
                              where=std[:, None] != 0)
                figures[i][species] = z
                stats[i][species] = np.column_stack((mean, std))
                lo, hi = min(lo, z.min()), max(hi, z.max())
            for i in group:
                figures[i]["limits"][species] = (lo, hi) if lo < hi else (-1., 1.)
    return figures, stats


def render(source, output, selected, reference, modes, maintext_titles=None):
    output.mkdir()
    rows = []
    main = {}
    with template.plt.rc_context(template.STYLE):
        for sid in selected:
            name = f"{int(sid)}_new.csv"
            grids = [load_grid(source / branch / "data" / name, reference / name)
                     for branch in BRANCHES]
            if any(len(grid["eta"]) != 169 for grid in grids):
                raise ValueError(f"set {sid}: expected 169 positive-dose paper grid points")
            for mode in modes:
                figures, stats = normalize(grids, mode)
                for branch, fig, refs in zip(BRANCHES, figures, stats, strict=True):
                    directory = output / branch / mode
                    directory.mkdir(parents=True, exist_ok=True)
                    template.make_supplement_figure(int(sid), fig, directory)
                    if sid in ("843", "586"):
                        main[branch, mode, sid] = fig
                    for species, label in zip(SPECIES, ("Bacteria", "Macrophage")):
                        for row, (mean, std) in zip(template.ROW_NAMES, refs[species]):
                            rows.append((sid, branch, mode, label, row, mean, std,
                                         *fig["limits"][species], len(fig["eta"])))
            print(f"set_{sid}: treatment heatmaps rendered ({', '.join(modes)})", flush=True)
        for branch in BRANCHES:
            for mode in modes:
                directory = output / branch / mode
                has_main = all((branch, mode, sid) in main for sid in ("843", "586"))
                if has_main:
                    template.make_maintext_figure(
                        (843, main[branch, mode, "843"]), (586, main[branch, mode, "586"]),
                        directory, titles=None if maintext_titles is None else maintext_titles.get(branch),
                    )
                write_latex(directory, branch, mode, selected, has_main,
                            None if maintext_titles is None else maintext_titles.get(branch))
    with (output / "scale_ranges.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("set", "branch", "normalization", "species", "row", "mean", "std",
                         "colorbar_min", "colorbar_max", "positive_points_per_branch"))
        writer.writerows(rows)


def write_latex(directory, branch, mode, selected, has_main, titles=None):
    descriptor = {
        "legacy_calibrated": "the original calibration parameters",
        "recalibrated": "the recalibrated parameters",
    }[branch]
    scaling = (
        "Within each parameter set, both calibration setups use the same "
        "standardization and species-specific color scales."
        if mode == "shared" else
        "Each panel is standardized separately; each species shares one color scale across the three rows."
    )
    caption = (
        r"The rows show total, mucus, and tissue concentrations, and the columns show bacteria "
        r"and macrophages. The axes denote the reference viscosity `$\eta_0$` and "
        r"`$\log_{10}(d_{\mathrm{anti}})$`. "
    ).replace("`", "")

    def figure(image, width, description, label):
        return (
            "\\begin{figure}[t]\n"
            "    \\centering\n"
            rf"    \includegraphics[width={width}\linewidth]{{\ntmTreatmentRoot/{image}}}" "\n"
            f"    \\caption{{{description} {caption}{scaling}}}\n"
            f"    \\label{{fig:ntm-tau-{branch.replace('_', '-')}-{mode}-{label}}}\n"
            "\\end{figure}\n"
        )

    pair = ("No.~843 (Scenario I) and No.~586 (Scenario II)" if titles else "No.~843 and No.~586")
    velocity = "using the mucus-domain mean-viscosity velocity"
    for basename, code in (
        ("maintext.tex", figure("maintext_843_586_normalized.png", "1.0",
         f"Normalized day-14 treatment responses for sets {pair} with {descriptor}, {velocity}.",
         "main") if has_main else ""),
        ("appendix.tex", "\n".join(figure(f"{int(sid)}_normalized.png", "0.7",
         f"Normalized day-14 treatment responses for parameter set No.~{int(sid)} with {descriptor}, {velocity}.",
         sid) for sid in selected)),
    ):
        if code:
            (directory / basename).write_text(
                "% Requires graphicx. Define \\ntmTreatmentRoot before input to change the image path.\n"
                "\\providecommand{\\ntmTreatmentRoot}{.}\n\n" + code,
                encoding="utf-8",
            )
