"""Paper-style velocity renderer adapted for saved original-mshr tau runs."""

import csv
import importlib.util
import shutil
import subprocess
from pathlib import Path

import fitz
import matplotlib as mpl
import numpy as np
from PIL import Image

mpl.use("Agg")
from matplotlib import pyplot as plt, ticker


SETS = ("048", "083", "182", "188", "335", "586", "646", "756", "808", "843")
DAYS = (0, 1, 2)
MODES = ("baseline", "velocity")
REGIONS = ("mucus", "tissue")
SPECIES = (("Bacteria", "bacteria", "B"), ("Macrophage", "macrophage", "M"))
PANELS = (("a_calibration", None),) + tuple(
    (f"{letter}_{tag}_t{day}", day)
    for letters, tag in (("bcd", "no_velocity"), ("efg", "with_velocity"))
    for letter, day in zip(letters, DAYS)
)

# Each entry is (pixel y of a labeled tick, its log10 value, pixels per decade).
# Measured on the 970 x 415 manuscript calibration PNGs, y increasing down.
LOG_ANCHORS = {
    "048": ((118, -6, 125), (120, -1, 116.5)),
    "083": ((91, -6, 113.5), (95, -3, 164)),
    "182": ((59, -4, 74.5), (99, -3, 210)),
    "188": ((161, -7, 112), (127, -3, 177)),
    "335": ((68, -4, 73), (96, -3, 166)),
    "586": ((85, -6, 119), (141, -1, 120)),
    "646": ((116, -6, 209), (175, -2, 178)),
    "756": ((122, -6, 121), (200, -3, 222.5)),
    "808": ((139, -6, 112), (123, -3, 192)),
    "843": ((93, -5, 88), (118, -3, 165)),
}


def original_log_limits(sid, index):
    y, log_value, decade = LOG_ANCHORS[sid][index]
    return tuple(10 ** (log_value + (y - edge) / decade) for edge in (358.5, 50.5))


def calibration_limits(sid, index, series):
    lo, hi = original_log_limits(sid, index)
    positive = np.concatenate(series)
    positive = positive[positive > 0]
    if not positive.size:
        return lo, hi
    low, high = positive.min(), positive.max()
    margin = (high / low) ** .05
    return (min(lo, low / margin) if low <= lo else lo,
            max(hi, high * margin) if high >= hi else hi)


def pair_limits(data, species, index):
    values = [mode[f"{species}_{region}"][index] for mode in data for region in REGIONS]
    return float(min(a.min() for a in values)), float(max(a.max() for a in values))


def reference_images(ref, sid):
    spatial = ref / "figs/spatial"
    cal = spatial / f"set{sid}_calibration_bm.png"
    if sid == "756":
        directory = ref / "figs/spatial_represent"
        return [cal] + [directory / f"fig_no756_{mode}_t{day}.png"
                        for mode in MODES for day in DAYS]
    return [cal] + [spatial / f"set{sid}_{mode}_t{day}.png"
                    for mode in ("no_velocity", "velocity") for day in DAYS]


def import_readers(path):
    spec = importlib.util.spec_from_file_location("revision_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pixel_axes(fig, size, box):
    w, h = size
    left, top, right, bottom = box
    return fig.add_axes((left / w, 1 - bottom / h, (right - left) / w, (bottom - top) / h))


def panel_figure(size, dpi):
    return plt.figure(figsize=tuple(np.nextafter(x / dpi, np.inf) for x in size), dpi=dpi)


def save_panel(fig, base):
    fig.savefig(base.with_suffix(".png"))
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def calibration(sid, source, size, base, rows):
    days, pde, obs_days, abm, fit = source
    dpi = 100
    with mpl.rc_context({"font.family": "Times New Roman", "font.size": 12}):
        fig = panel_figure(size, dpi)
        fig.text(.5, 1 - 1 / size[1], f"No. {int(sid)} Test", va="top", ha="center", fontsize=13)
        for i, ((title, key, symbol), x) in enumerate(zip(SPECIES, (51, 543))):
            ax = pixel_axes(fig, size, (x, 50.5, x + 410.5, 358.5))
            mucus, tissue = (pde[f"{key}_{region}"] for region in REGIONS)
            ax.semilogy(days, mucus, "r--", label="PDE-Mucus")
            ax.semilogy(obs_days, abm[symbol], "b-", label="ABM data-Mucus")
            if not fit.all():
                ax.semilogy(obs_days[~fit], abm[symbol][~fit], "bo", linestyle="none",
                            markerfacecolor="white", markersize=3, label="ABM not fitted")
            ax.semilogy(days, tissue, "g--", label="PDE-Tissue")
            series = [mucus, abm[symbol], tissue]
            lo, hi = calibration_limits(sid, i, series)
            ref_lo, ref_hi = original_log_limits(sid, i)
            rows.append((sid, key, ref_lo, ref_hi, lo, hi,
                         sum(np.count_nonzero(a <= 0) for a in series)))
            ax.set(xlim=(-.516666667, 14.516666667), ylim=(lo, hi),
                   xticks=np.arange(0, 15, 2), title=title, xlabel="$t$")
            ax.legend(loc="lower right", fontsize=11.52, framealpha=1,
                      edgecolor="#cccccc", borderpad=.625, handlelength=1.875,
                      handletextpad=.75)
        save_panel(fig, base)


def spatial(sid, day, meshes, data, size, norms, base):
    dpi = 200
    with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig = panel_figure(size, dpi)
        fig.text(1093 / size[0], 1 - 14 / size[1], f"T={day:.2f}",
                 va="top", ha="center", fontsize=12)
        for (title, key, symbol), x, c in zip(SPECIES, (118.5, 1217.5), (896.5, 1995.5)):
            ax = pixel_axes(fig, size, (x, 119.5, x + 733.5, 890))
            for region in REGIONS:
                ax.tripcolor(meshes[region], data[f"{key}_{region}"][day], shading="gouraud",
                             norm=norms[key], cmap="jet", rasterized=True)
            ax.set(xlim=(-.01, .21), ylim=(-.01, .21), title=title,
                   xticks=np.linspace(0, .2, 5), yticks=np.linspace(0, .2, 9))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
            cax = pixel_axes(fig, size, (c, 119.5, c + 39.5, 890))
            formatter = ticker.ScalarFormatter(useOffset=False, useMathText=False)
            if symbol == "B":
                # The manuscript writes bacterial concentrations with a multiplier.
                # A fixed exponent on the matched pair keeps those tick labels identical.
                magnitude = (norms[key].vmax if norms[key].vmax > 0
                             else abs(norms[key].vmin))
                power = int(np.floor(np.log10(magnitude)))
                formatter.set_powerlimits((power, power))
            fig.colorbar(mpl.cm.ScalarMappable(norm=norms[key], cmap="jet"), cax=cax,
                         ticks=ticker.MaxNLocator(nbins=8, steps=[1, 2, 4, 5, 10]),
                         format=formatter)
            fig.canvas.draw()
            lo, hi = cax.get_ylim()
            labels = [text for value, text in zip(cax.get_yticks(), cax.get_yticklabels())
                      if lo <= value <= hi]
            labels.append(cax.yaxis.get_offset_text())
            if any(t.get_window_extent().x1 > size[0] - 4 for t in labels if t.get_text()):
                raise ValueError(f"{sid}, day {day}, {key}: colorbar text exceeds reference width")
        save_panel(fig, base)


def load_spatial_pair(read, case):
    loaded = [read.load_spatial(case / mode) for mode in MODES]
    for meshes, _ in loaded:
        for region, y in zip(REGIONS, ((.1, .2), (0, .1))):
            mesh = meshes[region]
            if not np.allclose((mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()),
                               (0, .2, *y), atol=1e-10, rtol=0):
                raise ValueError(f"{case.name}: geometry does not match the paper")
    return loaded


def latex_fragment(text, ref, sid):
    marker = (r"\label{fig:no756-velocity-comparison}" if sid == "756"
              else rf"\label{{fig:appendix-set{sid}-velocity-comparison}}")
    stop = text.index(marker)
    start = text.rindex(r"\begin{figure}", 0, stop)
    stop = text.index(r"\end{figure}", stop) + len(r"\end{figure}")
    block = text[start:stop]
    block = block.replace(r"\begin{figure}[t]", r"\begin{figure}[H]")
    originals = [str(p.relative_to(ref)) for p in reference_images(ref, sid)]
    if sid != "756":
        # The manuscript repeats the baseline initial-condition asset in some
        # (e) slots. The new panel is always drawn from its own saved NPZ.
        originals[4] = originals[1] if originals[4] not in block else originals[4]
    for old, (name, _) in zip(originals, PANELS, strict=True):
        target = rf"\ntmVelocityRoot/set_{sid}/{name}.pdf"
        if "{" + old + "}" not in block:
            raise ValueError(f"{sid}: missing manuscript panel {old}")
        block = block.replace("{" + old + "}", "{" + target + "}", 1)
    return (r"% Requires graphicx, float, and subcaption." "\n"
            r"\providecommand{\ntmVelocityRoot}{.}" "\n" + block + "\n")


def write_driver(root, text, selected):
    preamble = text.split(r"\date{}", 1)[0]
    body = [preamble, r"\pagestyle{empty}", r"\begin{document}"]
    for sid in selected:
        prefix = r"\renewcommand{\thefigure}{\arabic{figure}}" if sid == "756" else \
                 r"\renewcommand{\thefigure}{S\arabic{figure}}"
        number = 2 if sid == "756" else SETS.index(sid) + 3
        body.extend([prefix, rf"\setcounter{{figure}}{{{number - 1}}}",
                     rf"\input{{set_{sid}/figure.tex}}", r"\clearpage"])
    body.append(r"\end{document}")
    (root / "velocity_figures.tex").write_text("\n".join(body) + "\n")


def clip_figure(page, top=70):
    cap, = [b for b in page.get_text("blocks")
            if b[4].startswith("Figure ") and "Evolution of the system" in b[4]]
    return fitz.Rect(70, top, 525.28, cap[3] + 2)


def render_composites(root, selected):
    pdflatex = shutil.which("pdflatex") or "/Library/TeX/texbin/pdflatex"
    build = root / ".build"
    build.mkdir(exist_ok=True)
    command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape",
               "-output-directory=.build", "velocity_figures.tex"]
    result = subprocess.run(command, cwd=root, stdout=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-4500:])
    with fitz.open(build / "velocity_figures.pdf") as full, fitz.open() as collection:
        if len(full) != len(selected):
            raise ValueError(f"Expected {len(selected)} figure pages, got {len(full)}")
        for i, sid in enumerate(selected):
            target = root / f"set_{sid}" / f"velocity_set_{sid}"
            clip = clip_figure(full[i])
            with fitz.open() as out:
                page = out.new_page(width=clip.width, height=clip.height)
                page.show_pdf_page(page.rect, full, i, clip=clip)
                out.save(target.with_suffix(".pdf"), garbage=4, deflate=True)
                page.get_pixmap(dpi=360, alpha=False).save(target.with_suffix(".png"))
                collection.insert_pdf(out)
        collection.save(root / "velocity_figures.pdf", garbage=4, deflate=True)
    shutil.rmtree(build)


def save_csv(path, header, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def render(source, root, selected, reference, reader_path):
    mpl.rcParams.update({"pdf.fonttype": 42, "savefig.facecolor": "white"})
    read = import_readers(reader_path)
    tex = (reference / "main.tex").read_text()
    spatial_rows, cal_rows = [], []
    for sid in selected:
        case = source / f"set_{sid}"
        out = root / case.name
        out.mkdir(parents=True)
        sizes = []
        for p in reference_images(reference, sid):
            with Image.open(p) as img:
                sizes.append(img.size)
        calibration(sid, read.load_calibration(case, None, sid), sizes[0],
                    out / PANELS[0][0], cal_rows)
        loaded = load_spatial_pair(read, case)
        data = [d for _, d in loaded]
        for day in DAYS:
            norms = {}
            for _, key, _ in SPECIES:
                lo, hi = pair_limits(data, key, day)
                if lo == hi:
                    raise ValueError(f"{sid}, day {day}, {key}: degenerate color range")
                norms[key] = mpl.colors.Normalize(lo, hi)
                source_limits = [v for d in data for v in pair_limits([d], key, day)]
                spatial_rows.append((sid, day, key, "g/cm^3", *source_limits, lo, hi))
            for row, (meshes, fields) in enumerate(loaded):
                i = 1 + 3 * row + day
                spatial(sid, day, meshes, fields, sizes[i], norms, out / PANELS[i][0])
        (out / "figure.tex").write_text(latex_fragment(tex, reference, sid))
        print(f"set_{sid}: 7 panels rendered", flush=True)
    save_csv(root / "spatial_ranges.csv",
             ("set", "day", "species", "unit", "no_velocity_min", "no_velocity_max",
              "with_velocity_min", "with_velocity_max", "shared_min", "shared_max"),
             spatial_rows)
    save_csv(root / "calibration_axes.csv",
             ("set", "species", "reference_ymin_est", "reference_ymax_est",
              "new_ymin", "new_ymax", "nonpositive_samples"), cal_rows)
    write_driver(root, tex, selected)
    render_composites(root, selected)
