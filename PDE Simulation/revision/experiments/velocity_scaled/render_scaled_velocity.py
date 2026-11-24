"""Create the thirty 5x panels and matched 0x / 1x / 5x comparisons."""

import argparse
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from run_scaled_velocity import SETS, STUDY


def compare(data, meshes, sid, day, species, region, area_and_cells, l2_squared):
    import numpy as np

    area, cells = area_and_cells(meshes[region]["xy"], meshes[region]["triangles"])
    start, original, scaled = (d[f"{species}_{region}"][day] for d in data)
    mean = [float(np.dot(area, a[cells].mean(axis=1)) / area.sum())
            for a in (start, original, scaled)]

    def relative_l2(a, b):
        norm = l2_squared(a, area, cells)
        return 100 * np.sqrt(l2_squared(b - a, area, cells) / norm) if norm else None

    return {
        "set_id": sid, "day": day, "species": species, "region": region,
        "mean_0x": mean[0], "mean_1x": mean[1], "mean_5x": mean[2],
        "relative_spatial_L2_pct_1x_vs_0x": relative_l2(start, original),
        "relative_spatial_L2_pct_5x_vs_0x": relative_l2(start, scaled),
        "relative_spatial_L2_pct_5x_vs_1x": relative_l2(original, scaled),
        "region_mean_change_pct_5x_vs_0x": 100 * (mean[2] - mean[0]) / mean[0]
        if mean[0] else None,
        "max_abs_nodal_change_5x_vs_0x": np.abs(scaled - start).max(),
    }


def csv_table(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_page(doc, sid, assets, fitz):
    page = doc.new_page(width=1060, height=880)
    page.insert_text((25, 38), f"Parameter set {sid}: effect of mucus velocity",
                     fontname="hebo", fontsize=22)
    page.insert_text((25, 62), "All three speeds share a concentration scale for each species and day.",
                     fontsize=13)
    for row, label in enumerate(("No velocity (0x)", "Original velocity (1x)",
                                 "Amplified velocity (5x)")):
        y = 95 + 253 * row
        page.insert_text((25, y), label, fontname="hebo", fontsize=15)
        for col, day in enumerate((1, 2)):
            x = 15 + 520 * col
            with fitz.open(assets[day, row]) as panel:
                page.show_pdf_page(fitz.Rect(x, y + 5, x + 510, y + 240), panel, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--study", type=Path, default=STUDY)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reproduce/figures"))

    import fitz
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib.tri import Triangulation
    from PIL import Image
    from ntm_revision_velocity_summary import area_and_cells, l2_squared
    from render_velocity import DAYS, REGIONS, SPECIES, pair_limits, spatial

    output = args.output
    figures = output / "figs"
    figures.mkdir(parents=True, exist_ok=False)
    metrics, ranges = [], []
    with fitz.open() as doc, TemporaryDirectory(prefix="velocity-plot-") as scratch:
        scratch = Path(scratch)
        for sid in SETS:
            key = f"set_{sid:03d}"
            reference = args.study / "velocity" / key
            scaled = args.results / key
            metadata = json.loads((scaled / "metadata.json").read_text())
            assert metadata["velocity_multiplier"] == 5.0 and metadata["actual_final_time"] == 2.0
            data = []
            for folder in (reference / "baseline", reference / "velocity", scaled):
                with np.load(folder / "spatial.npz") as f:
                    data.append(dict(f))
            for d in data:
                np.testing.assert_array_equal(d["time"], DAYS)
                for r in REGIONS:
                    for field in ("xy", "triangles"):
                        name = f"{r}_{field}"
                        np.testing.assert_array_equal(d[name], data[0][name])
                    for _, species, _ in SPECIES:
                        field = f"{species}_{r}"
                        assert np.isfinite(d[field]).all()
                        np.testing.assert_allclose(d[field][0], data[0][field][0],
                                                   rtol=1e-12, atol=1e-14)
            meshes = {r: {f: data[0][f"{r}_{f}"] for f in ("xy", "triangles")}
                      for r in REGIONS}
            triangulations = {
                r: Triangulation(*m["xy"].T, m["triangles"]) for r, m in meshes.items()
            }
            directory = figures / key
            directory.mkdir()
            assets = {}
            for day in DAYS:
                norms = {}
                for _, species, _ in SPECIES:
                    lo, hi = pair_limits(data, species, day)
                    norms[species] = Normalize(lo, hi)
                    ranges.append(dict(set_id=sid, day=day, species=species, vmin=lo, vmax=hi))
                    for r in REGIONS:
                        metrics.append(compare(data, meshes, sid, day, species, r,
                                               area_and_cells, l2_squared))
                old_fig = args.study / "velocity/figures" / key / f"{'efg'[day]}_with_velocity_t{day}.png"
                with Image.open(old_fig) as img:
                    size = img.size
                for mode in ((2,) if day == 0 else (0, 1, 2)):
                    name = f"set{sid:03d}_velocity_{(0, 1, 5)[mode]}x_t{day}"
                    base = scratch / name
                    spatial(f"{sid:03d}", day, triangulations, data[mode], size, norms, base)
                    assets[day, mode] = base.with_suffix(".pdf")
                    if mode == 2:
                        base.with_suffix(".png").replace(directory / f"{name}.png")
            add_page(doc, sid, assets, fitz)
            print(f"Set {sid}: 3 PNGs and comparison page", flush=True)
        doc.set_metadata({"title": "Mucus velocity amplitude comparison: 0x, 1x, 5x"})
        doc.save(output / "velocity_comparison_0x_1x_5x.pdf", deflate=True)
    csv_table(output / "velocity_metrics.csv", metrics)
    csv_table(output / "colorbar_ranges.csv", ranges)


if __name__ == "__main__":
    main()
