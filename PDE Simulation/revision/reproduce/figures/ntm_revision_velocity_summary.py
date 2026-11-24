"""Verify matched spatial pairs and quantify the velocity effect."""

import argparse
import csv
from pathlib import Path

import numpy as np


def area_and_cells(xy, cells):
    p, q, r = np.moveaxis(xy[cells], 1, 0)
    u, v = q - p, r - p
    area = .5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    assert np.all(area > 0)
    return area, cells


def l2_squared(field, area, cells):
    a, b, c = field[cells].T
    return np.dot(area / 6., a*a + b*b + c*c + a*b + a*c + b*c)


def measure(root):
    output = []
    for folder in sorted(root.glob("set_*")):
        set_id = int(folder.name[4:])
        with np.load(folder / "baseline/spatial.npz") as base, np.load(
                folder / "velocity/spatial.npz") as moved:
            np.testing.assert_array_equal(base["time"], [0., 1., 2.])
            np.testing.assert_array_equal(moved["time"], base["time"])
            for species in ("bacteria", "macrophage"):
                for region in ("mucus", "tissue"):
                    coords, indices = f"{region}_xy", f"{region}_triangles"
                    np.testing.assert_array_equal(moved[coords], base[coords])
                    np.testing.assert_array_equal(moved[indices], base[indices])
                    area, cells = area_and_cells(base[coords], base[indices])
                    key = f"{species}_{region}"
                    np.testing.assert_allclose(base[key][0], moved[key][0], atol=1e-14, rtol=0.)
                    for t, start, finish in zip(base["time"], base[key], moved[key]):
                        norm = np.sqrt(max(0., l2_squared(start, area, cells)))
                        delta = finish - start
                        average = np.dot(area, start[cells].mean(axis=1)) / area.sum()
                        delta_mean = np.dot(area, delta[cells].mean(axis=1)) / area.sum()
                        relative = np.sqrt(max(0., l2_squared(delta, area, cells))) / norm if norm else None
                        output.append({
                            "set_id": set_id, "day": t, "species": species, "region": region,
                            "baseline_region_mean": average, "with_velocity_region_mean": average + delta_mean,
                            "with_minus_without_mean": delta_mean,
                            "relative_spatial_L2_pct": None if relative is None else 100*relative,
                            "max_abs_nodal_difference": np.abs(delta).max(),
                        })
    assert len(output) == 120
    return output


def difference_plot(root, sid=756):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.cm import ScalarMappable
    from matplotlib.tri import Triangulation

    folder = root / f"set_{sid:03d}"
    figure, axes = plt.subplots(2, 2, figsize=(6.5, 5.6), layout="constrained")
    with np.load(folder / "baseline/spatial.npz") as base, np.load(folder / "velocity/spatial.npz") as moved:
        for col, species in enumerate(("bacteria", "macrophage")):
            delta = {r: moved[f"{species}_{r}"][1:] - base[f"{species}_{r}"][1:]
                     for r in ("mucus", "tissue")}
            extent = max(float(np.abs(v).max()) for v in delta.values())
            norm = TwoSlopeNorm(vmin=-extent, vcenter=0., vmax=extent)
            for row in range(2):
                ax = axes[row, col]
                for r, values in delta.items():
                    mesh = Triangulation(*base[f"{r}_xy"].T, base[f"{r}_triangles"])
                    ax.tripcolor(mesh, values[row], shading="gouraud", cmap="RdBu_r", norm=norm)
                ax.axhline(.1, lw=.45, color="#222222")
                ax.set(aspect="equal", title=f"{species.capitalize()}, day {row+1}",
                       xlabel="x (cm)", ylabel="y (cm)")
            figure.colorbar(ScalarMappable(norm=norm, cmap="RdBu_r"), ax=axes[:, col],
                            orientation="horizontal", fraction=.09, label="With minus without velocity")
    figure.suptitle(f"Parameter set {sid}: signed concentration differences")
    figure.savefig(root / "figures" / f"velocity_difference_{sid:03d}.png", dpi=240)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="velocity output directory")
    args = parser.parse_args()
    rows = measure(args.root)
    with (args.root / "matched_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    difference_plot(args.root)


if __name__ == "__main__":
    main()
