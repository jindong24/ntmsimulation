"""Check a legacy replay against the archived Excel by coordinates and day."""
import argparse
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from check_original_mesh import verify_region


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("coordinates", type=Path)
    p.add_argument("archive", type=Path)
    p.add_argument("replay", type=Path)
    args = p.parse_args()
    coord = load_workbook(args.coordinates, read_only=True, data_only=True)
    old = load_workbook(args.archive, read_only=True, data_only=True)
    try:
        with np.load(args.replay, allow_pickle=False) as new:
            for region in ("Mucus", "Tissue"):
                x = np.asarray(list(coord[region].values)[1:], dtype=float)
                order, error = verify_region(x, new[f"{region.lower()}_dof_xy"], region)
                for species in ("B", "M"):
                    name = f"{species}_{region.upper()}"
                    series = list(old[name].values)
                    if tuple(map(int, series[0])) != tuple(range(15)):
                        raise ValueError(f"{name}: unexpected archive days")
                    target = np.asarray(series[1:], dtype=float)
                    current = new[name][order, :]
                    scale = np.maximum(np.max(np.abs(target), axis=0), 1e-14)
                    diff = np.max(np.abs(current - target), axis=0) / scale
                    print(f"{name}: max coordinate error {error:.4g}; "
                          f"largest normalized nodal error {diff.max():.4g}; "
                          f"day 14 error {diff[-1]:.4g}")
                    if not np.allclose(current, target, atol=2e-9 * np.max(scale), rtol=5e-4):
                        raise ValueError(f"{name}: does not reproduce archived fields")
    finally:
        coord.close()
        old.close()


if __name__ == "__main__":
    main()
