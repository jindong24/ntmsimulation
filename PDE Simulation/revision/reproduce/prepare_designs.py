"""Keep the original tauM and change only tauB in the three new designs."""

import csv
import math
from pathlib import Path
import shutil
import sys


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(previous, target):
    legacy = previous / "fixed_db2/legacy_design.csv"
    original = read(legacy)
    assert [int(r["sample_id"]) for r in original] == list(range(1, 401))
    shutil.copy2(legacy, target / "legacy_design.csv")
    for branch, name, diffusion in (
        ("fixed_db1", "design_db1.csv", "DB1"),
        ("fixed_db2", "design.csv", "DB2"),
        ("range_rescaled", "design.csv", None),
    ):
        historical = read(previous / branch / name)
        assert len(historical) == len(original)
        result = []
        for old, row in zip(original, historical, strict=True):
            assert all(row[k] == old[k] for k in old if k not in ("tauB", "tauM"))
            if diffusion:
                assert math.isclose(float(row["tauM"]), float(old["DM"]) * float(old["tauM"]), rel_tol=1e-15)
                assert math.isclose(float(row["tauB"]), float(old[diffusion]) * float(old["tauB"]), rel_tol=1e-15)
            result.append(dict(old, tauB=row["tauB"]))
        out = target / branch
        out.mkdir()
        write(out / "design.csv", result)
        if not diffusion:
            rang = next(r for r in read(previous / branch / "ranges.csv") if r["parameter"] == "tauB")
            write(out / "tauB_mapping_range.csv", [rang])


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
