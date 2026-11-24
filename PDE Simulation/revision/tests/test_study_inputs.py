"""Check the portable study inputs independently of FEniCS."""

import csv
import json
import math
from pathlib import Path

from lung_pde.constants import SUPPORTED_SET_IDS


ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_fit_checkpoints_and_seeds():
    seeds = ROOT / "inputs/calibration_seeds"
    previous = rows(seeds / "outputs/conservative/latest_best/best_fit_summary.csv")
    assert {int(row["set_id"]) for row in previous} == set(SUPPORTED_SET_IDS)
    for row in previous:
        sid = int(row["set_id"])
        full_seed = seeds / row["full_checkpoint"]
        partial_seed = seeds / f"outputs/conservative/partial_day12/set_{sid:03d}/metrics.json"
        assert full_seed.is_file() and partial_seed.is_file()
        for mode, n in (("full", 83), ("partial", 72)):
            fit = json.loads((ROOT / f"calibration/{mode}/set_{sid:03d}/metrics.json").read_text())
            assert fit["set"] == sid and fit["train_count"] == n
            assert fit["mesh_backend"] == "original_mshr"
            assert len(fit["fitted_parameters"]) == 15
            assert math.isclose(fit["dt"], 1 / 6)


def test_only_tau_b_changes_across_sensitivity_designs():
    folder = ROOT / "inputs/sensitivity"
    original = rows(folder / "legacy_design.csv")
    assert [int(row["sample_id"]) for row in original] == list(range(1, 401))
    for branch in ("fixed_db1", "fixed_db2", "range_rescaled"):
        design = rows(folder / branch / "design.csv")
        for old, new in zip(original, design, strict=True):
            assert old.keys() == new.keys()
            assert all(old[k] == new[k] for k in old if k != "tauB")
            if branch != "range_rescaled":
                diffusion = "DB1" if branch == "fixed_db1" else "DB2"
                assert math.isclose(float(new["tauB"]), float(old["tauB"]) * float(old[diffusion]))


def test_treatment_uses_matched_coordinates():
    for sid in SUPPORTED_SET_IDS:
        def axes(directory):
            return {(float(r["eta"]), float(r["danti"])) for r in rows(directory / f"{sid}_new.csv")}

        grid = axes(ROOT / "inputs/treatment_grid")
        assert len(grid) == 170
        assert {point for point in grid if point[1] == 0.0} == {(1.0, 0.0)}
        for branch in ("legacy_calibrated", "recalibrated"):
            actual = axes(ROOT / "inputs/treatment" / branch / "data")
            assert len(actual) == len(grid)
            for a, b in zip(sorted(actual), sorted(grid), strict=True):
                assert all(math.isclose(x, y, rel_tol=2e-15) for x, y in zip(a, b))
