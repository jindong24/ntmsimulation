"""Check the original-mesh batch outputs before authoring figures and Excel."""

import argparse
import json
from pathlib import Path

import numpy as np

from lung_pde.constants import FIXED_KEYS, SUPPORTED_SET_IDS
from revision_runs import FIELDS, read_rows
from revision_sensitivity import GEOMETRY, SPATIAL


def verify(root, reference):
    ids = list(range(1, 401))
    summary = {}
    for mode, count in (("full", 83), ("partial", 72)):
        for s in SUPPORTED_SET_IDS:
            d = root / "calibration" / mode / f"set_{s:03d}"
            info = json.loads((d / "metrics.json").read_text())
            traj = read_rows(d / "trajectory.csv")
            assert info["set"] == s and info["fit"] == mode
            assert (info["mesh_vertices"], info["mesh_cells"]) == (1643, 3142)
            assert info["mesh_backend"] == "original_mshr" and info["resolution"] == 32
            assert info["train_count"] == count and info["evaluation_count"] == 83 - count
            assert "tauB" in info["fitted_parameters"] and "tauM" in info["fitted_parameters"]
            assert not any(k.startswith("kappa") for k in info["fitted_parameters"])
            assert [r["used_for_fit"] == "True" for r in traj] == [i < count for i in range(83)]
            np.testing.assert_allclose([float(r["day"]) for r in traj], np.arange(1, 84) / 6,
                                       atol=1e-12, rtol=0.)
    summary["calibrations"] = 20
    for sid in SUPPORTED_SET_IDS:
        d = root / "velocity" / f"set_{sid:03d}"
        with np.load(d / "baseline/spatial.npz") as a, np.load(d / "velocity/spatial.npz") as b:
            np.testing.assert_array_equal(a["time"], [0., 1., 2.])
            np.testing.assert_array_equal(a["time"], b["time"])
            for key in GEOMETRY:
                np.testing.assert_array_equal(a[key], b[key])
            for key, n in zip(SPATIAL, (838, 842, 838, 842)):
                assert a[key].shape == b[key].shape == (3, n)
                assert np.isfinite(a[key]).all() and np.isfinite(b[key]).all()
                np.testing.assert_allclose(a[key][0], b[key][0], atol=1e-14, rtol=0.)
    summary["velocity_pairs"] = len(SUPPORTED_SET_IDS)
    s = root / "sensitivity"
    original, design, endpoints = [read_rows(s / f"{name}.csv")
                                   for name in ("legacy_design", "design", "endpoints")]
    for r in (original, design, endpoints):
        assert [int(k["sample_id"]) for k in r] == ids
    for a, b in zip(original, design):
        assert all(a[k] == b[k] for k in a.keys() - {"tauB", "tauM"})
    for k in ("tauB", "tauM"):
        a, b = [[float(r[k]) for r in table] for table in (original, design)]
        np.testing.assert_array_equal(np.argsort(a), np.argsort(b))
    common = None
    for sid, end in zip(ids, endpoints):
        with np.load(s / "samples" / f"sample_{sid:03d}.npz", allow_pickle=False) as f:
            assert np.isfinite(f["concentrations"]).all() and np.isfinite(f["cumulative"]).all()
            np.testing.assert_allclose(f["time"], np.arange(141) / 10, atol=1e-12, rtol=0.)
            np.testing.assert_array_equal(f["daily_time"], np.arange(15))
            if common is None:
                common = {key: f[key] for key in GEOMETRY}
            for key in GEOMETRY:
                np.testing.assert_array_equal(f[key], common[key])
            for key, n in zip(SPATIAL, (838, 842, 838, 842)):
                assert f[key].shape == (15, n) and np.isfinite(f[key]).all()
            np.testing.assert_allclose(f["concentrations"][-1],
                                       [float(end[k]) for k in FIELDS], rtol=1e-14)
    assert len(read_rows(s / "prcc_day14.csv")) == 224
    summary["sensitivity_samples"] = len(ids)
    fixed = []
    for branch in ("legacy_calibrated", "recalibrated"):
        folder = root / "treatment" / branch
        meta = json.loads((folder / "batch_metadata.json").read_text())
        assert (meta["mesh"], meta["resolution"], meta["dt"]) == ("original_mshr", 32, .1)
        fixed.append({int(r["set_id"]): tuple(float(r[k]) for k in FIXED_KEYS)
                      for r in read_rows(folder / "applied_parameters.csv")})
        for sid in SUPPORTED_SET_IDS:
            name = f"{sid}_new.csv"
            original = read_rows(reference / name)
            records = read_rows(root / "treatment" / branch / "data" / name)
            grid = [(float(r["eta"]), float(r["danti"])) for r in records]
            assert grid == [(float(r["eta"]), float(r["danti"])) for r in original]
            assert len(set(grid)) == 170 and sum(d > 0. for _, d in grid) == 169
            assert (1., 0.) in grid
            assert all(float(r["dt_day"]) == .1 and float(r["time_day"]) == 14. for r in records)
            assert np.isfinite([[float(r[k]) for k in FIELDS] for r in records]).all()
            np.testing.assert_allclose([float(r["zeta"]) * float(r["dB"]) * float(r["M0"])
                                        for r in records], [dose for _, dose in grid], rtol=1e-13)
        summary[f"treatment_{branch}"] = 170 * len(SUPPORTED_SET_IDS)
    assert fixed[0] == fixed[1]
    summary["treatment_fixed_coefficients_match"] = True
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    summary = verify(args.root, args.reference)
    text = json.dumps(summary, indent=2) + "\n"
    (args.root / "verification.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
