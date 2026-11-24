"""Cross-check all directional metrics against the saved trajectories."""

import csv
from pathlib import Path
import sys

import numpy as np

local, root = map(Path, sys.argv[1:])
read = lambda path: list(csv.DictReader(path.open()))
aggregate = read(root / "outcomes.csv")
sets = sorted({int(row["set_id"]) for row in aggregate})
assert len(sets) == 10 and len(aggregate) == 120
for sid in sets:
    folder = root / f"set_{sid:03d}"
    trials = read(folder / "outcomes.csv")
    directions = read(folder / "directions.csv")
    keys = {(r["mode"], r["step"], r["sign"]) for r in trials}
    assert len(keys) == len(trials) == 12
    assert trials == [row for row in aggregate if int(row["set_id"]) == sid]
    with np.load(local / f"set_{sid:03d}.npz") as ref, \
            np.load(folder / "trajectories.npz") as data:
        q, box, j = (ref[key] for key in ("optimizer_vector", "optimizer_bounds", "jacobian"))
        obs = ref["observations"]
        np.testing.assert_array_equal(data["time"], ref["times"])
        values = dict(zip(data["variants"].tolist(), data["values"], strict=True))
        base = values["baseline"]
        residual = lambda v: ((v[:, (0, 2)] - obs) / obs).ravel()
        base_r = residual(base)
        modes = ("strong", "middle", "weak")
        basis = np.array([[float(r["coefficient"]) for r in directions if r["mode"] == mode]
                          for mode in modes])
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
        for r in trials:
            step, sign = float(r["step"]), int(r["sign"])
            v = basis[modes.index(r["mode"])]
            point = q + sign*step*v
            assert np.all(point >= box[:, 0]-1e-12) and np.all(point <= box[:, 1]+1e-12)
            np.testing.assert_allclose(np.linalg.norm(j @ v), float(r["singular_value"]),
                                       atol=1e-12, rtol=1e-8)
            trajectory = values[f'{r["mode"]}:{step}:{sign}']
            trial_r = residual(trajectory)
            np.testing.assert_allclose(float(r["joint_MSRE"]), trial_r @ trial_r / len(obs))
            np.testing.assert_allclose(float(r["mucus_change_pct"]),
                                       100*np.linalg.norm(trial_r-base_r)/np.sqrt(base_r.size))
            for i, channel in enumerate(data["channel"].tolist()):
                a, b = base[:, i], trajectory[:, i]
                expected = 200 * np.linalg.norm(a-b) / (np.linalg.norm(a) + np.linalg.norm(b))
                np.testing.assert_allclose(float(r[f"{channel}_L2_change_pct"]), expected)
print("Verified 120 trajectories, their feasible perturbations and all output metrics.")
