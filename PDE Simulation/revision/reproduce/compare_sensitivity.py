"""Compare the local and mucus-average velocity closures without changing inputs."""

import csv
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "solver/scripts"))

import numpy as np
from scipy.stats import spearmanr
from revision_sensitivity import VARIED, prcc
from revision_runs import read_rows, write_rows


BRANCHES = ("fixed_db1", "fixed_db2", "range_rescaled")
ADJ = "p_Bonferroni_16_per_outcome"
KEYS = (
    ("B_TOTAL", "DB1"), ("B_MUCUS", "DB1"), ("B_CLEARANCE", "DB1"),
    ("B_TISSUE", "DB1"), ("B_TISSUE", "tauB"), ("B_M_TO_T", "tauB"),
    ("B_M_TO_T", "eta"), ("B_M_TO_T", "DM"), ("B_NET_TO_T", "eta"),
    ("M_TOTAL", "eta"), ("M_TISSUE", "eta"), ("M_TISSUE", "DM"),
    ("M_TISSUE", "tauM"), ("M_M_TO_T", "eta"), ("M_M_TO_T", "DM"),
)


def endpoints(path):
    table = [{k: float(v) for k, v in r.items()} for r in read_rows(path)]
    assert [r["sample_id"] for r in table] == list(range(1, 401))
    assert all(r["time_day"] == 14 and np.isfinite(list(r.values())).all() for r in table)
    return table


def indexed(rows):
    return {(r["outcome"], r["parameter"]): r for r in rows}


def compare(branch, new_dir, previous, original):
    old_dir = previous / branch
    assert (old_dir / "design.csv").read_bytes() == (new_dir / "design.csv").read_bytes()
    solver_inputs = read_rows(new_dir / "design.csv")
    assert len(original) == 400 and len(VARIED) == 16
    for a, b in zip(original, solver_inputs, strict=True):
        assert all(a[k] == b[k] for k in a if k != "tauB"), branch
    old, new = (endpoints(directory / "endpoints.csv") for directory in (old_dir, new_dir))
    for label, inputs, filename in (
        ("common", original, "prcc_day14.csv"),
        ("solver", solver_inputs, "prcc_day14_solver_inputs.csv"),
    ):
        computed = prcc(inputs, new)
        current = indexed(computed)
        old_data = indexed(prcc(inputs, old))
        archive_file = old_dir / filename
        has_archive = (old_dir / "prcc_day14_solver_inputs.csv").is_file() and archive_file.is_file()
        archived = indexed(read_rows(archive_file)) if has_archive else None
        rows = []
        for outcome, parameter in current:
            k = outcome, parameter
            x, y = old_data[k], current[k]
            if archived is not None:
                np.testing.assert_allclose(x["PRCC"], float(archived[k]["PRCC"]), atol=1e-13)
            old_sig, new_sig = x[ADJ] < .05, y[ADJ] < .05
            rows.append(dict(
                branch=branch, predictor_basis=label, outcome=outcome, parameter=parameter, n=400,
                local_PRCC=x["PRCC"], mean_PRCC=y["PRCC"], difference=y["PRCC"] - x["PRCC"],
                local_p_adjusted=x[ADJ], mean_p_adjusted=y[ADJ],
                local_significant=int(old_sig), mean_significant=int(new_sig),
                significant_sign_reversal=int(old_sig and new_sig and x["PRCC"] * y["PRCC"] < 0),
            ))
        write_rows(new_dir / filename, computed)
        write_rows(new_dir / f"prcc_velocity_comparison_{label}.csv", rows)
    pairs = []
    for k in tuple(old[0])[2:]:
        a, b = (np.asarray([r[k] for r in values]) for values in (old, new))
        denom = np.abs(a) + np.abs(b)
        symmetric = np.divide(200*np.abs(b-a), denom, where=denom > 0, out=np.zeros_like(a))
        pairs.append(dict(branch=branch, outcome=k, n=400, local_median=np.median(a),
                          mean_median=np.median(b), median_symmetric_change_pct=np.median(symmetric),
                          spearman_rho=spearmanr(a,b).statistic))
    write_rows(new_dir / "velocity_paired_outcomes.csv", pairs)


def summarize(directory, previous):
    original = read_rows(previous / "legacy_design.csv")
    for branch in BRANCHES:
        compare(branch, directory / branch, previous, original)
    for label in ("common", "solver"):
        rows = [r for branch in BRANCHES
                for r in read_rows(directory / branch / f"prcc_velocity_comparison_{label}.csv")]
        assert len(rows) == 672
        write_rows(directory / f"prcc_velocity_comparison_{label}.csv", rows)
        table = [r for r in rows if (r["outcome"], r["parameter"]) in KEYS]
        write_rows(directory / f"prcc_key_results_{label}.csv", table)
        if label != "common":
            continue
        for branch in BRANCHES:
            data = [r for r in rows if r["branch"] == branch]
            shift = sum(r["local_significant"] != r["mean_significant"] for r in data)
            flips = sum(int(r["significant_sign_reversal"]) for r in data)
            biggest = max(data, key=lambda r: abs(float(r["difference"])))
            print(f"{branch}: {shift} significance changes, {flips} significant sign reversals; "
                  f"largest absolute PRCC change {biggest['difference']} "
                  f"({biggest['outcome']}, {biggest['parameter']})")
        for outcome, factor in KEYS:
            values = [r for r in table if (r["outcome"], r["parameter"]) == (outcome, factor)]
            print(outcome, factor, *(f"{float(r['local_PRCC']):+.3f}->{float(r['mean_PRCC']):+.3f}"
                                      for r in values))


if __name__ == "__main__":
    summarize(*map(Path, sys.argv[1:]))
