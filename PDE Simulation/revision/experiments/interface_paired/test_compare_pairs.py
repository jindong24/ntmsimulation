import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("compare_pairs.py")
FACTORS = ("DB1 DB2 lambdaB KB KBi dMb dB tauB DM dM tauM M0 alpha0 alpha1 eta danti").split()
OUTCOMES = ("B_MUCUS B_TISSUE M_MUCUS M_TISSUE B_TOTAL M_TOTAL B_CLEARANCE M_CLEARANCE "
            "B_M_TO_T M_M_TO_T B_T_TO_M M_T_TO_M B_NET_TO_T M_NET_TO_T").split()


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


class ComparePairsTest(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        self.out = self.root / "out"
        rng = np.random.default_rng(2026)
        n = 400
        ranks = np.column_stack([rng.permutation(n) + 1 for _ in FACTORS])
        ids = rng.permutation(n)
        write_csv(self.root / "design.csv", ["sample_id", *FACTORS, "danti_effective"],
                  [[i + 1, *ranks[i], 2 * ranks[i, -1]] for i in ids])
        # A and B have reversed B_MUCUS ranks and differ 2x on B_TISSUE.
        # The net flux covers negative baselines and zeros; B_T_TO_M is constant.
        for label in "abc":
            rows = []
            for i in rng.permutation(n):
                x = int(ranks[i, 7])
                values = [x if label != "b" else 401 - x, 2 * x if label == "b" else x]
                values += [int(ranks[i, 0])] * 8
                values += [0, 1, (-2, 0)[i % 2] if label == "a" else (1, 2)[i % 2], 0]
                rows.append([i + 1, 14, *values])
            write_csv(self.root / f"{label}.csv", ["sample_id", "time_day", *OUTCOMES], rows)

    def run_cli(self, *extra):
        return subprocess.run([
            sys.executable, str(SCRIPT), "--design", str(self.root / "design.csv"),
            "--a", str(self.root / "a.csv"), "--b", str(self.root / "b.csv"),
            "--c", str(self.root / "c.csv"), "--out-dir", str(self.out), *extra,
        ], capture_output=True, text=True)

    def read_output(self, name):
        with (self.out / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def change_b(self, change):
        path = self.root / "b.csv"
        with path.open(newline="") as stream:
            header, *rows = csv.reader(stream)
        write_csv(path, header, change(rows))

    def test_metrics_pair_by_id_and_handle_zero_or_negative_baselines(self):
        result = self.run_cli("--no-archive")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = self.read_output("paired_metrics.csv")
        self.assertEqual(len(rows), 42)
        ba = {r["outcome"]: r for r in rows if r["comparison"] == "B-A"}
        self.assertAlmostEqual(float(ba["B_MUCUS"]["spearman_rho"]), -1)
        self.assertAlmostEqual(float(ba["B_MUCUS"]["mean_abs_rank_shift"]), 200)
        self.assertAlmostEqual(float(ba["B_TISSUE"]["median_relative_delta"]), 1)
        self.assertAlmostEqual(float(ba["B_TISSUE"]["mae"]), 200.5)
        self.assertEqual(ba["B_NET_TO_T"]["n_relative"], "200")
        self.assertAlmostEqual(float(ba["B_NET_TO_T"]["median_relative_delta"]), 1.5)
        self.assertAlmostEqual(float(ba["B_NET_TO_T"]["median_abs_symmetric_delta"]), 2)
        self.assertEqual(ba["B_T_TO_M"]["spearman_rho"], "")
        self.assertEqual(ba["B_T_TO_M"]["median_abs_symmetric_delta"], "0.0")
        self.assertIn("B-A", (self.out / "summary.txt").read_text())

    def test_prcc_tracks_the_original_tau_and_significance(self):
        result = self.run_cli("--no-archive")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = self.read_output("prcc.csv")
        self.assertEqual(len(rows), 3 * 14 * 16)
        prcc = {(r["model"], r["outcome"], r["parameter"]): r for r in rows}
        for model, corr in [("A", 1), ("B", -1), ("C", 1)]:
            factor = prcc[model, "B_MUCUS", "tauB"]
            self.assertAlmostEqual(float(factor["PRCC"]), corr)
            self.assertEqual(factor["significant"], "True")
            self.assertEqual(prcc[model, "B_MUCUS", "DB1"]["PRCC"], "")
        nonperfect = prcc["B", "B_NET_TO_T", "DB1"]
        self.assertGreater(float(nonperfect["p_raw"]), 0)
        self.assertAlmostEqual(float(nonperfect["p_Bonferroni_16_per_outcome"]),
                               min(1, 16 * float(nonperfect["p_raw"])))
        delta = next(r for r in self.read_output("prcc_deltas.csv")
                     if (r["comparison"], r["outcome"], r["parameter"]) == ("B-A", "B_MUCUS", "tauB"))
        self.assertAlmostEqual(float(delta["delta_PRCC"]), -2)
        self.assertEqual(delta["sign_flip"], "True")
        self.assertEqual(delta["significance_change"], "retained")

    def test_optional_archive_uses_only_its_six_outcomes_and_available_ids(self):
        archive = self.root / "archive.csv"
        write_csv(archive, ["sample_id", *OUTCOMES[:6]],
                  [[3, *([1] * 6)], [1, *([2] * 6)]])
        result = self.run_cli("--archive", str(archive))
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [r for r in self.read_output("paired_metrics.csv") if "archive" in r["comparison"]]
        self.assertEqual(len(rows), 18)
        self.assertEqual({r["n"] for r in rows}, {"2"})
        self.assertEqual({r["outcome"] for r in rows}, set(OUTCOMES[:6]))
        self.assertIn("assumed day 14", (self.out / "summary.txt").read_text())

    def test_rejects_unpaired_or_bad_endpoints(self):
        for change, diagnostic in [
            (lambda r: r[1:], "missing IDs"),
            (lambda r: r + [r[0]], "duplicate sample_id"),
            (lambda r: [[r[0][0], 13, *r[0][2:]], *r[1:]], "time_day"),
            (lambda r: [[*r[0][:2], "nan", *r[0][3:]], *r[1:]], "non-finite"),
        ]:
            with self.subTest(diagnostic=diagnostic):
                original = (self.root / "b.csv").read_bytes()
                try:
                    self.change_b(change)
                    result = self.run_cli("--no-archive")
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(diagnostic, result.stderr)
                    self.assertFalse(self.out.exists())
                finally:
                    (self.root / "b.csv").write_bytes(original)

    def test_explicit_missing_archive_is_not_silently_skipped(self):
        result = self.run_cli("--archive", str(self.root / "missing.csv"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing.csv", result.stderr)


if __name__ == "__main__":
    unittest.main()
