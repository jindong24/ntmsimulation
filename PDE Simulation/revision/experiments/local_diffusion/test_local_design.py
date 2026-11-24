import unittest

from local_design import BRANCHES, SETS, design_for, load_inputs


class DesignTest(unittest.TestCase):
    def test_design_is_reproducible_and_only_varies_the_requested_coefficients(self):
        base = {'DB1': .01, 'DB2': .05, 'tauB': .001, 'tauM': .02, 'dB': .005}
        design = design_for(756, base)
        self.assertEqual(design, design_for(756, base))
        self.assertEqual(len(design), 400)
        for branch, (count, varied) in BRANCHES.items():
            records = [r for r in design if r['branch'] == branch]
            self.assertEqual([r['sample_id'] for r in records], list(range(1, count + 1)))
            for record in records:
                for name, x in base.items():
                    if name in varied:
                        self.assertTrue(.1 * x <= record[name] <= 10 * x)
                    else:
                        self.assertEqual(record[name], x)
                self.assertEqual((record['eta'], record['zeta']), (1., 1.))

    def test_sources_are_all_ten_calibrated_sets(self):
        sources, base = load_inputs()
        self.assertEqual(set(base), set(SETS))
        self.assertIn('selected', str(sources[182]))
        self.assertIn('selected', str(sources[335]))
        self.assertEqual(base[48]['dm'], 1e-10)
