import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from openpyxl import Workbook

from compute_flux import flux_rows, sample_parameters


class FluxDataTest(unittest.TestCase):
    def test_legacy_order_and_signed_reverse_columns(self):
        raw = np.tile([-3., 4., 1., 2.], (140, 1))
        values = flux_rows(raw)
        self.assertEqual(values.shape, (140, 7))
        np.testing.assert_array_equal(values[0], [0.1, -3., 4., 3., -4., 1., 2.])
        self.assertEqual(values[-1, 0], 14.)

    def test_read_workbook_parameters(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / 'sample_1.xlsx'
            book = Workbook()
            tab = book.active
            tab.title = 'variable'
            for row in (('Variable', 'Value'), ('tauB', 2.6580888768965656e-06),
                        ('tauM', 0.5696823905214384), ('eta0', .7), ('xi', .2)):
                tab.append(row)
            book.save(source)
            values = sample_parameters(source)
            self.assertAlmostEqual(values['tauB'], 2.6580888768965656e-06)
            self.assertAlmostEqual(values['tauM'], 0.5696823905214384)
            self.assertEqual(values['eta'], .7)
            self.assertEqual(values['danti'], .2)


if __name__ == '__main__':
    unittest.main()
