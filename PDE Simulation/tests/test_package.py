import subprocess
import sys

from lung_pde.constants import CALIBRATED_KEYS, FIXED_KEYS, SUPPORTED_SET_IDS


def test_supported_sets_and_calibration_keys_are_exact():
    assert SUPPORTED_SET_IDS == (48, 83, 182, 188, 335, 586, 646, 756, 808, 843)
    assert CALIBRATED_KEYS == (
        "DB1", "DB2", "lambdaB", "KB", "KBi", "dMb", "dB", "tauB",
        "DM", "delta", "dM", "tauM", "M0", "alpha0", "alpha1",
    )
    assert FIXED_KEYS == ("Dm", "dm", "m0", "beta0", "beta1")


def test_importing_package_does_not_import_dolfin():
    code = "import sys, lung_pde; assert 'dolfin' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
