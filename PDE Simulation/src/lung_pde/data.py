from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import CALIBRATION_DT, SUPPORTED_SET_IDS


@dataclass(frozen=True)
class ABMData:
    initial_bacteria: float
    initial_macrophages: float
    times: np.ndarray
    bacteria: np.ndarray
    macrophages: np.ndarray


def validate_set_id(set_id: int) -> int:
    if set_id not in SUPPORTED_SET_IDS:
        raise ValueError(f"{set_id} is not a supported parameter set")
    return set_id


def _finite_real_array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"ABM {name} must contain finite real numbers")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"ABM {name} must contain finite real numbers")
    return np.asarray(array, dtype=float)


def load_abm(path: str | Path) -> ABMData:
    with np.load(path, allow_pickle=False) as raw:
        if set(raw.files) != {"B", "M", "t"}:
            raise ValueError(f"ABM file must contain exactly B, M, and t: {path}")
        bacteria = _finite_real_array(raw["B"], "B")
        macrophages = _finite_real_array(raw["M"], "M")
        times = _finite_real_array(raw["t"], "t")

    if bacteria.shape != (84,) or macrophages.shape != (84,) or times.shape != (83,):
        raise ValueError("ABM arrays must have B/M length 84 and t length 83")
    expected_times = np.arange(1, 84, dtype=float) * CALIBRATION_DT
    if not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-12):
        raise ValueError("ABM target times must equal 1/6 through 83/6 day")

    return ABMData(
        initial_bacteria=float(bacteria[0]),
        initial_macrophages=float(macrophages[0]),
        times=times,
        bacteria=bacteria[1:],
        macrophages=macrophages[1:],
    )
