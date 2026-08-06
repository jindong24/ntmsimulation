import numpy as np
import pytest

from lung_pde.simulation import match_snapshot_times


def test_matching_returns_same_indices_for_baseline_and_velocity_times():
    available = np.arange(0.0, 2.01, 0.01)

    assert match_snapshot_times(available, (0.0, 1.0, 2.0), 0.0051) == (
        0,
        100,
        200,
    )


def test_missing_time_is_an_error_not_nearest_unbounded_choice():
    with pytest.raises(ValueError, match="no saved state"):
        match_snapshot_times(np.array([0.0, 1.0]), (2.0,), 0.01)


def test_empty_available_times_report_no_saved_state():
    with pytest.raises(ValueError, match="no saved state"):
        match_snapshot_times(np.array([]), (0.0,), 0.01)


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        match_snapshot_times(np.array([0.0]), (0.0,), -0.01)


@pytest.mark.parametrize(
    ("available", "requested"),
    [
        (np.array([0.0, np.nan]), (0.0,)),
        (np.array([0.0, 1.0]), (np.nan,)),
    ],
)
def test_nonfinite_times_are_rejected(available, requested):
    with pytest.raises(ValueError, match="finite"):
        match_snapshot_times(available, requested, 0.01)
