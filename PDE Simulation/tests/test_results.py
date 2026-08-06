import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

from lung_pde.results import (
    SimulationResult,
    SnapshotData,
    load_simulation_result,
    save_simulation_result,
)


def make_result(*, snapshots=None):
    return SimulationResult(
        time=np.array([0.0, 0.1]),
        bacteria_mucus=np.array([1.0, 2.0]),
        bacteria_tissue=np.array([0.0, 0.5]),
        macrophage_mucus=np.array([3.0, 4.0]),
        macrophage_tissue=np.array([0.0, 0.25]),
        mucus_density=np.array([0.001, 0.0009]),
        snapshots=snapshots,
        metadata={"set_id": 756, "mode": "baseline", "mesh_backend": "rectangle"},
    )


def make_snapshots():
    return SnapshotData(
        time=np.array([0.0, 0.1]),
        mucus_xy=np.array([[0.0, 0.1], [0.2, 0.1], [0.2, 0.2]]),
        mucus_triangles=np.array([[0, 1, 2]]),
        tissue_xy=np.array([[0.0, 0.0], [0.2, 0.0], [0.2, 0.1]]),
        tissue_triangles=np.array([[0, 1, 2]]),
        bacteria_mucus=np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]),
        bacteria_tissue=np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]),
        macrophage_mucus=np.array([[3.0, 2.0, 1.0], [3.5, 2.5, 1.5]]),
        macrophage_tissue=np.array([[0.0, 0.0, 0.0], [0.3, 0.2, 0.1]]),
    )


def test_result_round_trip_has_stable_names_and_metadata(tmp_path):
    result = make_result()

    save_simulation_result(result, tmp_path)
    restored = load_simulation_result(tmp_path)

    np.testing.assert_allclose(restored.bacteria_mucus, result.bacteria_mucus)
    assert restored.snapshots is None
    assert restored.metadata == result.metadata
    with np.load(tmp_path / "concentrations.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "time",
            "bacteria_mucus",
            "bacteria_tissue",
            "macrophage_mucus",
            "macrophage_tissue",
            "mucus_density",
        }


def test_spatial_round_trip_preserves_plotting_arrays(tmp_path):
    result = make_result(snapshots=make_snapshots())

    save_simulation_result(result, tmp_path)
    restored = load_simulation_result(tmp_path)

    assert restored.snapshots is not None
    np.testing.assert_array_equal(
        restored.snapshots.mucus_triangles, result.snapshots.mucus_triangles
    )
    np.testing.assert_allclose(
        restored.snapshots.macrophage_tissue,
        result.snapshots.macrophage_tissue,
    )


def test_loading_rejects_unequal_concentration_lengths(tmp_path):
    np.savez(
        tmp_path / "concentrations.npz",
        time=np.array([0.0, 0.1]),
        bacteria_mucus=np.array([1.0]),
        bacteria_tissue=np.array([0.0, 0.5]),
        macrophage_mucus=np.array([3.0, 4.0]),
        macrophage_tissue=np.array([0.0, 0.25]),
        mucus_density=np.array([0.001, 0.0009]),
    )
    (tmp_path / "metadata.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="equal lengths"):
        load_simulation_result(tmp_path)


def test_loading_rejects_snapshot_fields_with_wrong_vertex_dimension(tmp_path):
    save_simulation_result(make_result(snapshots=make_snapshots()), tmp_path)
    with np.load(tmp_path / "spatial.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["bacteria_mucus"] = np.ones((2, 2))
    np.savez(tmp_path / "spatial.npz", **arrays)

    with pytest.raises(ValueError, match="bacteria_mucus"):
        load_simulation_result(tmp_path)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("mucus_xy", np.array([[np.nan, 0.1], [0.2, 0.1], [0.2, 0.2]]), "finite"),
        ("mucus_triangles", np.array([[0.0, 1.0, np.nan]]), "finite"),
        ("mucus_triangles", np.array([[0.0, 1.0, 1.5]]), "integral"),
    ],
)
def test_loading_rejects_nonfinite_or_fractional_geometry(
    tmp_path, name, value, message
):
    save_simulation_result(make_result(snapshots=make_snapshots()), tmp_path)
    with np.load(tmp_path / "spatial.npz", allow_pickle=False) as archive:
        arrays = {array_name: archive[array_name] for array_name in archive.files}
    arrays[name] = value
    np.savez(tmp_path / "spatial.npz", **arrays)

    with pytest.raises(ValueError, match=message):
        load_simulation_result(tmp_path)


def test_serialization_failure_preserves_complete_previous_result(tmp_path):
    save_simulation_result(make_result(snapshots=make_snapshots()), tmp_path)
    previous = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    replacement = make_result()
    replacement.bacteria_mucus = np.array([9.0, 10.0])
    replacement.metadata = {"not_json_serializable": object()}

    with pytest.raises(TypeError, match="JSON serializable"):
        save_simulation_result(replacement, tmp_path)

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == previous


def test_commit_failure_restores_complete_previous_result(tmp_path, monkeypatch):
    save_simulation_result(make_result(snapshots=make_snapshots()), tmp_path)
    previous = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    replacement = make_result()
    replacement.bacteria_mucus = np.array([9.0, 10.0])
    replacement.metadata = {"set_id": 756, "mode": "velocity"}

    original_replace = Path.replace
    failure_injected = False

    def fail_metadata_install_once(source, target):
        nonlocal failure_injected
        target = Path(target)
        if (
            not failure_injected
            and target.name == "metadata.json"
            and source.name != "metadata.json"
        ):
            failure_injected = True
            raise OSError("injected metadata commit failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_metadata_install_once)

    with pytest.raises(OSError, match="injected metadata commit failure"):
        save_simulation_result(replacement, tmp_path)

    assert failure_injected
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == previous


def test_atomic_writes_leave_no_temporary_siblings(tmp_path):
    save_simulation_result(make_result(snapshots=make_snapshots()), tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "concentrations.npz",
        "metadata.json",
        "spatial.npz",
    ]


def test_results_module_does_not_import_dolfin():
    sys.modules.pop("lung_pde.results", None)
    sys.modules.pop("dolfin", None)

    importlib.import_module("lung_pde.results")

    assert "dolfin" not in sys.modules


@pytest.mark.parametrize(
    ("archive_name", "field"),
    [
        ("concentrations.npz", "time"),
        ("concentrations.npz", "bacteria_mucus"),
        ("concentrations.npz", "bacteria_tissue"),
        ("concentrations.npz", "macrophage_mucus"),
        ("concentrations.npz", "macrophage_tissue"),
        ("concentrations.npz", "mucus_density"),
        ("spatial.npz", "time"),
        ("spatial.npz", "bacteria_mucus"),
        ("spatial.npz", "bacteria_tissue"),
        ("spatial.npz", "macrophage_mucus"),
        ("spatial.npz", "macrophage_tissue"),
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, 1.0 + 2.0j, "not-a-number"],
    ids=("nan", "infinity", "complex", "nonnumeric"),
)
def test_save_and_load_reject_invalid_real_numeric_result_arrays(
    tmp_path, archive_name, field, invalid_value
):
    snapshots = make_snapshots() if archive_name == "spatial.npz" else None
    invalid_result = make_result(snapshots=snapshots)
    owner = invalid_result if snapshots is None else invalid_result.snapshots
    original = np.asarray(getattr(owner, field))
    setattr(owner, field, np.full(original.shape, invalid_value))

    invalid_destination = tmp_path / "invalid-save"
    with pytest.raises(ValueError, match=f"{field} must contain finite real numbers"):
        save_simulation_result(invalid_result, invalid_destination)
    assert not invalid_destination.exists()

    corrupt_destination = tmp_path / "corrupt-load"
    valid_result = make_result(
        snapshots=make_snapshots() if archive_name == "spatial.npz" else None
    )
    save_simulation_result(valid_result, corrupt_destination)
    archive_path = corrupt_destination / archive_name
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays[field] = np.full(arrays[field].shape, invalid_value)
    np.savez(archive_path, **arrays)

    with pytest.raises(ValueError, match=f"{field} must contain finite real numbers"):
        load_simulation_result(corrupt_destination)


def test_invalid_result_does_not_replace_a_complete_generation(tmp_path):
    save_simulation_result(make_result(snapshots=make_snapshots()), tmp_path)
    previous = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    replacement = make_result(snapshots=make_snapshots())
    replacement.snapshots.macrophage_tissue[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="macrophage_tissue must contain finite real numbers",
    ):
        save_simulation_result(replacement, tmp_path)

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == previous
