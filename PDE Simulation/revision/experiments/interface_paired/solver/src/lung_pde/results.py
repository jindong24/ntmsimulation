from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


_CONCENTRATION_NAMES = (
    "time",
    "bacteria_mucus",
    "bacteria_tissue",
    "macrophage_mucus",
    "macrophage_tissue",
    "mucus_density",
)
_SNAPSHOT_NAMES = (
    "time",
    "mucus_xy",
    "mucus_triangles",
    "tissue_xy",
    "tissue_triangles",
    "bacteria_mucus",
    "bacteria_tissue",
    "macrophage_mucus",
    "macrophage_tissue",
)
_RESULT_FILENAMES = ("concentrations.npz", "spatial.npz", "metadata.json")


@dataclass
class SnapshotData:
    time: np.ndarray
    mucus_xy: np.ndarray
    mucus_triangles: np.ndarray
    tissue_xy: np.ndarray
    tissue_triangles: np.ndarray
    bacteria_mucus: np.ndarray
    bacteria_tissue: np.ndarray
    macrophage_mucus: np.ndarray
    macrophage_tissue: np.ndarray


@dataclass
class SimulationResult:
    time: np.ndarray
    bacteria_mucus: np.ndarray
    bacteria_tissue: np.ndarray
    macrophage_mucus: np.ndarray
    macrophage_tissue: np.ndarray
    mucus_density: np.ndarray
    snapshots: SnapshotData | None
    metadata: dict[str, Any]


def _stage_savez(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        return temporary_path
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _stage_json(path: Path, value: dict[str, Any]) -> Path:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(value, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        return temporary_path
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _backup_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".backup",
    )
    os.close(descriptor)
    backup = Path(name)
    backup.unlink()
    return backup


def _commit_generation(destination: Path, staged: dict[str, Path]) -> None:
    backups: dict[str, Path] = {}
    installed: list[str] = []
    try:
        for name in _RESULT_FILENAMES:
            current = destination / name
            if current.exists():
                backup = _backup_path(current)
                current.replace(backup)
                backups[name] = backup

        for name in _RESULT_FILENAMES:
            if name in staged:
                staged[name].replace(destination / name)
                installed.append(name)
    except BaseException:
        for name in reversed(installed):
            (destination / name).unlink(missing_ok=True)
        for name in reversed(_RESULT_FILENAMES):
            backup = backups.get(name)
            if backup is not None:
                backup.replace(destination / name)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _as_arrays(value: object, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(value, name)) for name in names}


def _validate_finite_real_array(name: str, array: np.ndarray) -> None:
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain finite real numbers")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real numbers")


def _validate_concentrations(arrays: dict[str, np.ndarray]) -> None:
    if any(arrays[name].ndim != 1 for name in _CONCENTRATION_NAMES):
        raise ValueError("concentration arrays must all be one-dimensional")
    lengths = {arrays[name].shape[0] for name in _CONCENTRATION_NAMES}
    if len(lengths) != 1:
        raise ValueError("concentration arrays must have equal lengths")
    for name in _CONCENTRATION_NAMES:
        _validate_finite_real_array(name, arrays[name])


def _validate_geometry(name: str, xy: np.ndarray, triangles: np.ndarray) -> None:
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"{name}_xy must have shape (vertices, 2)")
    if not np.issubdtype(xy.dtype, np.number) or np.issubdtype(
        xy.dtype, np.complexfloating
    ):
        raise ValueError(f"{name}_xy must contain finite real coordinates")
    if not np.all(np.isfinite(xy)):
        raise ValueError(f"{name}_xy must contain finite coordinates")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"{name}_triangles must have shape (cells, 3)")
    if not np.issubdtype(triangles.dtype, np.number) or np.issubdtype(
        triangles.dtype, np.complexfloating
    ):
        raise ValueError(f"{name}_triangles must contain integral vertex indices")
    if not np.all(np.isfinite(triangles)):
        raise ValueError(f"{name}_triangles must contain finite vertex indices")
    if not np.all(triangles == np.floor(triangles)):
        raise ValueError(f"{name}_triangles must contain integral vertex indices")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= xy.shape[0]):
        raise ValueError(f"{name}_triangles contain an invalid vertex index")


def _validate_snapshots(arrays: dict[str, np.ndarray]) -> None:
    time = arrays["time"]
    if time.ndim != 1:
        raise ValueError("snapshot time must be one-dimensional")
    _validate_finite_real_array("time", time)
    _validate_geometry(
        "mucus", arrays["mucus_xy"], arrays["mucus_triangles"]
    )
    _validate_geometry(
        "tissue", arrays["tissue_xy"], arrays["tissue_triangles"]
    )

    expected_shapes = {
        "bacteria_mucus": (time.size, arrays["mucus_xy"].shape[0]),
        "macrophage_mucus": (time.size, arrays["mucus_xy"].shape[0]),
        "bacteria_tissue": (time.size, arrays["tissue_xy"].shape[0]),
        "macrophage_tissue": (time.size, arrays["tissue_xy"].shape[0]),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
        _validate_finite_real_array(name, arrays[name])


def save_simulation_result(result: SimulationResult, directory: str | Path) -> None:
    destination = Path(directory)

    concentration_arrays = _as_arrays(result, _CONCENTRATION_NAMES)
    _validate_concentrations(concentration_arrays)
    snapshot_arrays = None
    if result.snapshots is not None:
        snapshot_arrays = _as_arrays(result.snapshots, _SNAPSHOT_NAMES)
        _validate_snapshots(snapshot_arrays)
    if not isinstance(result.metadata, dict):
        raise ValueError("metadata must be a JSON object")

    destination.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    try:
        staged["concentrations.npz"] = _stage_savez(
            destination / "concentrations.npz", concentration_arrays
        )
        if snapshot_arrays is not None:
            staged["spatial.npz"] = _stage_savez(
                destination / "spatial.npz", snapshot_arrays
            )
        staged["metadata.json"] = _stage_json(
            destination / "metadata.json", result.metadata
        )
        _commit_generation(destination, staged)
    finally:
        for temporary_path in staged.values():
            temporary_path.unlink(missing_ok=True)


def _load_archive(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(names):
            raise ValueError(
                f"{path.name} must contain exactly {', '.join(names)}"
            )
        return {name: np.asarray(archive[name]) for name in names}


def load_simulation_result(directory: str | Path) -> SimulationResult:
    source = Path(directory)
    concentration_arrays = _load_archive(
        source / "concentrations.npz", _CONCENTRATION_NAMES
    )
    _validate_concentrations(concentration_arrays)

    spatial_path = source / "spatial.npz"
    snapshots = None
    if spatial_path.exists():
        snapshot_arrays = _load_archive(spatial_path, _SNAPSHOT_NAMES)
        _validate_snapshots(snapshot_arrays)
        snapshots = SnapshotData(**snapshot_arrays)

    with (source / "metadata.json").open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object")

    return SimulationResult(
        **concentration_arrays,
        snapshots=snapshots,
        metadata=metadata,
    )
