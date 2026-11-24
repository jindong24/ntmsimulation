from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .results import SimulationResult, SnapshotData


def match_snapshot_times(available, requested, tolerance):
    available_times = np.asarray(available, dtype=float)
    requested_times = tuple(float(target) for target in requested)
    tolerance = float(tolerance)

    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("snapshot tolerance must be finite and non-negative")
    if available_times.ndim != 1:
        raise ValueError("available times must be one-dimensional")
    if not np.all(np.isfinite(available_times)) or not all(
        np.isfinite(target) for target in requested_times
    ):
        raise ValueError("available and requested times must be finite")
    if requested_times and available_times.size == 0:
        raise ValueError(f"no saved state for t={requested_times[0]}")

    indices = []
    for target in requested_times:
        index = int(np.argmin(np.abs(available_times - target)))
        if abs(float(available_times[index]) - target) > tolerance:
            raise ValueError(f"no saved state for t={target}")
        indices.append(index)
    return tuple(indices)


def _time_grid(final_time: float, dt: float) -> np.ndarray:
    final_time = float(final_time)
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("simulation dt must be finite and positive")
    if not np.isfinite(final_time) or final_time < 0.0:
        raise ValueError("simulation final time must be finite and non-negative")
    quotient = final_time / dt
    # Legacy loops retain a fixed dt and stop at the last full step before T.
    step_count = int(np.floor(quotient + 1.0e-10))
    return np.arange(step_count + 1, dtype=float) * dt


def _combine_snapshots(snapshots: list[SnapshotData]) -> SnapshotData | None:
    if not snapshots:
        return None
    first = snapshots[0]
    return SnapshotData(
        time=np.concatenate([snapshot.time for snapshot in snapshots]),
        mucus_xy=first.mucus_xy,
        mucus_triangles=first.mucus_triangles,
        tissue_xy=first.tissue_xy,
        tissue_triangles=first.tissue_triangles,
        bacteria_mucus=np.concatenate(
            [snapshot.bacteria_mucus for snapshot in snapshots], axis=0
        ),
        bacteria_tissue=np.concatenate(
            [snapshot.bacteria_tissue for snapshot in snapshots], axis=0
        ),
        macrophage_mucus=np.concatenate(
            [snapshot.macrophage_mucus for snapshot in snapshots], axis=0
        ),
        macrophage_tissue=np.concatenate(
            [snapshot.macrophage_tissue for snapshot in snapshots], axis=0
        ),
    )


def _package_version() -> str:
    try:
        return version("lung-pde")
    except PackageNotFoundError:
        return "0.1.0"


def _validate_run_config(model, config) -> None:
    provenance = model.provenance
    expected = {
        "set_id": provenance.set_id,
        "mode": provenance.mode,
        "parameter_source": provenance.parameter_source,
        "dt": provenance.dt,
        "resolution": provenance.resolution,
        "mesh_backend": provenance.mesh_backend,
        "solver_settings": provenance.solver_settings,
        "eta": provenance.eta,
        "danti": provenance.danti,
    }
    actual = {
        "set_id": config.set_id,
        "mode": config.mode,
        "parameter_source": Path(config.parameter_source).expanduser().resolve(),
        "dt": float(config.dt),
        "resolution": config.resolution,
        "mesh_backend": config.mesh_backend,
        "solver_settings": config.solver_settings,
        "eta": float(config.eta),
        "danti": float(config.danti),
    }
    mismatches = [name for name, value in actual.items() if value != expected[name]]
    if mismatches:
        raise ValueError(
            "simulation config structurally mismatches the built model: "
            + ", ".join(mismatches)
        )


def _result_metadata(model, config, time: np.ndarray) -> dict:
    provenance = model.provenance
    metadata = {
        "set_id": provenance.set_id,
        "mode": provenance.mode,
        "parameter_source": str(provenance.parameter_source),
        "parameter_source_kind": provenance.parameter_source_kind,
        "parameter_source_sha256": provenance.parameter_source_sha256,
        "dt": float(model.dt),
        "final_time": float(config.final_time),
        "actual_final_time": float(time[-1]),
        "requested_snapshot_times": [
            float(value) for value in config.snapshot_times
        ],
        "resolution": provenance.resolution,
        "mesh_backend": provenance.mesh_backend,
        "solver_settings": asdict(provenance.solver_settings),
        "command_arguments": list(config.command_arguments),
        "package_version": _package_version(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "initial_bacteria": provenance.initial_bacteria,
        "initial_macrophages": provenance.initial_macrophages,
    }
    if provenance.mode == "treatment":
        metadata.update(
            {
                "eta": float(model.eta),
                "input_danti": float(model.danti),
                "effective_danti": model.effective_danti,
            }
        )
    return metadata


def run_simulation(model, config):
    _validate_run_config(model, config)

    time = _time_grid(config.final_time, config.dt)
    tolerance = max(1.0e-12, abs(float(config.dt)) * 1.0e-6)
    snapshot_indices = match_snapshot_times(
        time,
        config.snapshot_times,
        tolerance,
    )
    capture_indices = set(snapshot_indices)
    captured = {}

    model.dt.assign(float(config.dt))
    model.reset()
    concentrations = [model.concentrations()]
    if 0 in capture_indices:
        captured[0] = model.snapshot(float(time[0]))

    for index, step_time in enumerate(time[1:], start=1):
        model.step(float(step_time))
        concentrations.append(model.concentrations())
        if index in capture_indices:
            captured[index] = model.snapshot(float(step_time))

    concentration_array = np.asarray(concentrations, dtype=float)
    snapshots = _combine_snapshots(
        [captured[index] for index in snapshot_indices]
    )
    return SimulationResult(
        time=time,
        bacteria_mucus=concentration_array[:, 0],
        bacteria_tissue=concentration_array[:, 1],
        macrophage_mucus=concentration_array[:, 2],
        macrophage_tissue=concentration_array[:, 3],
        mucus_density=concentration_array[:, 4],
        snapshots=snapshots,
        metadata=_result_metadata(model, config, time),
    )
