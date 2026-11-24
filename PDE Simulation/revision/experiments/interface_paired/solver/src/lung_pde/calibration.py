from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Protocol
from uuid import uuid4

import numpy as np
from scipy.optimize import minimize

from .calibration_config import (
    PipelineConfig,
    StageConfig,
    load_pipeline,
    pack,
    project_initial_to_bounds,
    resolve_bounds,
    unpack,
)
from .constants import CALIBRATED_KEYS, CALIBRATION_DT, FIXED_KEYS
from .data import load_abm, validate_set_id
from .objectives import OBJECTIVES
from .parameters import (
    ParameterSet,
    parameter_path,
    read_parameter_xml,
    write_parameter_xml,
)
from .validation import validate_positive_integer


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_OBSERVATION_COUNT = 83
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class CalibrationSimulator(Protocol):
    def simulate(
        self, updates: Mapping[str, float]
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    input_parameters: ParameterSet
    final_parameters: ParameterSet
    loss: float | None
    evaluations: int
    output_dir: Path
    error: dict[str, str] | None = None


@dataclass(frozen=True)
class CalibrationResult:
    status: str
    final_parameters: ParameterSet
    stages: tuple[StageResult, ...]
    output_dir: Path


@dataclass(frozen=True)
class _EvaluationRecord:
    vector: np.ndarray
    parameters: dict[str, float]
    loss: float
    bacteria_mucus: np.ndarray
    macrophage_mucus: np.ndarray


class _EvaluationLimitReached(Exception):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _validate_run_id(run_id: str) -> str:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must contain only letters, digits, dots, underscores, and hyphens"
        )
    return run_id


def _stage_directory_name(index: int, name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not safe_name:
        raise ValueError("stage name must contain a filesystem-safe character")
    return f"stage_{index:02d}_{safe_name}"


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
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
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_trajectory(
    path: Path, bacteria_mucus: np.ndarray, macrophage_mucus: np.ndarray
) -> None:
    temporary_path = None
    time = np.arange(1, _OBSERVATION_COUNT + 1, dtype=float) * CALIBRATION_DT
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez(
                temporary,
                time=time,
                bacteria_mucus=bacteria_mucus,
                macrophage_mucus=macrophage_mucus,
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _initialize_evaluation_log(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_evaluation(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("evaluation log append was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_series(name: str, values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (_OBSERVATION_COUNT,):
        raise ValueError(f"{name} trajectory must have shape (83,)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} trajectory must contain only finite values")
    return result


def _validate_max_evaluations(max_evaluations: int | None) -> int | None:
    if max_evaluations is None:
        return None
    return validate_positive_integer(max_evaluations, "max_evaluations")


def _optimizer_value(result: Any, name: str, default: Any) -> Any:
    return getattr(result, name, default) if result is not None else default


def _optimizer_diagnostics(result: Any) -> dict[str, Any]:
    if result is None:
        return {
            "optimizer_success": None,
            "optimizer_status": None,
            "optimizer_message": None,
            "optimizer_iterations": None,
            "optimizer_reported_evaluations": None,
        }
    status = _optimizer_value(result, "status", None)
    iterations = _optimizer_value(result, "nit", None)
    evaluations = _optimizer_value(result, "nfev", None)
    return {
        "optimizer_success": bool(_optimizer_value(result, "success", False)),
        "optimizer_status": int(status) if status is not None else None,
        "optimizer_message": str(_optimizer_value(result, "message", "")),
        "optimizer_iterations": int(iterations) if iterations is not None else None,
        "optimizer_reported_evaluations": (
            int(evaluations) if evaluations is not None else None
        ),
    }


class CalibrationRunner:
    def __init__(
        self,
        simulator: CalibrationSimulator,
        initial_parameters: ParameterSet,
        observed_bacteria: np.ndarray,
        observed_macrophages: np.ndarray,
        output_dir: str | Path,
    ) -> None:
        self.simulator = simulator
        self.initial_parameters = initial_parameters
        self.observed_bacteria = _validate_series(
            "observed bacteria", observed_bacteria
        )
        self.observed_macrophages = _validate_series(
            "observed macrophage", observed_macrophages
        )
        self.output_dir = Path(output_dir)

    def run(
        self, pipeline: PipelineConfig, max_evaluations: int | None = None
    ) -> CalibrationResult:
        max_evaluations = _validate_max_evaluations(max_evaluations)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        current = self.initial_parameters
        stage_results = []

        for index, stage in enumerate(pipeline.stages, start=1):
            result = self._run_stage(
                index, stage, current, max_evaluations=max_evaluations
            )
            stage_results.append(result)
            if result.status == "failed":
                return CalibrationResult(
                    status="failed",
                    final_parameters=current,
                    stages=tuple(stage_results),
                    output_dir=self.output_dir,
                )
            current = result.final_parameters

        return CalibrationResult(
            status="complete",
            final_parameters=current,
            stages=tuple(stage_results),
            output_dir=self.output_dir,
        )

    def _run_stage(
        self,
        index: int,
        stage: StageConfig,
        input_parameters: ParameterSet,
        *,
        max_evaluations: int | None,
    ) -> StageResult:
        stage_dir = self.output_dir / _stage_directory_name(index, stage.name)
        stage_dir.mkdir(parents=False, exist_ok=False)
        input_path = stage_dir / "parameters_input.xml"
        output_path = stage_dir / "parameters.xml"
        evaluation_path = stage_dir / "evaluations.jsonl"
        write_parameter_xml(input_parameters, input_path)
        write_parameter_xml(input_parameters, output_path)
        _initialize_evaluation_log(evaluation_path)

        records: list[_EvaluationRecord] = []
        projection_changes: list[dict[str, float | str]] = []
        optimizer_result = None
        optimizer_settings = {
            "method": "L-BFGS-B",
            "maxiter": stage.optimizer.maxiter,
            "gtol": stage.optimizer.gtol,
            "ftol": stage.optimizer.ftol,
        }
        termination = "pre_optimizer_exception"
        running_status = {
            "status": "running",
            "stage": stage.name,
            "objective": stage.objective,
            "active_parameters": [spec.name for spec in stage.parameters],
            "calibration_dt": CALIBRATION_DT,
            "observation_count": _OBSERVATION_COUNT,
            "bound_projection": projection_changes,
            "started_at_utc": _utc_now(),
        }

        try:
            initial_vector = pack(input_parameters.values, stage.parameters)
            bounds = resolve_bounds(stage.parameters, input_parameters)
            initial_vector, projection_changes = project_initial_to_bounds(
                initial_vector, bounds, stage.parameters
            )
            running_status["bound_projection"] = projection_changes
            _atomic_write_json(stage_dir / "status.json", running_status)
            self._synchronize_simulator(input_parameters)

            objective_function = OBJECTIVES[stage.objective]

            def evaluate(vector: np.ndarray) -> float:
                if (
                    max_evaluations is not None
                    and len(records) >= max_evaluations
                ):
                    raise _EvaluationLimitReached
                updates = unpack(np.asarray(vector, dtype=float), stage.parameters)
                bacteria, macrophages = self.simulator.simulate(updates)
                bacteria = _validate_series("bacteria mucus", bacteria)
                macrophages = _validate_series("macrophage mucus", macrophages)
                loss = float(
                    objective_function(
                        bacteria,
                        macrophages,
                        self.observed_bacteria,
                        self.observed_macrophages,
                    )
                )
                if not np.isfinite(loss):
                    raise ValueError("calibration objective must be finite")
                record = _EvaluationRecord(
                    vector=np.asarray(vector, dtype=float).copy(),
                    parameters=updates,
                    loss=loss,
                    bacteria_mucus=bacteria.copy(),
                    macrophage_mucus=macrophages.copy(),
                )
                records.append(record)
                _append_evaluation(
                    evaluation_path,
                    {
                        "evaluation": len(records),
                        "parameters": updates,
                        "loss": loss,
                    },
                )
                return loss

            options = {
                "maxiter": stage.optimizer.maxiter,
                "gtol": stage.optimizer.gtol,
                "ftol": stage.optimizer.ftol,
            }
            if max_evaluations is not None:
                options["maxfun"] = max_evaluations

            termination = "optimizer_exception"
            try:
                optimizer_result = minimize(
                    evaluate,
                    initial_vector,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options=options,
                )
            except _EvaluationLimitReached:
                termination = "max_evaluations"
                raise RuntimeError(
                    "L-BFGS-B did not report success before "
                    f"max_evaluations={max_evaluations}"
                ) from None

            if (
                optimizer_result is None
                or not bool(_optimizer_value(optimizer_result, "success", False))
            ):
                termination = "optimizer_failure"
                raise RuntimeError(
                    "L-BFGS-B failed: "
                    + str(_optimizer_value(optimizer_result, "message", "unknown error"))
                )
            if not records:
                termination = "optimizer_contract_failure"
                raise RuntimeError("optimizer completed without an objective evaluation")
            termination = "optimizer_success"
            final_record = min(records, key=lambda record: record.loss)
            final_parameters = input_parameters.updated(final_record.parameters)
            self._synchronize_simulator(final_parameters)

            metrics = {
                "objective": stage.objective,
                "final_loss": final_record.loss,
                "best_evaluated_loss": final_record.loss,
                "best_evaluated_parameters": final_record.parameters,
                "evaluations": len(records),
                "termination": termination,
                **_optimizer_diagnostics(optimizer_result),
                "optimizer_settings": optimizer_settings,
                "max_evaluations": max_evaluations,
                "bound_projection": projection_changes,
                "checkpoint_source": "lowest_finite_evaluated_loss",
                "trajectory_kind": "checkpoint",
            }
            _atomic_write_trajectory(
                stage_dir / "trajectory.npz",
                final_record.bacteria_mucus,
                final_record.macrophage_mucus,
            )
            _atomic_write_json(stage_dir / "metrics.json", metrics)
            write_parameter_xml(final_parameters, output_path)
            _atomic_write_json(
                stage_dir / "status.json",
                {
                    **running_status,
                    "status": "complete",
                    "completed_at_utc": _utc_now(),
                    "evaluations": len(records),
                    "final_loss": final_record.loss,
                    "termination": termination,
                },
            )
            return StageResult(
                name=stage.name,
                status="complete",
                input_parameters=input_parameters,
                final_parameters=final_parameters,
                loss=final_record.loss,
                evaluations=len(records),
                output_dir=stage_dir,
            )
        except Exception as error:
            write_parameter_xml(input_parameters, output_path)
            error_data = {
                "type": type(error).__name__,
                "message": str(error),
            }
            best_record = min(records, key=lambda record: record.loss) if records else None
            if best_record is not None:
                _atomic_write_trajectory(
                    stage_dir / "trajectory.npz",
                    best_record.bacteria_mucus,
                    best_record.macrophage_mucus,
                )
            failure_metrics = {
                "objective": stage.objective,
                "final_loss": None,
                "best_evaluated_loss": (
                    best_record.loss if best_record is not None else None
                ),
                "best_evaluated_parameters": (
                    best_record.parameters if best_record is not None else None
                ),
                "evaluations": len(records),
                "termination": termination,
                **_optimizer_diagnostics(optimizer_result),
                "optimizer_settings": optimizer_settings,
                "max_evaluations": max_evaluations,
                "bound_projection": projection_changes,
                "checkpoint_source": "input_checkpoint_preserved",
                "trajectory_kind": (
                    "best_evaluation_diagnostic" if best_record is not None else None
                ),
                "error": error_data,
            }
            _atomic_write_json(stage_dir / "metrics.json", failure_metrics)
            _atomic_write_json(
                stage_dir / "status.json",
                {
                    **running_status,
                    "status": "failed",
                    "completed_at_utc": _utc_now(),
                    "evaluations": len(records),
                    "termination": termination,
                    "best_evaluated_loss": (
                        best_record.loss if best_record is not None else None
                    ),
                    "error": error_data,
                },
            )
            return StageResult(
                name=stage.name,
                status="failed",
                input_parameters=input_parameters,
                final_parameters=input_parameters,
                loss=None,
                evaluations=len(records),
                output_dir=stage_dir,
                error=error_data,
            )

    def _synchronize_simulator(self, parameters: ParameterSet) -> None:
        synchronize = getattr(self.simulator, "synchronize", None)
        if synchronize is not None:
            synchronize(parameters)

class FenicsCalibrationSimulator:
    def __init__(
        self,
        *,
        set_id: int | None = None,
        parameter_source: str | Path | None = None,
        resolution: int = 32,
        mesh_backend: str = "auto",
        model: Any = None,
        dt: float | None = None,
    ) -> None:
        if dt is not None:
            raise ValueError("calibration dt cannot be overridden at runtime")
        if model is None:
            if set_id is None or parameter_source is None:
                raise ValueError("set_id and parameter_source are required")
            from .model import build_model

            model = build_model(
                set_id=set_id,
                mode="calibration",
                resolution=resolution,
                mesh_backend=mesh_backend,
                parameter_source=parameter_source,
            )
        if float(model.config.dt) != CALIBRATION_DT:
            raise ValueError(f"calibration model dt must equal {CALIBRATION_DT!r}")
        self.model = model

    @property
    def resolved_mesh_backend(self) -> str:
        backend = getattr(self.model.config, "mesh_backend", None)
        if backend not in {"legacy_mshr", "rectangle"}:
            raise RuntimeError("calibration model did not resolve its mesh backend")
        return backend

    def synchronize(self, parameters: ParameterSet) -> None:
        values = parameters.values
        model_keys = set(self.model.parameters)
        updates = {
            name: float(values[name])
            for name in CALIBRATED_KEYS + FIXED_KEYS
            if name in model_keys
        }
        self.model.assign_parameters(updates)

    def simulate(
        self, updates: Mapping[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        unknown = sorted(set(updates) - set(CALIBRATED_KEYS))
        if unknown:
            raise ValueError(
                "calibration updates contain non-calibratable parameters: "
                + ", ".join(unknown)
            )
        self.model.dt.assign(CALIBRATION_DT)
        self.model.reset()
        self.model.assign_parameters(dict(updates))

        bacteria = np.empty(_OBSERVATION_COUNT, dtype=float)
        macrophages = np.empty(_OBSERVATION_COUNT, dtype=float)
        for index in range(_OBSERVATION_COUNT):
            self.model.step((index + 1) * CALIBRATION_DT)
            concentrations = self.model.concentrations()
            bacteria[index] = concentrations.bacteria_mucus
            macrophages[index] = concentrations.macrophage_mucus
        return bacteria, macrophages


FEniCSCalibrationSimulator = FenicsCalibrationSimulator


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_calibration(
    set_id: int,
    pipeline: PipelineConfig | str | Path,
    output_root: str | Path,
    run_id: str | None = None,
    *,
    mesh_backend: str = "auto",
    resolution: int = 32,
    max_evaluations: int | None = None,
    command_arguments: tuple[str, ...] = (),
) -> CalibrationResult:
    set_id = validate_set_id(set_id)
    resolution = validate_positive_integer(resolution, "resolution")
    max_evaluations = _validate_max_evaluations(max_evaluations)
    if isinstance(pipeline, PipelineConfig):
        pipeline_config = pipeline
        pipeline_source = None
    else:
        pipeline_source = Path(pipeline).expanduser().resolve()
        pipeline_config = load_pipeline(pipeline_source)

    initial_xml = parameter_path(_REPOSITORY_ROOT, set_id, calibrated=False).resolve()
    initial_parameters = read_parameter_xml(initial_xml)
    abm_path = (_REPOSITORY_ROOT / "data" / "abm" / f"data_{set_id}.npz").resolve()
    abm = load_abm(abm_path)

    selected_run_id = _validate_run_id(run_id or _default_run_id())
    run_dir = Path(output_root) / f"set_{set_id:03d}" / selected_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "set_id": set_id,
        "run_id": selected_run_id,
        "created_at_utc": _utc_now(),
        "command": [str(argument) for argument in command_arguments],
        "pipeline": str(pipeline_source) if pipeline_source is not None else None,
        "pipeline_sha256": (
            _sha256(pipeline_source) if pipeline_source is not None else None
        ),
        "initial_parameters": str(initial_xml),
        "initial_parameters_sha256": _sha256(initial_xml),
        "abm_data": str(abm_path),
        "abm_data_sha256": _sha256(abm_path),
        "calibration_dt": CALIBRATION_DT,
        "observation_count": _OBSERVATION_COUNT,
        "stages": [stage.name for stage in pipeline_config.stages],
        "active_parameters": [
            name
            for name in CALIBRATED_KEYS
            if any(
                name == spec.name
                for stage in pipeline_config.stages
                for spec in stage.parameters
            )
        ],
        "resolution": resolution,
        "requested_mesh_backend": mesh_backend,
        "mesh_backend": None,
        "max_evaluations": max_evaluations,
    }
    _atomic_write_json(run_dir / "metadata.json", metadata)
    _atomic_write_json(
        run_dir / "status.json",
        {
            "status": "running",
            "started_at_utc": _utc_now(),
            "calibration_dt": CALIBRATION_DT,
        },
    )

    try:
        simulator = FenicsCalibrationSimulator(
            set_id=set_id,
            parameter_source=initial_xml,
            resolution=resolution,
            mesh_backend=mesh_backend,
        )
        resolved_mesh_backend = simulator.resolved_mesh_backend
        metadata["mesh_backend"] = resolved_mesh_backend
        _atomic_write_json(run_dir / "metadata.json", metadata)
        if (
            mesh_backend != "auto"
            and resolved_mesh_backend != mesh_backend
        ):
            raise RuntimeError(
                "calibration model resolved a different explicit mesh backend"
            )
        runner = CalibrationRunner(
            simulator,
            initial_parameters,
            abm.bacteria,
            abm.macrophages,
            run_dir,
        )
        result = runner.run(
            pipeline_config,
            max_evaluations=max_evaluations,
        )
    except Exception as error:
        error_data = {"type": type(error).__name__, "message": str(error)}
        _atomic_write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "completed_at_utc": _utc_now(),
                "calibration_dt": CALIBRATION_DT,
                "error": error_data,
            },
        )
        return CalibrationResult(
            status="failed",
            final_parameters=initial_parameters,
            stages=(),
            output_dir=run_dir,
        )

    status = {
        "status": result.status,
        "calibration_dt": CALIBRATION_DT,
        "completed_at_utc": _utc_now(),
        "stages_completed": [
            stage.name for stage in result.stages if stage.status == "complete"
        ],
    }
    if result.status == "failed" and result.stages:
        status["error"] = result.stages[-1].error
    _atomic_write_json(run_dir / "status.json", status)
    return result


__all__ = [
    "CalibrationResult",
    "CalibrationRunner",
    "FEniCSCalibrationSimulator",
    "FenicsCalibrationSimulator",
    "StageResult",
    "run_calibration",
]
