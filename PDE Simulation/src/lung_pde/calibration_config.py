from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from .constants import CALIBRATED_KEYS, CALIBRATION_DT
from .objectives import OBJECTIVES
from .parameters import ParameterSet


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    transform: str
    lower: float | str
    upper: float | str


@dataclass(frozen=True)
class OptimizerSpec:
    maxiter: int
    gtol: float
    ftol: float


@dataclass(frozen=True)
class StageConfig:
    name: str
    objective: str
    parameters: tuple[ParameterSpec, ...]
    optimizer: OptimizerSpec


@dataclass(frozen=True)
class PipelineConfig:
    calibration_dt: float
    stages: tuple[StageConfig, ...]


def pack(values: Mapping[str, float], specs: tuple[ParameterSpec, ...]) -> np.ndarray:
    packed = []
    for spec in specs:
        value = float(values[spec.name])
        if spec.transform == "log" and value <= 0:
            raise ValueError(f"{spec.name} must be positive for a log transform")
        packed.append(log(value) if spec.transform == "log" else value)
    return np.array(packed, dtype=float)


def unpack(vector: np.ndarray, specs: tuple[ParameterSpec, ...]) -> dict[str, float]:
    _require_matching_length(vector, specs, "vector")
    return {
        spec.name: exp(float(value)) if spec.transform == "log" else float(value)
        for spec, value in zip(specs, vector)
    }


def resolve_bounds(
    specs: tuple[ParameterSpec, ...], parameters: ParameterSet | Mapping[str, float]
) -> tuple[tuple[float, float], ...]:
    values = parameters.values if isinstance(parameters, ParameterSet) else parameters
    bounds = []
    for spec in specs:
        lower = _resolve_bound(spec.lower, values)
        upper = _resolve_bound(spec.upper, values)
        if spec.transform == "log":
            if lower <= 0 or upper <= 0:
                raise ValueError(f"{spec.name} log bounds must be positive")
            lower, upper = log(lower), log(upper)
        if lower > upper:
            raise ValueError(f"{spec.name} lower bound exceeds upper bound")
        bounds.append((lower, upper))
    return tuple(bounds)


def project_initial_to_bounds(
    vector: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    specs: tuple[ParameterSpec, ...],
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    _require_matching_length(vector, specs, "vector")
    _require_matching_length(bounds, specs, "bounds")
    projected = np.clip(np.asarray(vector, dtype=float), *np.asarray(bounds, dtype=float).T)
    changes = []
    for spec, original, updated in zip(specs, vector, projected):
        if original != updated:
            changes.append({
                "name": spec.name,
                "original": _physical_value(float(original), spec),
                "projected": _physical_value(float(updated), spec),
            })
    return projected, changes


def load_pipeline(path: str | Path) -> PipelineConfig:
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("calibration configuration must be a mapping")

    calibration_dt = document.get("calibration_dt")
    if not isinstance(calibration_dt, (int, float)) or calibration_dt != CALIBRATION_DT:
        raise ValueError(f"calibration_dt must equal {CALIBRATION_DT!r}")

    raw_stages = document.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("stages must be a non-empty list")
    return PipelineConfig(
        calibration_dt=float(calibration_dt),
        stages=tuple(_load_stage(raw_stage) for raw_stage in raw_stages),
    )


def _load_stage(raw_stage: object) -> StageConfig:
    if not isinstance(raw_stage, dict):
        raise ValueError("each stage must be a mapping")
    parameters = _load_parameters(raw_stage.get("parameters"))

    name = raw_stage.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("stage name must be a non-empty string")
    objective = raw_stage.get("objective")
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective}")
    return StageConfig(
        name=name,
        objective=objective,
        parameters=parameters,
        optimizer=_load_optimizer(raw_stage.get("optimizer")),
    )


def _load_parameters(raw_parameters: object) -> tuple[ParameterSpec, ...]:
    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise ValueError("stage parameters must be a non-empty list")
    specs = []
    names = set()
    for raw_parameter in raw_parameters:
        if not isinstance(raw_parameter, dict):
            raise ValueError("each parameter must be a mapping")
        name = raw_parameter.get("name")
        if name not in CALIBRATED_KEYS:
            raise ValueError(f"{name} is not calibratable")
        if name in names:
            raise ValueError(f"duplicate parameter in stage: {name}")
        names.add(name)

        transform = raw_parameter.get("transform")
        if transform not in {"linear", "log"}:
            raise ValueError(f"{name} transform must be linear or log")
        lower = _load_bound(raw_parameter.get("lower"), name, "lower")
        upper = _load_bound(raw_parameter.get("upper"), name, "upper")
        if transform == "log":
            for bound_name, bound in (("lower", lower), ("upper", upper)):
                if isinstance(bound, float) and bound <= 0:
                    raise ValueError(f"{name} {bound_name} log bound must be positive")
        specs.append(ParameterSpec(name, transform, lower, upper))
    return tuple(specs)


def _load_bound(value: object, name: str, bound_name: str) -> float | str:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.startswith("parameter:") and value[10:]:
        return value
    raise ValueError(f"{name} {bound_name} bound must be a number or parameter reference")


def _load_optimizer(raw_optimizer: object) -> OptimizerSpec:
    if not isinstance(raw_optimizer, dict):
        raise ValueError("optimizer must be a mapping")
    maxiter = raw_optimizer.get("maxiter")
    gtol = raw_optimizer.get("gtol")
    ftol = raw_optimizer.get("ftol")
    if not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter < 0:
        raise ValueError("optimizer maxiter must be a non-negative integer")
    if not isinstance(gtol, (int, float)) or not isinstance(ftol, (int, float)):
        raise ValueError("optimizer gtol and ftol must be numeric")
    return OptimizerSpec(maxiter=maxiter, gtol=float(gtol), ftol=float(ftol))


def _resolve_bound(bound: float | str, values: Mapping[str, float]) -> float:
    if isinstance(bound, str):
        reference = bound.removeprefix("parameter:")
        try:
            return float(values[reference])
        except KeyError as error:
            raise ValueError(f"unknown parameter bound reference: {reference}") from error
    return float(bound)


def _physical_value(value: float, spec: ParameterSpec) -> float:
    return exp(value) if spec.transform == "log" else value


def _require_matching_length(values: object, specs: tuple[ParameterSpec, ...], label: str) -> None:
    if len(values) != len(specs):
        raise ValueError(f"{label} length must match parameter specifications")
