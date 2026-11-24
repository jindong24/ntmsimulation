from dataclasses import dataclass
from typing import Any


@dataclass
class FormContext:
    w: Any
    w_n: Any
    dt: Any
    velocity: Any
    parameters: dict[str, Any]
    dx: Any
    ds: Any
    dS: Any
    normal: Any
    down_normal: Any
    eta: Any = None
    danti: Any = None
    initial_macrophages: float | None = None
    mean_viscosity_velocity: bool = False


@dataclass
class FormBundle:
    residual: Any
    jacobian: Any
    terms: tuple[Any, Any, Any, Any, Any]
