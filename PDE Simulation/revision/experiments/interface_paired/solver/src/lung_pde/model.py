from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from dolfin import Constant, Expression, Function, LogLevel, as_vector, assemble, cells, set_log_level

from .constants import CALIBRATION_DT, REQUIRED_PARAMETER_KEYS
from .conservative_interface import replace_interface_exchange
from .data import ABMData, load_abm, validate_set_id
from .form_context import FormContext
from .forms import baseline, calibration, treatment, velocity
from .initial_conditions import initialize_state
from .mesh import MUCUS_FLAG, TISSUE_FLAG, MeshBundle, MeshConfig, build_mesh
from .parameters import parameter_path, read_parameter_xml
from .results import SnapshotData
from .solver import (
    ConstrainedNonlinearProblem,
    SolverSettings,
    compute_fixed_dofs,
    make_newton_solver,
)
from .velocity_fields import MucusVelocity, TreatmentVelocity, ZeroVelocity
from .validation import validate_positive_integer


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_MODES = ("calibration", "baseline", "velocity", "treatment")
_FORM_BUILDERS = {
    "calibration": calibration.build_form,
    "baseline": baseline.build_form,
    "velocity": velocity.build_form,
    "treatment": treatment.build_form,
}


class ConcentrationData(NamedTuple):
    bacteria_mucus: float
    bacteria_tissue: float
    macrophage_mucus: float
    macrophage_tissue: float
    mucus_density: float


def _default_parameter_source(set_id: int, mode: str) -> Path:
    return parameter_path(
        _REPOSITORY_ROOT,
        set_id,
        calibrated=mode != "calibration",
    )


def _resolve_parameter_source(
    set_id: int, mode: str, source: str | Path | None
) -> Path:
    if source is None:
        result = _default_parameter_source(set_id, mode)
    elif str(source) in {"initial", "calibrated"}:
        result = parameter_path(
            _REPOSITORY_ROOT,
            set_id,
            calibrated=str(source) == "calibrated",
        )
    else:
        result = Path(source).expanduser()
        if not result.is_absolute():
            result = Path.cwd() / result
    result = result.resolve()
    if not result.is_file():
        raise FileNotFoundError(f"parameter source does not exist: {result}")
    return result


def _solver_settings(mode: str) -> SolverSettings:
    if mode == "calibration":
        return SolverSettings(maximum_iterations=100)
    if mode in {"velocity", "treatment"}:
        return SolverSettings(linear_solver="gmres", preconditioner="ilu")
    return SolverSettings()


def _parameter_source_kind(source: Path) -> str:
    if source.parent.name in {"initial", "calibrated"}:
        return source.parent.name
    return "custom"


@dataclass(frozen=True)
class ModelConfig:
    set_id: int
    mode: str
    dt: float
    final_time: float
    snapshot_times: tuple[float, ...]
    resolution: int
    mesh_backend: str
    parameter_source: Path
    solver_settings: SolverSettings
    eta: float = 1.0
    danti: float = 1.0
    command_arguments: tuple[str, ...] = ()

    @property
    def T(self) -> float:
        return self.final_time

    @classmethod
    def for_mode(
        cls,
        set_id: int,
        mode: str,
        *,
        dt: float | None = None,
        resolution: int = 32,
        mesh_backend: str = "auto",
        eta: float = 1.0,
        danti: float = 1.0,
        parameter_source: str | Path | None = None,
        command_arguments: tuple[str, ...] = (),
    ) -> "ModelConfig":
        set_id = validate_set_id(set_id)
        if mode not in _MODES:
            raise ValueError(
                f"unknown model mode {mode!r}; expected {', '.join(_MODES)}"
            )
        abm = load_abm(_REPOSITORY_ROOT / "data" / "abm" / f"data_{set_id}.npz")
        defaults = {
            "calibration": (CALIBRATION_DT, float(abm.times[-1]), ()),
            "baseline": (0.01, float(abm.times[-1]), (0.0, 1.0, 2.0)),
            "velocity": (0.01, 9.0, (0.0, 1.0, 2.0)),
            "treatment": (0.1, 14.0, ()),
        }
        default_dt, final_time, snapshot_times = defaults[mode]
        dt = default_dt if dt is None else float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        resolution = validate_positive_integer(resolution, "resolution")
        return cls(
            set_id=set_id,
            mode=mode,
            dt=dt,
            final_time=final_time,
            snapshot_times=snapshot_times,
            resolution=resolution,
            mesh_backend=mesh_backend,
            parameter_source=_resolve_parameter_source(
                set_id, mode, parameter_source
            ),
            solver_settings=_solver_settings(mode),
            eta=float(eta),
            danti=float(danti),
            command_arguments=tuple(str(argument) for argument in command_arguments),
        )


@dataclass(frozen=True)
class ModelProvenance:
    set_id: int
    mode: str
    parameter_source: Path
    parameter_source_kind: str
    parameter_source_sha256: str
    resolution: int
    mesh_backend: str
    solver_settings: SolverSettings
    dt: float
    eta: float
    danti: float
    initial_bacteria: float
    initial_macrophages: float
    initial_parameter_values: tuple[tuple[str, float], ...]


@dataclass
class LungModel:
    config: ModelConfig
    provenance: ModelProvenance
    abm: ABMData
    mesh: MeshBundle
    parameter_values: dict[str, float]
    parameters: dict[str, Any]
    dt: Any
    eta: Any
    danti: Any
    velocity: Any
    w: Any
    w_n: Any
    initial_state: Any
    form: Any
    fixed_dofs: list[int]
    problem: Any
    solver: Any

    @property
    def mesh_bundle(self) -> MeshBundle:
        return self.mesh

    @property
    def effective_danti(self) -> float | None:
        if self.provenance.mode != "treatment":
            return None
        return (
            float(self.danti)
            * self.provenance.initial_macrophages
            * float(self.parameters["dB"])
        )

    def reset(self) -> None:
        self.w.assign(self.initial_state)
        self.w_n.assign(self.initial_state)
        self.velocity.t = 0.0

    def assign_parameters(self, updates: dict[str, float]) -> None:
        unknown = sorted(set(updates) - set(self.parameters))
        if unknown:
            raise KeyError(f"unknown model parameter: {', '.join(unknown)}")
        for name, value in updates.items():
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                raise ValueError(f"model parameter {name} must be finite")
            if name in {"kappaB", "kappaM"} and numeric_value < 0:
                raise ValueError(f"{name} must be non-negative")
            self.parameters[name].assign(numeric_value)
            self.parameter_values[name] = numeric_value

    def step(self, t: float):
        time = float(t)
        if not np.isfinite(time):
            raise ValueError("step time must be finite")
        self.velocity.t = time
        outcome = self.solver.solve(self.problem, self.w.vector())
        if isinstance(outcome, tuple) and len(outcome) > 1 and not outcome[1]:
            raise RuntimeError(f"Newton solve failed at t={time}")
        self.w_n.assign(self.w)
        return outcome

    def concentrations(self) -> ConcentrationData:
        dx = self.mesh.dx
        return ConcentrationData(
            bacteria_mucus=50.0 * float(assemble(self.w.sub(0) * dx(MUCUS_FLAG))),
            bacteria_tissue=50.0 * float(assemble(self.w.sub(1) * dx(TISSUE_FLAG))),
            macrophage_mucus=50.0
            * float(assemble(self.w.sub(2) * dx(MUCUS_FLAG))),
            macrophage_tissue=50.0
            * float(assemble(self.w.sub(3) * dx(TISSUE_FLAG))),
            mucus_density=50.0 * float(assemble(self.w.sub(4) * dx(MUCUS_FLAG))),
        )

    def _vertex_values(self, component: int, submesh, space) -> np.ndarray:
        output = Function(space)
        output.interpolate(self.w.sub(component))
        return np.asarray(output.compute_vertex_values(submesh), dtype=float)

    def snapshot(self, time: float | None = None) -> SnapshotData:
        snapshot_time = self.velocity.t if time is None else float(time)
        mucus_xy = np.asarray(self.mesh.mucus_mesh.coordinates(), dtype=float).copy()
        tissue_xy = np.asarray(self.mesh.tissue_mesh.coordinates(), dtype=float).copy()
        mucus_triangles = np.asarray(
            [cell.entities(0) for cell in cells(self.mesh.mucus_mesh)], dtype=np.int64
        ).reshape((-1, 3))
        tissue_triangles = np.asarray(
            [cell.entities(0) for cell in cells(self.mesh.tissue_mesh)], dtype=np.int64
        ).reshape((-1, 3))

        bacteria_mucus = self._vertex_values(
            0, self.mesh.mucus_mesh, self.mesh.mucus_space
        )
        bacteria_tissue = self._vertex_values(
            1, self.mesh.tissue_mesh, self.mesh.tissue_space
        )
        macrophage_mucus = self._vertex_values(
            2, self.mesh.mucus_mesh, self.mesh.mucus_space
        )
        macrophage_tissue = self._vertex_values(
            3, self.mesh.tissue_mesh, self.mesh.tissue_space
        )
        return SnapshotData(
            time=np.asarray([snapshot_time], dtype=float),
            mucus_xy=mucus_xy,
            mucus_triangles=mucus_triangles,
            tissue_xy=tissue_xy,
            tissue_triangles=tissue_triangles,
            bacteria_mucus=bacteria_mucus[np.newaxis, :],
            bacteria_tissue=bacteria_tissue[np.newaxis, :],
            macrophage_mucus=macrophage_mucus[np.newaxis, :],
            macrophage_tissue=macrophage_tissue[np.newaxis, :],
        )


def build_model(
    set_id,
    mode,
    resolution=32,
    mesh_backend="auto",
    eta=1.0,
    danti=1.0,
    parameter_source=None,
    *,
    dt=None,
    interface_permeabilities: tuple[float, float] | None = None,
    initial_loads: tuple[float, float] | None = None,
):
    if interface_permeabilities is not None:
        kappa_b, kappa_m = map(float, interface_permeabilities)
        if not np.all(np.isfinite([kappa_b, kappa_m])) or min(kappa_b, kappa_m) < 0:
            raise ValueError("interface permeabilities must be finite and non-negative")
    config = ModelConfig.for_mode(
        set_id,
        mode,
        dt=dt,
        resolution=resolution,
        mesh_backend=mesh_backend,
        eta=eta,
        danti=danti,
        parameter_source=parameter_source,
    )
    abm = load_abm(
        _REPOSITORY_ROOT / "data" / "abm" / f"data_{config.set_id}.npz"
    )
    if initial_loads is not None:
        b0, m0 = map(float, initial_loads)
        if not np.all(np.isfinite([b0, m0])) or min(b0, m0) < 0:
            raise ValueError("initial loads must be finite and non-negative")
        abm = replace(abm, initial_bacteria=b0, initial_macrophages=m0)
    parameter_source_sha256 = hashlib.sha256(
        config.parameter_source.read_bytes()
    ).hexdigest()
    parameter_set = read_parameter_xml(config.parameter_source)
    parameter_values = {
        name: float(parameter_set[name]) for name in REQUIRED_PARAMETER_KEYS
    }
    if interface_permeabilities is not None:
        parameter_values.update(kappaB=kappa_b, kappaM=kappa_m)
    parameters = {
        name: Constant(value, name=name) for name, value in parameter_values.items()
    }

    mesh = build_mesh(
        MeshConfig(
            resolution=config.resolution,
            backend=config.mesh_backend,
            treatment_boundary=config.mode == "treatment",
        )
    )
    config = replace(config, mesh_backend=mesh.backend)
    w, w_n = initialize_state(
        mesh,
        initial_bacteria=abm.initial_bacteria,
        initial_macrophages=abm.initial_macrophages,
    )
    initial_state = Function(mesh.function_space)
    initial_state.assign(w)

    dt = Constant(config.dt, name="dt")
    eta_constant = Constant(config.eta, name="eta")
    danti_constant = Constant(config.danti, name="danti")
    velocity_expression = {
        "calibration": ZeroVelocity,
        "baseline": ZeroVelocity,
        "velocity": MucusVelocity,
        "treatment": TreatmentVelocity,
    }[config.mode](t=0.0, degree=3)
    if config.mode in {"velocity", "treatment"}:
        scale, frequency = (1.0, np.pi) if config.mode == "velocity" else (1.44, 2.0 * np.pi)
        velocity_expression = Expression(
            ("0.5*scale*(1.0 - cos(frequency*t))*pow(x[1]-0.1,2)", "0.0"),
            t=0.0, scale=scale, frequency=frequency, degree=3,
        )
    context = FormContext(
        w=w,
        w_n=w_n,
        dt=dt,
        velocity=(
            as_vector((0.0, 0.0))
            if interface_permeabilities is not None and config.mode in {"baseline", "calibration"}
            else velocity_expression
        ),
        parameters=parameters,
        dx=mesh.dx,
        ds=mesh.ds,
        dS=mesh.dS,
        normal=mesh.normal,
        down_normal=mesh.down_normal,
        eta=eta_constant,
        danti=danti_constant,
        initial_macrophages=abm.initial_macrophages,
    )
    form = _FORM_BUILDERS[config.mode](context)
    if interface_permeabilities is not None:
        form = replace_interface_exchange(form, context, viscosity_scaled=config.mode == "treatment")
    fixed_dofs = compute_fixed_dofs(mesh)
    problem = ConstrainedNonlinearProblem(
        form.residual,
        form.jacobian,
        fixed_dofs,
    )
    solver = make_newton_solver(config.solver_settings)
    set_log_level(LogLevel.WARNING)

    provenance = ModelProvenance(
        set_id=config.set_id,
        mode=config.mode,
        parameter_source=config.parameter_source,
        parameter_source_kind=_parameter_source_kind(config.parameter_source),
        parameter_source_sha256=parameter_source_sha256,
        resolution=config.resolution,
        mesh_backend=mesh.backend,
        solver_settings=config.solver_settings,
        dt=config.dt,
        eta=config.eta,
        danti=config.danti,
        initial_bacteria=abm.initial_bacteria,
        initial_macrophages=abm.initial_macrophages,
        initial_parameter_values=tuple(parameter_values.items()),
    )
    return LungModel(
        config=config,
        provenance=provenance,
        abm=abm,
        mesh=mesh,
        parameter_values=parameter_values,
        parameters=parameters,
        dt=dt,
        eta=eta_constant,
        danti=danti_constant,
        velocity=velocity_expression,
        w=w,
        w_n=w_n,
        initial_state=initial_state,
        form=form,
        fixed_dofs=fixed_dofs,
        problem=problem,
        solver=solver,
    )
