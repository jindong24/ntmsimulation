#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import math
from multiprocessing import get_context
from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from lung_pde.data import load_abm
from lung_pde.parameters import parameter_path, read_parameter_xml
from lung_pde.treatment import (
    CASE_SCHEMA_VERSION,
    CASE_STATUS_SUCCESS,
    TreatmentCase,
    aggregate_treatment_cases,
    case_output_path,
    compute_runtime_revision,
    exclusive_case_lock,
    iter_treatment_cases,
    load_case_result,
    load_treatment_grid,
    resolve_requested_mesh_backend,
    write_case_result,
)


@dataclass(frozen=True)
class TreatmentInputs:
    set_id: int
    cases: tuple[TreatmentCase, ...]
    full_grid: bool
    output_dir: Path
    workers: int
    resolution: int
    mesh_backend: str
    resolved_mesh_backend: str
    dt: float
    final_time: float
    parameter_source: Path
    parameter_source_sha256: str
    runtime_revision: str
    initial_bacteria: float
    initial_macrophages: float
    dB: float

    @property
    def expected_identity(self) -> dict:
        return {
            "set_id": self.set_id,
            "parameter_source_sha256": self.parameter_source_sha256,
            "runtime_revision": self.runtime_revision,
            "B0": self.initial_bacteria,
            "M0": self.initial_macrophages,
            "dB": self.dB,
            "mesh_backend": self.mesh_backend,
            "resolved_mesh_backend": self.resolved_mesh_backend,
            "resolution": self.resolution,
            "dt": self.dt,
            "final_time": self.final_time,
        }


@dataclass(frozen=True)
class CaseRunRequest:
    case: TreatmentCase
    output_path: Path
    set_id: int
    resolution: int
    mesh_backend: str
    resolved_mesh_backend: str
    dt: float
    final_time: float
    parameter_source: Path
    parameter_source_sha256: str
    runtime_revision: str
    initial_bacteria: float
    initial_macrophages: float
    dB: float
    command_arguments: tuple[str, ...]
    overwrite: bool

    @property
    def expected_identity(self) -> dict:
        return {
            "set_id": self.set_id,
            "parameter_source_sha256": self.parameter_source_sha256,
            "runtime_revision": self.runtime_revision,
            "B0": self.initial_bacteria,
            "M0": self.initial_macrophages,
            "dB": self.dB,
            "mesh_backend": self.mesh_backend,
            "resolved_mesh_backend": self.resolved_mesh_backend,
            "resolution": self.resolution,
            "dt": self.dt,
            "final_time": self.final_time,
        }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run and aggregate the calibrated treatment eta/danti sweep.",
        allow_abbrev=False,
    )
    parser.add_argument("--set", dest="set_id", type=int, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/treatment_grid.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument(
        "--mesh-backend",
        choices=("auto", "legacy_mshr", "rectangle"),
        default="auto",
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--final-time", type=float, default=14.0)
    parser.add_argument("--case", type=float, nargs=2, metavar=("ETA", "DANTI"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--aggregate-only", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--parameter-source",
        choices=("calibrated",),
        default="calibrated",
        help=argparse.SUPPRESS,
    )
    return parser


def _resolve_config_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    current = Path.cwd() / path
    if current.is_file():
        return current
    return _REPOSITORY_ROOT / path


def _resolve_output_dir(path: Path | None, set_id: int) -> Path:
    if path is None:
        return _REPOSITORY_ROOT / "outputs" / "treatment" / f"set_{set_id:03d}"
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded


def _validate_numerical_settings(args) -> None:
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.resolution <= 0:
        raise ValueError("resolution must be positive")
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(args.final_time) or args.final_time < 0.0:
        raise ValueError("final time must be finite and non-negative")
    if args.case is not None and args.aggregate_only:
        raise ValueError("--case cannot be combined with --aggregate-only")
    if args.overwrite and (args.aggregate_only or args.validate_only):
        mode = "--aggregate-only" if args.aggregate_only else "--validate-only"
        raise ValueError(f"--overwrite cannot be combined with {mode}")
    if args.aggregate_only and args.workers != 1:
        raise ValueError("--workers cannot be combined with --aggregate-only")


def _validate_mode_arguments(args, arguments: list[str]) -> None:
    workers_was_requested = any(
        argument == "--workers" or argument.startswith("--workers=")
        for argument in arguments
    )
    if args.aggregate_only and workers_was_requested:
        raise ValueError("--workers cannot be combined with --aggregate-only")


def resolve_treatment_inputs(args) -> TreatmentInputs:
    _validate_numerical_settings(args)
    if args.case is None:
        grid = load_treatment_grid(_resolve_config_path(args.config))
        cases = tuple(iter_treatment_cases(grid))
        full_grid = True
    else:
        cases = (TreatmentCase(args.case[0], args.case[1]),)
        full_grid = False

    abm_source = (
        _REPOSITORY_ROOT / "data" / "abm" / f"data_{args.set_id}.npz"
    ).resolve()
    abm = load_abm(abm_source)
    source = parameter_path(
        _REPOSITORY_ROOT,
        args.set_id,
        calibrated=True,
    ).resolve()
    parameters = read_parameter_xml(source)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()

    return TreatmentInputs(
        set_id=args.set_id,
        cases=cases,
        full_grid=full_grid,
        output_dir=_resolve_output_dir(args.output_dir, args.set_id),
        workers=args.workers,
        resolution=args.resolution,
        mesh_backend=args.mesh_backend,
        resolved_mesh_backend=resolve_requested_mesh_backend(args.mesh_backend),
        dt=float(args.dt),
        final_time=float(args.final_time),
        parameter_source=source,
        parameter_source_sha256=checksum,
        runtime_revision=compute_runtime_revision(),
        initial_bacteria=float(abm.initial_bacteria),
        initial_macrophages=float(abm.initial_macrophages),
        dB=float(parameters["dB"]),
    )


def print_treatment_validation(inputs: TreatmentInputs) -> None:
    print("validation=ok")
    print("mode=treatment")
    print("authoritative_source=test_anti.py")
    print(f"set_id={inputs.set_id}")
    print(f"parameter_source={inputs.parameter_source}")
    print("parameter_source_kind=calibrated")
    print(f"parameter_source_sha256={inputs.parameter_source_sha256}")
    print(f"runtime_revision={inputs.runtime_revision}")
    print(f"mesh_backend={inputs.mesh_backend}")
    print(f"resolved_mesh_backend={inputs.resolved_mesh_backend}")
    print(f"resolution={inputs.resolution}")
    print(f"dt={inputs.dt}")
    print(f"final_time={inputs.final_time}")
    print(f"case_count={len(inputs.cases)}")
    if len(inputs.cases) == 1:
        print(f"case_keys={inputs.cases[0].key}")
    print(f"output_dir={inputs.output_dir}")


def _solve_case(request: CaseRunRequest) -> None:
    if compute_runtime_revision() != request.runtime_revision:
        raise RuntimeError("treatment runtime revision does not match parent request")
    # DOLFIN stays behind this worker-only import boundary.
    from lung_pde.model import build_model
    from lung_pde.simulation import run_simulation

    model = build_model(
        set_id=request.set_id,
        mode="treatment",
        resolution=request.resolution,
        mesh_backend=request.mesh_backend,
        eta=request.case.eta,
        danti=request.case.danti_input,
        parameter_source=request.parameter_source,
        dt=request.dt,
    )
    config = replace(
        model.config,
        final_time=request.final_time,
        snapshot_times=(),
        command_arguments=request.command_arguments,
    )
    result = run_simulation(model, config)
    if compute_runtime_revision() != request.runtime_revision:
        raise RuntimeError("treatment runtime revision changed while case was running")
    if model.provenance.parameter_source_sha256 != request.parameter_source_sha256:
        raise RuntimeError("parameter checksum changed while treatment case was running")
    scientific_identity = {
        "resolved_mesh_backend": model.provenance.mesh_backend,
        "B0": model.provenance.initial_bacteria,
        "M0": model.provenance.initial_macrophages,
        "dB": float(model.parameters["dB"]),
    }
    expected_scientific_identity = {
        "resolved_mesh_backend": request.resolved_mesh_backend,
        "B0": request.initial_bacteria,
        "M0": request.initial_macrophages,
        "dB": request.dB,
    }
    mismatches = [
        name
        for name, expected in expected_scientific_identity.items()
        if scientific_identity[name] != expected
    ]
    if mismatches:
        raise RuntimeError(
            "treatment model identity changed while case was running: "
            + ", ".join(mismatches)
        )

    record = request.case.result_row(
        set_id=request.set_id,
        initial_bacteria=model.provenance.initial_bacteria,
        initial_macrophages=model.provenance.initial_macrophages,
        dB=float(model.parameters["dB"]),
        outputs=(
            result.bacteria_mucus[-1],
            result.bacteria_tissue[-1],
            result.macrophage_mucus[-1],
            result.macrophage_tissue[-1],
        ),
    )
    record.update(
        {
            "schema_version": CASE_SCHEMA_VERSION,
            "parameter_source": str(model.provenance.parameter_source),
            "parameter_source_kind": model.provenance.parameter_source_kind,
            "parameter_source_sha256": model.provenance.parameter_source_sha256,
            "runtime_revision": request.runtime_revision,
            "mesh_backend": request.mesh_backend,
            "resolved_mesh_backend": scientific_identity[
                "resolved_mesh_backend"
            ],
            "resolution": model.provenance.resolution,
            "dt": request.dt,
            "final_time": request.final_time,
            "status": CASE_STATUS_SUCCESS,
        }
    )
    write_case_result(request.output_path, record)


def _execute_case(request: CaseRunRequest) -> str:
    with exclusive_case_lock(request.output_path):
        if request.output_path.exists() and not request.overwrite:
            try:
                load_case_result(
                    request.output_path,
                    expected_case=request.case,
                    expected_identity=request.expected_identity,
                )
            except ValueError as error:
                raise ValueError(
                    f"existing case {request.output_path} does not match this run; "
                    f"use --overwrite ({error})"
                ) from error
            return "skipped"
        _solve_case(request)
        return "completed"


def _execute_case_captured(request: CaseRunRequest):
    try:
        return request.case.key, _execute_case(request), None
    except Exception as error:
        return request.case.key, None, f"{type(error).__name__}: {error}"


def _run_pending_cases(requests: tuple[CaseRunRequest, ...], workers: int):
    paths = [request.output_path for request in requests]
    if len(paths) != len(set(paths)):
        raise ValueError("treatment workers must have disjoint case output paths")
    if workers == 1:
        outcomes = [_execute_case_captured(request) for request in requests]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
        ) as executor:
            outcomes = list(
                executor.map(_execute_case_captured, requests, chunksize=1)
            )
    failures = [
        f"{key}: {error}" for key, _, error in outcomes if error is not None
    ]
    if failures:
        raise RuntimeError("treatment case failures: " + "; ".join(failures))
    return outcomes


def _run_cases(
    inputs: TreatmentInputs,
    *,
    overwrite: bool,
    command_arguments: tuple[str, ...],
) -> tuple[int, int]:
    requests = tuple(
        CaseRunRequest(
            case=case,
            output_path=case_output_path(inputs.output_dir, case),
            set_id=inputs.set_id,
            resolution=inputs.resolution,
            mesh_backend=inputs.mesh_backend,
            resolved_mesh_backend=inputs.resolved_mesh_backend,
            dt=inputs.dt,
            final_time=inputs.final_time,
            parameter_source=inputs.parameter_source,
            parameter_source_sha256=inputs.parameter_source_sha256,
            runtime_revision=inputs.runtime_revision,
            initial_bacteria=inputs.initial_bacteria,
            initial_macrophages=inputs.initial_macrophages,
            dB=inputs.dB,
            command_arguments=command_arguments,
            overwrite=overwrite,
        )
        for case in inputs.cases
    )
    outcomes = _run_pending_cases(requests, inputs.workers)
    completed = sum(status == "completed" for _, status, _ in outcomes)
    skipped = sum(status == "skipped" for _, status, _ in outcomes)
    return completed, skipped


def _aggregate(inputs: TreatmentInputs) -> Path:
    return aggregate_treatment_cases(
        inputs.output_dir / "cases",
        inputs.output_dir / "treatment.csv",
        expected_cases=inputs.cases,
        expected_identity=inputs.expected_identity,
    )


def main(argv=None):
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        _validate_mode_arguments(args, arguments)
        inputs = resolve_treatment_inputs(args)
        if args.validate_only:
            print_treatment_validation(inputs)
            return 0
        if args.aggregate_only:
            output = _aggregate(inputs)
            print("status=aggregated")
            print(f"case_count={len(inputs.cases)}")
            print(f"output={output}")
            return 0

        command_arguments = (
            tuple(sys.argv)
            if argv is None
            else ("scripts/sweep_danti_eta.py", *arguments)
        )
        completed, skipped = _run_cases(
            inputs,
            overwrite=args.overwrite,
            command_arguments=command_arguments,
        )
        if inputs.full_grid:
            output = _aggregate(inputs)
            print(f"output={output}")
        print("status=skipped" if completed == 0 and skipped else "status=complete")
        print(f"completed={completed}")
        print(f"skipped={skipped}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
