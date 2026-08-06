from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys

from .data import load_abm
from .parameters import parameter_path, read_parameter_xml
from .validation import validate_positive_integer


@dataclass(frozen=True)
class SimulationInputs:
    set_id: int
    mode: str
    dt: float
    final_time: float
    snapshot_times: tuple[float, ...]
    output_dir: Path
    resolution: int
    mesh_backend: str
    abm_source: Path
    parameter_source: Path


def add_simulation_arguments(parser, *, default_final_time):
    parser.add_argument("--set", dest="set_id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument(
        "--mesh-backend",
        choices=("auto", "legacy_mshr", "rectangle"),
        default="auto",
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--final-time", type=float, default=default_final_time)
    parser.add_argument(
        "--snapshot-times",
        type=float,
        nargs="+",
        default=(0.0, 1.0, 2.0),
    )
    parser.add_argument(
        "--parameter-source",
        choices=("calibrated",),
        default="calibrated",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _validate_times(
    dt: float, final_time: float, snapshot_times: tuple[float, ...]
) -> None:
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(final_time) or final_time < 0.0:
        raise ValueError("final time must be finite and non-negative")
    if any(not math.isfinite(time) or time < 0.0 for time in snapshot_times):
        raise ValueError("snapshot times must be finite and non-negative")

    tolerance = max(1.0e-12, abs(dt) * 1.0e-6)
    if snapshot_times and max(snapshot_times) > final_time:
        raise ValueError("final time must cover every snapshot time")
    for time in snapshot_times:
        quotient = time / dt
        if not math.isfinite(quotient):
            raise ValueError("snapshot times must align to dt")
        nearest = round(quotient) * dt
        if abs(time - nearest) > tolerance:
            raise ValueError("snapshot times must align to dt")


def resolve_simulation_inputs(args, *, mode: str, repository_root: Path):
    resolution = validate_positive_integer(args.resolution, "resolution")
    abm_path = (
        repository_root / "data" / "abm" / f"data_{args.set_id}.npz"
    ).resolve()
    abm = load_abm(abm_path)
    parameter_source = parameter_path(
        repository_root, args.set_id, calibrated=True
    ).resolve()
    read_parameter_xml(parameter_source)

    dt = float(args.dt)
    final_time = (
        float(abm.times[-1])
        if args.final_time is None
        else float(args.final_time)
    )
    snapshot_times = tuple(float(time) for time in args.snapshot_times)
    _validate_times(dt, final_time, snapshot_times)

    if args.output_dir is None:
        output_dir = (
            repository_root
            / "outputs"
            / "simulations"
            / mode
            / f"set_{args.set_id:03d}"
        )
    else:
        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir

    return SimulationInputs(
        set_id=args.set_id,
        mode=mode,
        dt=dt,
        final_time=final_time,
        snapshot_times=snapshot_times,
        output_dir=output_dir.resolve(),
        resolution=resolution,
        mesh_backend=args.mesh_backend,
        abm_source=abm_path,
        parameter_source=parameter_source,
    )


def _format_time(value: float) -> str:
    return str(int(value)) if value.is_integer() else repr(value)


def print_simulation_validation(
    inputs: SimulationInputs, *, authoritative_source: str
) -> None:
    print("validation=ok")
    print(f"mode={inputs.mode}")
    print(f"authoritative_source={authoritative_source}")
    print(f"set={inputs.set_id}")
    print(f"abm_source={inputs.abm_source}")
    print(f"parameter_source={inputs.parameter_source}")
    print("parameter_source_kind=calibrated")
    print(f"dt={inputs.dt}")
    print(f"final_time={inputs.final_time!r}")
    print(
        "snapshot_times="
        + ",".join(_format_time(time) for time in inputs.snapshot_times)
    )
    print(f"output_dir={inputs.output_dir}")


def run_simulation_inputs(
    inputs: SimulationInputs, *, command_arguments: tuple[str, ...]
) -> None:
    from .model import build_model
    from .results import save_simulation_result
    from .simulation import run_simulation

    model = build_model(
        set_id=inputs.set_id,
        mode=inputs.mode,
        dt=inputs.dt,
        resolution=inputs.resolution,
        mesh_backend=inputs.mesh_backend,
        parameter_source=inputs.parameter_source,
    )
    config = replace(
        model.config,
        final_time=inputs.final_time,
        snapshot_times=inputs.snapshot_times,
        command_arguments=command_arguments,
    )
    result = run_simulation(model, config)
    save_simulation_result(result, inputs.output_dir)


def simulation_main(
    parser,
    *,
    mode: str,
    authoritative_source: str,
    repository_root: Path,
    script_name: str,
    argv=None,
):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        inputs = resolve_simulation_inputs(
            args,
            mode=mode,
            repository_root=repository_root,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if args.validate_only:
        print_simulation_validation(
            inputs,
            authoritative_source=authoritative_source,
        )
        return 0

    command_arguments = (
        tuple(sys.argv)
        if argv is None
        else (script_name, *arguments)
    )
    run_simulation_inputs(inputs, command_arguments=command_arguments)
    print("status=complete")
    print(f"output={inputs.output_dir}")
    return 0
