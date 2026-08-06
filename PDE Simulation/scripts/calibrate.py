#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from lung_pde.calibration import run_calibration
from lung_pde.calibration_config import load_pipeline
from lung_pde.constants import CALIBRATED_KEYS
from lung_pde.data import load_abm
from lung_pde.parameters import parameter_path, read_parameter_xml
from lung_pde.validation import validate_positive_integer


def build_parser():
    parser = argparse.ArgumentParser(
        description="Calibrate the lung PDE model against ABM mucus trajectories."
    )
    parser.add_argument("--set", type=int, required=True, dest="set_id")
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/calibration")
    )
    parser.add_argument(
        "--mesh-backend",
        choices=("auto", "legacy_mshr", "rectangle"),
        default="auto",
    )
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--max-evaluations", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _validate_inputs(
    set_id: int,
    pipeline_path: Path,
    *,
    resolution: int,
    max_evaluations: int | None,
):
    validate_positive_integer(resolution, "resolution")
    if max_evaluations is not None:
        validate_positive_integer(max_evaluations, "max_evaluations")
    pipeline = load_pipeline(pipeline_path)
    initial_xml = parameter_path(_REPOSITORY_ROOT, set_id, calibrated=False).resolve()
    read_parameter_xml(initial_xml)
    abm_path = _REPOSITORY_ROOT / "data" / "abm" / f"data_{set_id}.npz"
    abm = load_abm(abm_path)
    return pipeline, initial_xml, abm


def _print_validation(
    set_id: int,
    pipeline_path: Path,
    *,
    resolution: int,
    max_evaluations: int | None,
) -> None:
    pipeline, initial_xml, abm = _validate_inputs(
        set_id,
        pipeline_path,
        resolution=resolution,
        max_evaluations=max_evaluations,
    )
    active = {
        spec.name for stage in pipeline.stages for spec in stage.parameters
    }
    print("validation=ok")
    print(f"set={set_id}")
    print(f"pipeline={pipeline_path.resolve()}")
    print(f"initial_parameters={initial_xml}")
    print(f"observations={abm.times.size}")
    print(f"dt={pipeline.calibration_dt}")
    print(f"resolution={resolution}")
    print(f"max_evaluations={max_evaluations}")
    print("stages:")
    for index, stage in enumerate(pipeline.stages, start=1):
        names = ",".join(spec.name for spec in stage.parameters)
        print(f"  {index}. {stage.name}: {names}")
    print(
        "active_parameters="
        + ",".join(name for name in CALIBRATED_KEYS if name in active)
    )


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        validate_positive_integer(args.resolution, "resolution")
        if args.max_evaluations is not None:
            validate_positive_integer(args.max_evaluations, "max_evaluations")
        if args.validate_only:
            _print_validation(
                args.set_id,
                args.pipeline,
                resolution=args.resolution,
                max_evaluations=args.max_evaluations,
            )
            return 0
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if argv is None:
        command_arguments = tuple(sys.argv)
    else:
        command_arguments = ("scripts/calibrate.py", *arguments)
    result = run_calibration(
        args.set_id,
        args.pipeline,
        args.output_dir,
        mesh_backend=args.mesh_backend,
        resolution=args.resolution,
        max_evaluations=args.max_evaluations,
        command_arguments=command_arguments,
    )
    print(f"status={result.status}")
    print(f"output={result.output_dir}")
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
