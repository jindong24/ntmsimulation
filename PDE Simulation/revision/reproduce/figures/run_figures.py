#!/usr/bin/env python3
"""Build paper-style figures from saved tau NPZ/CSV results; no PDE is run."""

import argparse
import os
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAPER_GRID = HERE.parents[1] / "inputs/treatment_grid"
VELOCITY_REFERENCE = None
VELOCITY_READER = HERE / "ntm_revision_fig_velocity.py"
SETS = (48, 83, 182, 188, 335, 586, 646, 756, 808, 843)


def output_path(path, source):
    output = path.resolve()
    if output.is_relative_to(source.resolve()):
        raise ValueError("output must be outside the input tree")
    if path.is_symlink() or output.exists():
        raise ValueError(f"output already exists: {path}; choose a fresh --output directory")
    return output


def require_files(paths):
    missing = [p for p in paths if not p.is_file()]
    if missing:
        extra = f" (and {len(missing) - 1} more)" if len(missing) > 1 else ""
        raise FileNotFoundError(f"missing input: {missing[0]}{extra}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=Path(os.environ.get("NTM_RUNS", HERE.parents[1] / "runs")))
    parser.add_argument("--output", type=Path, default=HERE / "figures",
                        help="fresh output directory outside the input tree")
    parser.add_argument("--only", choices=("all", "velocity", "treatment"), default="all")
    parser.add_argument("--sets", nargs="+", type=int, choices=SETS, default=SETS)
    parser.add_argument("--normalization", choices=("shared", "paper", "both"), default="shared")
    parser.add_argument("--treatment-reference", type=Path, default=PAPER_GRID)
    parser.add_argument("--velocity-reference", type=Path, default=VELOCITY_REFERENCE)
    parser.add_argument("--velocity-reader", type=Path, default=VELOCITY_READER)
    args = parser.parse_args()
    selected = tuple(f"{sid:03d}" for sid in dict.fromkeys(args.sets))
    do_velocity = args.only != "treatment"
    do_treatment = args.only != "velocity"
    if do_velocity and args.velocity_reference is None:
        parser.error("--velocity-reference is required for the original manuscript layout")
    if do_treatment and args.treatment_reference is None:
        parser.error("--treatment-reference is required for comparisons against the archived results")
    source = args.input_root.resolve()
    try:
        output = output_path(args.output, source)
        if do_velocity:
            cases = [source / "velocity" / f"set_{sid}" for sid in selected]
            require_files([args.velocity_reader, args.velocity_reference / "main.tex", *(
                case / name for case in cases for name in (
                    "calibration.csv", "baseline/spatial.npz", "velocity/spatial.npz"
                )
            )])
        if do_treatment:
            names = [f"{int(sid)}_new.csv" for sid in selected]
            require_files(
                directory / name
                for directory in (args.treatment_reference,
                                  source / "treatment/legacy_calibrated/data",
                                  source / "treatment/recalibrated/data")
                for name in names
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".render-", dir=output.parent) as temp:
            stage = Path(temp) / "figures"
            stage.mkdir()
            cache = Path(temp) / "cache"
            os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
            os.environ["XDG_CACHE_HOME"] = str(cache)
            os.environ["MPLBACKEND"] = "Agg"
            sys.dont_write_bytecode = True
            from matplotlib import font_manager

            font = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
            if font.is_file():
                font_manager.fontManager.addfont(font)
            if do_velocity:
                font_manager.findfont("Times New Roman", fallback_to_default=False)
            if do_velocity:
                import render_velocity

                render_velocity.render(source / "velocity", stage / "velocity", selected,
                                       args.velocity_reference, args.velocity_reader)
            if do_treatment:
                import treatment

                modes = ("shared", "paper") if args.normalization == "both" else (args.normalization,)
                treatment.render(source / "treatment", stage / "treatment", selected,
                                 args.treatment_reference, modes)
            output.parent.mkdir(parents=True, exist_ok=True)
            stage.rename(output)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Figures: {output}")


if __name__ == "__main__":
    main()
