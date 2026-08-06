import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.simulate import build_parser as baseline_parser
from scripts.simulate_velocity import build_parser as velocity_parser
from lung_pde.cli import resolve_simulation_inputs


BASELINE_SCRIPT = Path("scripts/simulate.py")
VELOCITY_SCRIPT = Path("scripts/simulate_velocity.py")


def _run_cli(script, arguments, *, env=None):
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _dolfin_guard_env(tmp_path):
    (tmp_path / "sitecustomize.py").write_text(
        "import builtins\n"
        "original_import = builtins.__import__\n"
        "def guarded_import(name, *args, **kwargs):\n"
        "    if name == 'dolfin' or name.startswith('dolfin.'):\n"
        "        raise RuntimeError('validate-only imported DOLFIN')\n"
        "    return original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded_import\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path("src").resolve()))
    )
    return env


def test_baseline_defaults_preserve_test_final_settings():
    args = baseline_parser().parse_args(["--set", "756"])

    assert args.dt == 0.01
    assert args.final_time is None
    assert args.snapshot_times == (0.0, 1.0, 2.0)
    assert args.parameter_source == "calibrated"


def test_velocity_defaults_preserve_test_v4_settings():
    args = velocity_parser().parse_args(["--set", "756"])

    assert args.dt == 0.01
    assert args.final_time == 9.0
    assert args.snapshot_times == (0.0, 1.0, 2.0)
    assert args.parameter_source == "calibrated"


@pytest.mark.parametrize(
    ("script", "mode", "source", "final_time"),
    [
        (BASELINE_SCRIPT, "baseline", "test_final.py", "13.833333333333334"),
        (VELOCITY_SCRIPT, "velocity", "test-v4.py", "9.0"),
    ],
)
def test_validate_only_resolves_inputs_without_importing_dolfin(
    tmp_path, script, mode, source, final_time
):
    completed = _run_cli(
        script,
        ["--set", "756", "--validate-only"],
        env=_dolfin_guard_env(tmp_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert "validation=ok" in completed.stdout
    assert f"mode={mode}" in completed.stdout
    assert f"authoritative_source={source}" in completed.stdout
    assert "data/abm/data_756.npz" in completed.stdout
    assert "data/parameters/calibrated/parameters_756.xml" in completed.stdout
    assert "parameter_source_kind=calibrated" in completed.stdout
    assert "dt=0.01" in completed.stdout
    assert f"final_time={final_time}" in completed.stdout
    assert "snapshot_times=0,1,2" in completed.stdout
    assert f"outputs/simulations/{mode}/set_756" in completed.stdout
    assert "validate-only imported DOLFIN" not in completed.stderr


def test_validate_only_keeps_exact_output_directory(tmp_path):
    destination = tmp_path / "one-result-set"

    completed = _run_cli(
        BASELINE_SCRIPT,
        [
            "--set",
            "756",
            "--output-dir",
            str(destination),
            "--validate-only",
        ],
    )

    assert completed.returncode == 0, completed.stderr
    assert f"output_dir={destination}" in completed.stdout
    assert "one-result-set/set_756" not in completed.stdout


def test_validate_only_rejects_uncovered_snapshot_within_alignment_tolerance():
    completed = _run_cli(
        BASELINE_SCRIPT,
        [
            "--set",
            "756",
            "--dt",
            "0.01",
            "--final-time",
            "0.019999995",
            "--snapshot-times",
            "0.02",
            "--validate-only",
        ],
    )

    assert completed.returncode == 2
    assert "final time must cover every snapshot time" in completed.stderr


@pytest.mark.parametrize(
    ("final_time", "snapshot_time"),
    [
        ("0.02", "0.02"),
        ("0.020000000000000004", "0.02"),
        ("0.02", "0.019999999999999997"),
    ],
)
def test_validate_only_accepts_covered_nearby_float_times(
    final_time, snapshot_time
):
    completed = _run_cli(
        BASELINE_SCRIPT,
        [
            "--set",
            "756",
            "--dt",
            "0.01",
            "--final-time",
            final_time,
            "--snapshot-times",
            snapshot_time,
            "--validate-only",
        ],
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--dt", "0"), "dt must be finite and positive"),
        (("--dt", "nan"), "dt must be finite and positive"),
        (("--dt", "inf"), "dt must be finite and positive"),
        (("--final-time", "-0.01"), "final time must be finite and non-negative"),
        (("--final-time", "nan"), "final time must be finite and non-negative"),
        (
            ("--snapshot-times", "-0.01"),
            "snapshot times must be finite and non-negative",
        ),
        (
            ("--snapshot-times", "nan"),
            "snapshot times must be finite and non-negative",
        ),
        (
            ("--final-time", "1", "--snapshot-times", "0", "2"),
            "final time must cover every snapshot time",
        ),
        (
            ("--final-time", "0.02", "--snapshot-times", "0", "0.015"),
            "snapshot times must align to dt",
        ),
    ],
)
def test_validate_only_rejects_invalid_times(arguments, message):
    completed = _run_cli(
        BASELINE_SCRIPT,
        ["--set", "756", *arguments, "--validate-only"],
    )

    assert completed.returncode == 2
    assert message in completed.stderr


@pytest.mark.parametrize("script", [BASELINE_SCRIPT, VELOCITY_SCRIPT])
@pytest.mark.parametrize("resolution", ["0", "-1"])
def test_validate_only_rejects_nonpositive_resolution(script, resolution):
    completed = _run_cli(
        script,
        ["--set", "756", "--resolution", resolution, "--validate-only"],
    )

    assert completed.returncode == 2
    assert "resolution must be a positive integer" in completed.stderr
    assert "validation=ok" not in completed.stdout


@pytest.mark.parametrize("resolution", [0, -1, True, 1.5, "4"])
def test_simulation_input_api_rejects_noninteger_or_nonpositive_resolution(
    tmp_path, resolution
):
    args = baseline_parser().parse_args(["--set", "756"])
    args.resolution = resolution

    with pytest.raises(ValueError, match="resolution must be a positive integer"):
        resolve_simulation_inputs(
            args,
            mode="baseline",
            repository_root=Path.cwd(),
        )
