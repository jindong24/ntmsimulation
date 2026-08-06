import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from lung_pde.constants import CALIBRATED_KEYS


SCRIPT = Path("scripts/calibrate.py")
PIPELINE = Path("configs/calibration/legacy_reconstructed.yaml")


def _load_script():
    spec = importlib.util.spec_from_file_location("calibrate_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validate_only_loads_every_input_without_importing_dolfin(tmp_path):
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

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--set",
            "756",
            "--pipeline",
            str(PIPELINE),
            "--validate-only",
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "aaanew2_parameter_subset" in completed.stdout
    assert "aaasimple_parameter_subset" in completed.stdout
    assert "test_lamB2_parameter_subset" in completed.stdout
    assert "dt=0.16666666666666666" in completed.stdout
    assert "data/parameters/initial/parameters_756.xml" in completed.stdout
    assert "observations=83" in completed.stdout
    for name in CALIBRATED_KEYS:
        assert name in completed.stdout


def test_parser_has_required_calibration_arguments_and_defaults():
    module = _load_script()

    args = module.build_parser().parse_args(
        ["--set", "756", "--pipeline", str(PIPELINE)]
    )

    assert args.set_id == 756
    assert args.pipeline == PIPELINE
    assert args.output_dir == Path("outputs/calibration")
    assert args.mesh_backend == "auto"
    assert args.resolution == 32
    assert args.max_evaluations is None
    assert args.validate_only is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--resolution", "0"), "resolution must be a positive integer"),
        (("--resolution", "-1"), "resolution must be a positive integer"),
        (("--max-evaluations", "0"), "max_evaluations must be a positive integer"),
        (("--max-evaluations", "-1"), "max_evaluations must be a positive integer"),
    ],
)
def test_validate_only_rejects_invalid_runtime_settings_without_dolfin(
    tmp_path, arguments, message
):
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
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--set",
            "756",
            "--pipeline",
            str(PIPELINE),
            *arguments,
            "--validate-only",
        ],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (str(tmp_path), str(Path("src").resolve()))
            ),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert message in completed.stderr
    assert "validation=ok" not in completed.stdout
    assert "validate-only imported DOLFIN" not in completed.stderr


def test_normal_cli_forwards_original_command_metadata(tmp_path, monkeypatch, capsys):
    module = _load_script()
    captured = {}
    output_dir = tmp_path / "set_756" / "run"

    def fake_run_calibration(set_id, pipeline, root, **kwargs):
        captured.update(
            set_id=set_id,
            pipeline=pipeline,
            root=root,
            kwargs=kwargs,
        )
        return type(
            "Result",
            (),
            {"status": "complete", "output_dir": output_dir},
        )()

    monkeypatch.setattr(module, "run_calibration", fake_run_calibration)
    arguments = [
        "--set",
        "756",
        "--pipeline",
        str(PIPELINE),
        "--output-dir",
        str(tmp_path),
        "--mesh-backend",
        "rectangle",
        "--resolution",
        "4",
        "--max-evaluations",
        "1",
    ]

    exit_code = module.main(arguments)

    assert exit_code == 0
    assert captured["set_id"] == 756
    assert captured["pipeline"] == PIPELINE
    assert captured["root"] == tmp_path
    assert captured["kwargs"]["mesh_backend"] == "rectangle"
    assert captured["kwargs"]["resolution"] == 4
    assert captured["kwargs"]["max_evaluations"] == 1
    assert captured["kwargs"]["command_arguments"] == (
        str(SCRIPT),
        *arguments,
    )
    assert str(output_dir) in capsys.readouterr().out


def test_normal_cli_returns_failure_when_calibration_fails(
    tmp_path, monkeypatch, capsys
):
    module = _load_script()
    output_dir = tmp_path / "set_756" / "failed-run"
    monkeypatch.setattr(
        module,
        "run_calibration",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {"status": "failed", "output_dir": output_dir},
        )(),
    )

    exit_code = module.main(
        ["--set", "756", "--pipeline", str(PIPELINE)]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "status=failed" in output
    assert str(output_dir) in output
