import json
from pathlib import Path

import numpy as np
import pytest

from lung_pde.calibration import run_calibration
from lung_pde.constants import CALIBRATED_KEYS, CALIBRATION_DT, FIXED_KEYS
from lung_pde.parameters import read_parameter_xml


pytestmark = [pytest.mark.fenics, pytest.mark.slow]


def test_public_run_calibration_writes_one_complete_fixed_dt_objective(tmp_path):
    initial_xml = Path("data/parameters/initial/parameters_756.xml").resolve()
    initial = read_parameter_xml(initial_xml)
    calibrated = read_parameter_xml(
        "data/parameters/calibrated/parameters_756.xml"
    )
    assert initial["dt"] == pytest.approx(1.0e-4)

    pipeline_path = tmp_path / "smoke_pipeline.yaml"
    parameter_lines = "".join(
        "      - {name: "
        + name
        + ", transform: linear, lower: "
        + repr(calibrated[name])
        + ", upper: "
        + repr(calibrated[name])
        + "}\n"
        for name in CALIBRATED_KEYS
    )
    pipeline_path.write_text(
        "calibration_dt: 0.16666666666666666\n"
        "stages:\n"
        "  - name: resolution_four_smoke\n"
        "    objective: relative_l2_target\n"
        "    optimizer: {maxiter: 30, gtol: 1.0e-12, ftol: 1.0e-12}\n"
        "    parameters:\n"
        + parameter_lines,
        encoding="utf-8",
    )
    command = (
        "scripts/calibrate.py",
        "--set",
        "756",
        "--pipeline",
        str(pipeline_path),
    )
    result = run_calibration(
        756,
        pipeline_path,
        tmp_path / "outputs",
        run_id="fenics-success",
        resolution=4,
        mesh_backend="rectangle",
        max_evaluations=1,
        command_arguments=command,
    )

    run_dir = tmp_path / "outputs" / "set_756" / "fenics-success"
    assert result.status == "complete"
    assert result.output_dir == run_dir
    assert result.stages[0].evaluations == 1
    for name in CALIBRATED_KEYS:
        assert result.final_parameters[name] == pytest.approx(calibrated[name])
    for name in FIXED_KEYS:
        assert result.final_parameters[name] == initial[name]
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["command"] == list(command)
    assert metadata["initial_parameters"] == str(initial_xml)
    assert metadata["calibration_dt"] == CALIBRATION_DT
    assert metadata["observation_count"] == 83
    assert metadata["mesh_backend"] == "rectangle"
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))[
        "status"
    ] == "complete"

    stage_dir = run_dir / "stage_01_resolution_four_smoke"
    assert read_parameter_xml(stage_dir / "parameters_input.xml").values == initial.values
    assert read_parameter_xml(stage_dir / "parameters.xml")["dm"] == initial["dm"]
    status = json.loads((stage_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["calibration_dt"] == CALIBRATION_DT
    assert {change["name"] for change in status["bound_projection"]} == {
        name for name in CALIBRATED_KEYS if initial[name] != calibrated[name]
    }
    with np.load(stage_dir / "trajectory.npz", allow_pickle=False) as trajectory:
        assert trajectory["time"].shape == (83,)
        assert trajectory["time"] == pytest.approx(
            np.arange(1, 84, dtype=float) * CALIBRATION_DT
        )
        assert np.all(np.isfinite(trajectory["bacteria_mucus"]))
        assert np.all(np.isfinite(trajectory["macrophage_mucus"]))
    evaluation_lines = (stage_dir / "evaluations.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(evaluation_lines) == 1
    evaluation = json.loads(evaluation_lines[0])
    assert evaluation["evaluation"] == 1
    assert set(evaluation["parameters"]) == set(CALIBRATED_KEYS)
    assert np.isfinite(evaluation["loss"])
    metrics = json.loads((stage_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["optimizer_success"] is True
    assert metrics["optimizer_status"] is None
    assert metrics["optimizer_message"] == (
        "All independent variables were fixed by bounds."
    )
    assert metrics["max_evaluations"] == 1
    assert metrics["evaluations"] == 1
    assert metrics["checkpoint_source"] == "lowest_finite_evaluated_loss"
    assert metrics["trajectory_kind"] == "checkpoint"


def test_public_reconstructed_rectangle_run_fails_and_rolls_back(tmp_path):
    initial = read_parameter_xml(
        "data/parameters/initial/parameters_756.xml"
    )
    result = run_calibration(
        756,
        "configs/calibration/legacy_reconstructed.yaml",
        tmp_path / "outputs",
        run_id="fenics-raw-failure",
        resolution=4,
        mesh_backend="rectangle",
        max_evaluations=1,
        command_arguments=("reduced-rectangle-regression",),
    )

    run_dir = tmp_path / "outputs" / "set_756" / "fenics-raw-failure"
    first_stage = run_dir / "stage_01_aaanew2_parameter_subset"
    assert result.status == "failed"
    assert result.output_dir == run_dir
    assert len(result.stages) == 1
    assert result.final_parameters.values == initial.values
    assert read_parameter_xml(first_stage / "parameters_input.xml").values == initial.values
    assert read_parameter_xml(first_stage / "parameters.xml").values == initial.values
    assert not (run_dir / "stage_02_aaasimple_parameter_subset").exists()
    assert not (first_stage / "trajectory.npz").exists()

    run_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    stage_status = json.loads(
        (first_stage / "status.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((first_stage / "metrics.json").read_text(encoding="utf-8"))
    assert run_status["status"] == "failed"
    assert stage_status["status"] == "failed"
    assert stage_status["error"]["type"] == "RuntimeError"
    assert "Newton solver did not converge" in stage_status["error"]["message"]
    assert stage_status["evaluations"] == 0
    assert metrics["final_loss"] is None
    assert metrics["best_evaluated_loss"] is None
    assert metrics["optimizer_success"] is None
    assert metrics["optimizer_status"] is None
    assert metrics["optimizer_message"] is None
    assert metrics["optimizer_iterations"] is None
    assert metrics["optimizer_reported_evaluations"] is None
    assert metrics["checkpoint_source"] == "input_checkpoint_preserved"
