import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lung_pde.calibration as calibration
from lung_pde.calibration import (
    CalibrationRunner,
    FenicsCalibrationSimulator,
    run_calibration,
)
from lung_pde.calibration_config import load_pipeline
from lung_pde.constants import CALIBRATION_DT, FIXED_KEYS
from lung_pde.parameters import read_parameter_xml


INITIAL_XML = Path("data/parameters/initial/parameters_756.xml")
LEGACY_STAGE = Path("configs/calibration/legacy/test_lamB2_parameters.yaml")


class RecordingSimulator:
    def __init__(self, resolved_mesh_backend="rectangle"):
        self.calls = []
        self.resolved_mesh_backend = resolved_mesh_backend

    def simulate(self, updates):
        self.calls.append(dict(updates))
        value = updates.get("lambdaB", updates.get("DB1", 1.0))
        return np.full(83, value), np.full(83, 2.0 * value)


def _write_two_stage_pipeline(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "calibration_dt: 0.16666666666666666\n"
        "stages:\n"
        "  - name: first\n"
        "    objective: relative_mse_mean\n"
        "    optimizer: {maxiter: 4, gtol: 1.0e-9, ftol: 1.0e-10}\n"
        "    parameters:\n"
        "      - {name: lambdaB, transform: linear, lower: 1.0, upper: 1.0}\n"
        "  - name: second\n"
        "    objective: relative_mse_mean\n"
        "    optimizer: {maxiter: 3, gtol: 1.0e-8, ftol: 1.0e-11}\n"
        "    parameters:\n"
        "      - {name: DB1, transform: linear, lower: 0.25, upper: 0.25}\n",
        encoding="utf-8",
    )
    return load_pipeline(path)


def _write_single_stage_pipeline(tmp_path, *, lower=0.5, upper=2.0):
    path = tmp_path / "single_stage.yaml"
    path.write_text(
        "calibration_dt: 0.16666666666666666\n"
        "stages:\n"
        "  - name: single\n"
        "    objective: relative_mse_mean\n"
        "    optimizer: {maxiter: 4, gtol: 1.0e-9, ftol: 1.0e-10}\n"
        "    parameters:\n"
        f"      - {{name: lambdaB, transform: linear, lower: {lower}, upper: {upper}}}\n",
        encoding="utf-8",
    )
    return path


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_stage_outputs_chain_active_subsets_and_preserve_fixed_parameters(tmp_path):
    pipeline = _write_two_stage_pipeline(tmp_path)
    initial = read_parameter_xml(INITIAL_XML)
    simulator = RecordingSimulator()
    runner = CalibrationRunner(
        simulator,
        initial,
        np.ones(83),
        np.full(83, 2.0),
        tmp_path / "run",
    )

    result = runner.run(pipeline, max_evaluations=1)

    assert result.status == "complete"
    assert [stage.status for stage in result.stages] == ["complete", "complete"]
    assert [set(call) for call in simulator.calls] == [{"lambdaB"}, {"DB1"}]
    for name in FIXED_KEYS:
        assert result.final_parameters[name] == initial[name]

    first_dir = tmp_path / "run" / "stage_01_first"
    second_dir = tmp_path / "run" / "stage_02_second"
    first_output = read_parameter_xml(first_dir / "parameters.xml")
    second_input = read_parameter_xml(second_dir / "parameters_input.xml")
    assert second_input.values == first_output.values
    assert first_output["lambdaB"] == 1.0
    assert result.final_parameters["DB1"] == 0.25

    status = _read_json(first_dir / "status.json")
    assert status["status"] == "complete"
    assert status["calibration_dt"] == CALIBRATION_DT
    assert status["active_parameters"] == ["lambdaB"]
    assert status["bound_projection"] == [
        {
            "name": "lambdaB",
            "original": initial["lambdaB"],
            "projected": 1.0,
        }
    ]
    evaluations = _read_json_lines(first_dir / "evaluations.jsonl")
    assert evaluations == [
        {
            "evaluation": 1,
            "loss": pytest.approx(0.0),
            "parameters": {"lambdaB": 1.0},
        }
    ]
    with np.load(first_dir / "trajectory.npz", allow_pickle=False) as trajectory:
        assert set(trajectory.files) == {
            "time",
            "bacteria_mucus",
            "macrophage_mucus",
        }
        assert trajectory["time"].shape == (83,)
        assert trajectory["time"][-1] == pytest.approx(83.0 / 6.0)
    assert not list((tmp_path / "run").rglob(".*.tmp"))


def test_stage_uses_lbfgsb_settings_and_explicit_evaluation_budget(tmp_path, monkeypatch):
    captured = {}

    def fake_minimize(objective, x0, *, method, bounds, options):
        captured.update(method=method, bounds=bounds, options=options)
        loss = objective(x0)
        return SimpleNamespace(
            x=np.asarray(x0),
            fun=loss,
            success=True,
            status=0,
            message="converged",
            nit=0,
            nfev=1,
        )

    monkeypatch.setattr(calibration, "minimize", fake_minimize)
    pipeline = load_pipeline(LEGACY_STAGE)
    initial = read_parameter_xml(INITIAL_XML)
    runner = CalibrationRunner(
        RecordingSimulator(), initial, np.ones(83), np.ones(83), tmp_path
    )

    result = runner.run(pipeline, max_evaluations=7)

    optimizer = pipeline.stages[0].optimizer
    assert result.status == "complete"
    assert captured["method"] == "L-BFGS-B"
    assert captured["options"] == {
        "maxiter": optimizer.maxiter,
        "gtol": optimizer.gtol,
        "ftol": optimizer.ftol,
        "maxfun": 7,
    }
    metrics = _read_json(
        tmp_path / "stage_01_test_lamB2_parameter_subset" / "metrics.json"
    )
    assert metrics["max_evaluations"] == 7
    assert metrics["optimizer_success"] is True
    assert metrics["optimizer_settings"] == {
        "method": "L-BFGS-B",
        "maxiter": optimizer.maxiter,
        "gtol": optimizer.gtol,
        "ftol": optimizer.ftol,
    }


@pytest.mark.parametrize(
    ("optimizer_status", "optimizer_message"),
    [
        (1, "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"),
        (2, "ABNORMAL_TERMINATION_IN_LNSRCH"),
    ],
)
def test_any_unsuccessful_scipy_result_preserves_input_and_diagnostics(
    tmp_path, monkeypatch, optimizer_status, optimizer_message
):
    def abnormal_minimize(objective, x0, **kwargs):
        objective(x0)
        return SimpleNamespace(
            x=np.asarray(x0),
            fun=1.0,
            success=False,
            status=optimizer_status,
            message=optimizer_message,
            nit=4,
            nfev=1,
        )

    monkeypatch.setattr(calibration, "minimize", abnormal_minimize)
    initial = read_parameter_xml(INITIAL_XML)
    runner = CalibrationRunner(
        RecordingSimulator(), initial, np.ones(83), np.ones(83), tmp_path
    )

    result = runner.run(load_pipeline(LEGACY_STAGE))

    assert result.status == "failed"
    assert result.final_parameters.values == initial.values
    stage_dir = tmp_path / "stage_01_test_lamB2_parameter_subset"
    assert read_parameter_xml(stage_dir / "parameters.xml").values == initial.values
    status = _read_json(stage_dir / "status.json")
    assert status["error"] == {
        "type": "RuntimeError",
        "message": f"L-BFGS-B failed: {optimizer_message}",
    }
    assert "completed_at_utc" in status
    metrics = _read_json(stage_dir / "metrics.json")
    assert metrics["final_loss"] is None
    assert metrics["best_evaluated_loss"] == pytest.approx(
        _read_json_lines(stage_dir / "evaluations.jsonl")[0]["loss"]
    )
    assert metrics["optimizer_success"] is False
    assert metrics["optimizer_status"] == optimizer_status
    assert metrics["optimizer_message"] == optimizer_message
    assert metrics["optimizer_iterations"] == 4
    assert metrics["optimizer_reported_evaluations"] == 1
    assert metrics["optimizer_settings"]["method"] == "L-BFGS-B"
    assert metrics["trajectory_kind"] == "best_evaluation_diagnostic"
    assert (stage_dir / "trajectory.npz").exists()


def test_successful_stage_checkpoints_lowest_evaluated_loss_not_result_x(
    tmp_path, monkeypatch
):
    def misleading_minimize(objective, x0, **kwargs):
        objective(np.array([0.5]))
        best_loss = objective(np.array([1.0]))
        objective(np.array([2.0]))
        return SimpleNamespace(
            x=np.array([2.0]),
            fun=best_loss,
            success=True,
            status=0,
            message="converged",
            nit=1,
            nfev=3,
        )

    monkeypatch.setattr(calibration, "minimize", misleading_minimize)
    initial = read_parameter_xml(INITIAL_XML)
    runner = CalibrationRunner(
        RecordingSimulator(),
        initial,
        np.ones(83),
        np.full(83, 2.0),
        tmp_path / "run",
    )

    result = runner.run(load_pipeline(_write_single_stage_pipeline(tmp_path)))

    stage_dir = tmp_path / "run" / "stage_01_single"
    assert result.status == "complete"
    assert result.final_parameters["lambdaB"] == 1.0
    assert read_parameter_xml(stage_dir / "parameters.xml")["lambdaB"] == 1.0
    metrics = _read_json(stage_dir / "metrics.json")
    assert metrics["final_loss"] == pytest.approx(0.0)
    assert metrics["best_evaluated_parameters"] == {"lambdaB": 1.0}
    assert metrics["checkpoint_source"] == "lowest_finite_evaluated_loss"
    with np.load(stage_dir / "trajectory.npz", allow_pickle=False) as trajectory:
        assert trajectory["bacteria_mucus"] == pytest.approx(np.ones(83))


def test_max_evaluations_caps_calls_but_fails_without_optimizer_success(tmp_path):
    simulator = RecordingSimulator()
    initial = read_parameter_xml(INITIAL_XML)
    runner = CalibrationRunner(
        simulator,
        initial,
        np.ones(83),
        np.full(83, 2.0),
        tmp_path / "run",
    )

    result = runner.run(
        load_pipeline(_write_single_stage_pipeline(tmp_path)), max_evaluations=1
    )

    stage_dir = tmp_path / "run" / "stage_01_single"
    assert result.status == "failed"
    assert len(simulator.calls) == 1
    assert result.final_parameters.values == initial.values
    assert read_parameter_xml(stage_dir / "parameters.xml").values == initial.values
    metrics = _read_json(stage_dir / "metrics.json")
    assert metrics["termination"] == "max_evaluations"
    assert metrics["max_evaluations"] == 1
    assert metrics["evaluations"] == 1
    assert metrics["optimizer_success"] is None
    assert metrics["optimizer_status"] is None
    assert metrics["optimizer_message"] is None
    assert metrics["optimizer_iterations"] is None
    assert metrics["optimizer_reported_evaluations"] is None
    assert metrics["best_evaluated_loss"] == pytest.approx(0.5)
    assert metrics["trajectory_kind"] == "best_evaluation_diagnostic"
    assert (stage_dir / "trajectory.npz").exists()


def test_failed_simulation_preserves_input_checkpoint_and_stops_pipeline(tmp_path):
    class FailingSimulator:
        def simulate(self, updates):
            raise RuntimeError("Newton solve failed")

    pipeline = _write_two_stage_pipeline(tmp_path)
    initial = read_parameter_xml(INITIAL_XML)
    run_dir = tmp_path / "run"
    runner = CalibrationRunner(
        FailingSimulator(), initial, np.ones(83), np.ones(83), run_dir
    )

    result = runner.run(pipeline, max_evaluations=1)

    assert result.status == "failed"
    assert result.final_parameters.values == initial.values
    failed_dir = run_dir / "stage_01_first"
    assert read_parameter_xml(failed_dir / "parameters_input.xml").values == initial.values
    assert read_parameter_xml(failed_dir / "parameters.xml").values == initial.values
    assert not (run_dir / "stage_02_second").exists()
    assert not (failed_dir / "trajectory.npz").exists()
    status = _read_json(failed_dir / "status.json")
    assert status["status"] == "failed"
    assert status["error"] == {
        "type": "RuntimeError",
        "message": "Newton solve failed",
    }
    metrics = _read_json(failed_dir / "metrics.json")
    assert metrics["optimizer_success"] is None
    assert metrics["optimizer_status"] is None
    assert metrics["optimizer_message"] is None
    assert metrics["optimizer_iterations"] is None
    assert metrics["optimizer_reported_evaluations"] is None
    assert metrics["optimizer_settings"] == {
        "method": "L-BFGS-B",
        "maxiter": 4,
        "gtol": 1.0e-9,
        "ftol": 1.0e-10,
    }


def test_invalid_trajectory_shape_fails_the_stage(tmp_path):
    class ShortSimulator:
        def simulate(self, updates):
            return np.ones(82), np.ones(83)

    initial = read_parameter_xml(INITIAL_XML)
    runner = CalibrationRunner(
        ShortSimulator(), initial, np.ones(83), np.ones(83), tmp_path
    )

    result = runner.run(load_pipeline(LEGACY_STAGE), max_evaluations=1)

    assert result.status == "failed"
    status = _read_json(tmp_path / "stage_01_test_lamB2_parameter_subset" / "status.json")
    assert status["error"]["type"] == "ValueError"
    assert "shape (83,)" in status["error"]["message"]


class _FakeDt:
    def __init__(self):
        self.assigned = []

    def assign(self, value):
        self.assigned.append(float(value))


class _FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(dt=CALIBRATION_DT)
        self.dt = _FakeDt()
        self.parameters = {name: object() for name in ("lambdaB", "dm")}
        self.reset_count = 0
        self.updates = []
        self.step_times = []

    def reset(self):
        self.reset_count += 1

    def assign_parameters(self, updates):
        self.updates.append(dict(updates))

    def step(self, time):
        self.step_times.append(float(time))

    def concentrations(self):
        value = len(self.step_times)
        return SimpleNamespace(bacteria_mucus=value, macrophage_mucus=2 * value)


def test_fenics_adapter_resets_and_advances_exactly_83_fixed_steps():
    model = _FakeModel()
    simulator = FenicsCalibrationSimulator(model=model)

    bacteria, macrophages = simulator.simulate({"lambdaB": 0.25})

    assert model.reset_count == 1
    assert model.updates == [{"lambdaB": 0.25}]
    assert model.dt.assigned == [CALIBRATION_DT]
    assert len(model.step_times) == 83
    assert model.step_times == pytest.approx(
        [CALIBRATION_DT * index for index in range(1, 84)]
    )
    assert bacteria.shape == macrophages.shape == (83,)
    assert bacteria[-1] == 83
    assert macrophages[-1] == 166


def test_fenics_adapter_rejects_runtime_dt_override():
    with pytest.raises(ValueError, match="cannot be overridden"):
        FenicsCalibrationSimulator(model=_FakeModel(), dt=0.01)


def test_run_calibration_creates_unique_scoped_output_and_command_metadata(
    tmp_path, monkeypatch
):
    simulator = RecordingSimulator()
    monkeypatch.setattr(
        calibration,
        "FenicsCalibrationSimulator",
        lambda **kwargs: simulator,
    )
    command = ("scripts/calibrate.py", "--set", "756", "--max-evaluations", "1")
    pipeline_path = _write_single_stage_pipeline(tmp_path, lower=1.0, upper=1.0)

    result = run_calibration(
        756,
        pipeline_path,
        tmp_path / "outputs",
        run_id="20260804T120000000000Z-test",
        resolution=4,
        mesh_backend="rectangle",
        max_evaluations=1,
        command_arguments=command,
    )

    expected = tmp_path / "outputs" / "set_756" / "20260804T120000000000Z-test"
    assert result.output_dir == expected
    assert result.status == "complete"
    metadata = _read_json(expected / "metadata.json")
    assert metadata["command"] == list(command)
    assert metadata["set_id"] == 756
    assert metadata["initial_parameters"].endswith(
        "data/parameters/initial/parameters_756.xml"
    )
    assert metadata["resolution"] == 4
    assert metadata["mesh_backend"] == "rectangle"
    assert metadata["requested_mesh_backend"] == "rectangle"
    assert _read_json(expected / "status.json")["status"] == "complete"
    with pytest.raises(FileExistsError):
        run_calibration(
            756,
            pipeline_path,
            tmp_path / "outputs",
            run_id="20260804T120000000000Z-test",
            max_evaluations=1,
        )


@pytest.mark.parametrize(
    ("requested", "resolved"),
    [
        ("auto", "rectangle"),
        ("rectangle", "rectangle"),
        ("legacy_mshr", "legacy_mshr"),
    ],
)
def test_successful_calibration_records_requested_and_resolved_mesh_backends(
    tmp_path, monkeypatch, requested, resolved
):
    simulator = RecordingSimulator(resolved_mesh_backend=resolved)
    monkeypatch.setattr(
        calibration,
        "FenicsCalibrationSimulator",
        lambda **kwargs: simulator,
    )
    pipeline_path = _write_single_stage_pipeline(tmp_path, lower=1.0, upper=1.0)

    result = run_calibration(
        756,
        pipeline_path,
        tmp_path / "outputs",
        run_id=f"mesh-{requested}",
        resolution=4,
        mesh_backend=requested,
        max_evaluations=1,
    )

    assert result.status == "complete"
    metadata = _read_json(result.output_dir / "metadata.json")
    assert metadata["requested_mesh_backend"] == requested
    assert metadata["mesh_backend"] == resolved


def test_model_construction_failure_keeps_resolved_mesh_backend_null(
    tmp_path, monkeypatch
):
    def fail_before_resolution(**kwargs):
        raise RuntimeError("mesh construction failed")

    monkeypatch.setattr(
        calibration,
        "FenicsCalibrationSimulator",
        fail_before_resolution,
    )
    pipeline_path = _write_single_stage_pipeline(tmp_path, lower=1.0, upper=1.0)

    result = run_calibration(
        756,
        pipeline_path,
        tmp_path / "outputs",
        run_id="mesh-failure",
        resolution=4,
        mesh_backend="auto",
        max_evaluations=1,
    )

    assert result.status == "failed"
    metadata = _read_json(result.output_dir / "metadata.json")
    assert metadata["requested_mesh_backend"] == "auto"
    assert metadata["mesh_backend"] is None
    status = _read_json(result.output_dir / "status.json")
    assert status["status"] == "failed"
    assert status["error"] == {
        "type": "RuntimeError",
        "message": "mesh construction failed",
    }


def test_explicit_backend_mismatch_fails_but_records_actual_resolution(
    tmp_path, monkeypatch
):
    simulator = RecordingSimulator(resolved_mesh_backend="rectangle")
    monkeypatch.setattr(
        calibration,
        "FenicsCalibrationSimulator",
        lambda **kwargs: simulator,
    )
    pipeline_path = _write_single_stage_pipeline(tmp_path, lower=1.0, upper=1.0)

    result = run_calibration(
        756,
        pipeline_path,
        tmp_path / "outputs",
        run_id="mesh-mismatch",
        resolution=4,
        mesh_backend="legacy_mshr",
        max_evaluations=1,
    )

    assert result.status == "failed"
    metadata = _read_json(result.output_dir / "metadata.json")
    assert metadata["requested_mesh_backend"] == "legacy_mshr"
    assert metadata["mesh_backend"] == "rectangle"
    assert _read_json(result.output_dir / "status.json")["status"] == "failed"


@pytest.mark.parametrize("resolution", [0, -1, True, 1.5, "4"])
def test_run_calibration_rejects_invalid_resolution_before_creating_output(
    tmp_path, monkeypatch, resolution
):
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(
        calibration,
        "FenicsCalibrationSimulator",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid resolution reached FEniCS construction")
        ),
    )

    with pytest.raises(ValueError, match="resolution must be a positive integer"):
        run_calibration(
            756,
            LEGACY_STAGE,
            output_root,
            run_id="invalid-resolution",
            resolution=resolution,
            max_evaluations=1,
        )

    assert not output_root.exists()


@pytest.mark.parametrize("max_evaluations", [0, -1, True, 1.5, "1"])
def test_run_calibration_rejects_invalid_evaluation_budget_before_output(
    tmp_path, max_evaluations
):
    output_root = tmp_path / "outputs"

    with pytest.raises(ValueError, match="max_evaluations must be a positive integer"):
        run_calibration(
            756,
            LEGACY_STAGE,
            output_root,
            run_id="invalid-budget",
            resolution=4,
            max_evaluations=max_evaluations,
        )

    assert not output_root.exists()
