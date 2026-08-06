import numpy as np
import pytest

from lung_pde.calibration_config import (
    ParameterSpec,
    load_pipeline,
    pack,
    project_initial_to_bounds,
    resolve_bounds,
    unpack,
)
from lung_pde.constants import FIXED_KEYS
from lung_pde.parameters import read_parameter_xml


def _write_pipeline(tmp_path, *, calibration_dt="0.16666666666666666", objective="relative_mse_sum", parameters):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "calibration_dt: " + calibration_dt + "\nstages:\n"
        "- name: stage\n"
        "  objective: " + objective + "\n"
        "  optimizer: {maxiter: 1, gtol: 0, ftol: 0}\n"
        "  parameters:\n" + parameters
    )
    return path


def test_reconstructed_pipeline_collectively_covers_exactly_15_variables():
    pipeline = load_pipeline("configs/calibration/legacy_reconstructed.yaml")
    active = {item.name for stage in pipeline.stages for item in stage.parameters}
    assert active == {
        "DB1", "DB2", "lambdaB", "KB", "KBi", "dMb", "dB", "tauB",
        "DM", "delta", "dM", "tauM", "M0", "alpha0", "alpha1",
    }
    assert pipeline.calibration_dt == pytest.approx(1.0 / 6.0)


def test_log_transform_round_trip_uses_current_xml_values():
    pipeline = load_pipeline("configs/calibration/legacy/aaanew2_parameters.yaml")
    values = read_parameter_xml("data/parameters/initial/parameters_756.xml").values
    vector = pack(values, pipeline.stages[0].parameters)
    restored = unpack(vector, pipeline.stages[0].parameters)
    for name, value in restored.items():
        assert value == pytest.approx(values[name])


@pytest.mark.parametrize("name", FIXED_KEYS)
def test_every_fixed_parameter_is_rejected(tmp_path, name):
    pipeline = _write_pipeline(
        tmp_path,
        parameters=f"  - {{name: {name}, transform: linear, lower: 0, upper: 1}}\n",
    )
    with pytest.raises(ValueError, match=f"{name} is not calibratable"):
        load_pipeline(pipeline)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ("  - {name: unknown, transform: linear, lower: 0, upper: 1}\n", "unknown is not calibratable"),
        (
            "  - {name: DB1, transform: linear, lower: 0, upper: 1}\n"
            "  - {name: DB1, transform: linear, lower: 0, upper: 1}\n",
            "duplicate parameter in stage: DB1",
        ),
        ("  - {name: DB1, transform: invalid, lower: 0, upper: 1}\n", "DB1 transform must be linear or log"),
        ("  - {name: DB1, transform: log, lower: 0, upper: 1}\n", "DB1 lower log bound must be positive"),
        ("  - {name: DB1, transform: log, lower: 0.1, upper: -1}\n", "DB1 upper log bound must be positive"),
    ],
)
def test_invalid_stage_parameters_are_rejected(tmp_path, parameters, message):
    pipeline = _write_pipeline(tmp_path, parameters=parameters)
    with pytest.raises(ValueError, match=message):
        load_pipeline(pipeline)


def test_unknown_objective_is_rejected(tmp_path):
    pipeline = _write_pipeline(
        tmp_path,
        objective="not_an_objective",
        parameters="  - {name: DB1, transform: linear, lower: 0, upper: 1}\n",
    )
    with pytest.raises(ValueError, match="unknown objective: not_an_objective"):
        load_pipeline(pipeline)


def test_invalid_calibration_dt_is_rejected(tmp_path):
    pipeline = _write_pipeline(
        tmp_path,
        calibration_dt="0.25",
        parameters="  - {name: DB1, transform: linear, lower: 0, upper: 1}\n",
    )
    with pytest.raises(ValueError, match="calibration_dt must equal"):
        load_pipeline(pipeline)


def test_unknown_parameter_bound_reference_is_rejected_when_resolved(tmp_path):
    pipeline = load_pipeline(_write_pipeline(
        tmp_path,
        parameters="  - {name: DB1, transform: linear, lower: parameter:missing, upper: 1}\n",
    ))
    with pytest.raises(ValueError, match="unknown parameter bound reference: missing"):
        resolve_bounds(pipeline.stages[0].parameters, {})


@pytest.mark.parametrize(
    ("path", "stage_index"),
    [
        ("configs/calibration/legacy/aaanew2_parameters.yaml", 0),
        ("configs/calibration/legacy/aaasimple_parameters.yaml", 1),
        ("configs/calibration/legacy/test_lamB2_parameters.yaml", 2),
    ],
)
def test_single_stage_legacy_configs_exactly_match_reconstructed_stages(path, stage_index):
    reconstructed = load_pipeline("configs/calibration/legacy_reconstructed.yaml")
    single_stage = load_pipeline(path)
    assert single_stage.stages == (reconstructed.stages[stage_index],)


def test_out_of_bound_xml_initial_value_is_projected_and_reported():
    pipeline = load_pipeline("configs/calibration/legacy/aaanew2_parameters.yaml")
    values = read_parameter_xml("data/parameters/initial/parameters_756.xml").values
    specs = pipeline.stages[0].parameters
    vector = pack(values, specs)
    bounds = resolve_bounds(specs, values)
    projected, changes = project_initial_to_bounds(vector, bounds, specs)
    assert projected[5] == 100.0
    alpha1_change = {change["name"]: change for change in changes}["alpha1"]
    assert alpha1_change["original"] == values["alpha1"]
    assert alpha1_change["projected"] == 100.0


def test_log_coordinate_projection_reports_physical_values():
    specs = (ParameterSpec("DB1", "log", 1.0e-6, 1.0),)
    vector = np.array([np.log(1.0e-8)])
    bounds = ((np.log(1.0e-6), np.log(1.0)),)

    projected, changes = project_initial_to_bounds(vector, bounds, specs)

    assert projected[0] == pytest.approx(np.log(1.0e-6))
    assert changes == [{
        "name": "DB1",
        "original": pytest.approx(1.0e-8),
        "projected": pytest.approx(1.0e-6),
    }]
