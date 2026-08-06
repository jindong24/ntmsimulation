from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest


dolfin = pytest.importorskip("dolfin")

from lung_pde.data import load_abm
from lung_pde.model import ModelConfig, build_model
from lung_pde.simulation import run_simulation
from lung_pde.velocity_fields import MucusVelocity, TreatmentVelocity, ZeroVelocity


ROOT = Path(__file__).resolve().parents[2]
MODES = ("calibration", "baseline", "velocity", "treatment")


@pytest.mark.fenics
def test_velocity_expressions_match_authoritative_sources():
    point = np.array([0.05, 0.2])
    values = np.empty(2)

    zero = ZeroVelocity(t=3.0, degree=3)
    zero.eval(values, point)
    np.testing.assert_allclose(values, [0.0, 0.0])
    assert zero.value_shape() == (2,)

    velocity = MucusVelocity(t=0.0, degree=3)
    for time, expected_x in ((0.0, 0.0), (0.5, 0.005), (1.0, 0.01)):
        velocity.t = time
        velocity.eval(values, point)
        np.testing.assert_allclose(values, [expected_x, 0.0], atol=1.0e-15)
    assert velocity.value_shape() == (2,)

    treatment = TreatmentVelocity(t=0.0, degree=3)
    for time, expected_x in ((0.0, 0.0), (0.25, 0.0072), (0.5, 0.0144)):
        treatment.t = time
        treatment.eval(values, point)
        np.testing.assert_allclose(values, [expected_x, 0.0], atol=1.0e-15)
    assert treatment.value_shape() == (2,)


@pytest.mark.fenics
@pytest.mark.parametrize(
    ("mode", "dt", "final_time", "parameter_kind", "snapshot_times"),
    [
        ("calibration", 1.0 / 6.0, None, "initial", ()),
        ("baseline", 0.01, None, "calibrated", (0.0, 1.0, 2.0)),
        ("velocity", 0.01, 9.0, "calibrated", (0.0, 1.0, 2.0)),
        ("treatment", 0.1, 14.0, "calibrated", ()),
    ],
)
def test_mode_defaults_and_parameter_sources(
    mode, dt, final_time, parameter_kind, snapshot_times
):
    abm = load_abm(ROOT / "data" / "abm" / "data_756.npz")

    config = ModelConfig.for_mode(756, mode)

    assert config.dt == pytest.approx(dt)
    assert config.final_time == pytest.approx(
        abm.times[-1] if final_time is None else final_time
    )
    assert config.parameter_source.parent.name == parameter_kind
    assert config.snapshot_times == snapshot_times
    assert config.solver_settings.maximum_iterations == (
        100 if mode == "calibration" else None
    )


@pytest.mark.fenics
def test_treatment_passes_unscaled_danti_and_records_effective_value():
    input_danti = 2.5
    model = build_model(
        set_id=756,
        mode="treatment",
        resolution=4,
        mesh_backend="rectangle",
        danti=input_danti,
    )

    assert float(model.danti) == pytest.approx(input_danti)
    assert model.effective_danti == pytest.approx(
        input_danti
        * model.abm.initial_macrophages
        * float(model.parameters["dB"])
    )


@pytest.mark.fenics
def test_treatment_metadata_uses_current_db_after_parameter_assignment():
    input_danti = 2.5
    model = build_model(
        set_id=756,
        mode="treatment",
        resolution=4,
        mesh_backend="rectangle",
        danti=input_danti,
    )
    updated_db = float(model.parameters["dB"]) * 1.5
    model.assign_parameters({"dB": updated_db})
    config = replace(model.config, final_time=0.0, snapshot_times=())

    result = run_simulation(model, config)

    assert result.metadata["effective_danti"] == pytest.approx(
        input_danti * model.abm.initial_macrophages * updated_db
    )


@pytest.mark.fenics
def test_metadata_keeps_build_time_parameter_hash_after_xml_changes(tmp_path):
    source = ModelConfig.for_mode(756, "baseline").parameter_source
    parameter_copy = tmp_path / "parameters.xml"
    parameter_copy.write_bytes(source.read_bytes())
    expected_hash = hashlib.sha256(parameter_copy.read_bytes()).hexdigest()
    model = build_model(
        set_id=756,
        mode="baseline",
        resolution=4,
        mesh_backend="rectangle",
        parameter_source=parameter_copy,
    )
    assert model.provenance.parameter_source == parameter_copy.resolve()
    assert model.provenance.parameter_source_kind == "custom"
    assert model.provenance.parameter_source_sha256 == expected_hash
    assert model.provenance.resolution == 4
    assert model.provenance.mesh_backend == "rectangle"
    assert model.provenance.solver_settings == model.config.solver_settings
    assert dict(model.provenance.initial_parameter_values) == model.parameter_values
    parameter_copy.write_text("<changed-after-build />\n", encoding="utf-8")
    config = replace(model.config, final_time=0.0, snapshot_times=())

    result = run_simulation(model, config)

    assert result.metadata["parameter_source"] == str(parameter_copy.resolve())
    assert result.metadata["parameter_source_kind"] == "custom"
    assert result.metadata["parameter_source_sha256"] == expected_hash


@pytest.mark.fenics
def test_run_rejects_structurally_mismatched_config():
    model = build_model(
        set_id=756,
        mode="baseline",
        resolution=4,
        mesh_backend="rectangle",
    )
    mismatch_values = {
        "parameter_source": ModelConfig.for_mode(756, "calibration").parameter_source,
        "dt": model.config.dt * 2.0,
        "resolution": 8,
        "mesh_backend": "auto",
        "solver_settings": replace(model.config.solver_settings, report=True),
        "eta": 2.0,
        "danti": 2.0,
    }

    for name, value in mismatch_values.items():
        mismatched = replace(
            model.config,
            **{name: value},
            final_time=0.0,
            snapshot_times=(),
        )
        with pytest.raises(ValueError, match=name):
            run_simulation(model, mismatched)


@pytest.mark.fenics
def test_default_auto_backend_config_is_reusable_and_rejects_incompatible_backend():
    model = build_model(
        set_id=756,
        mode="baseline",
        resolution=4,
    )
    model.config = replace(model.config, final_time=0.0, snapshot_times=())

    result = run_simulation(model, model.config)

    assert model.config.mesh_backend == "rectangle"
    assert model.provenance.mesh_backend == "rectangle"
    assert result.metadata["mesh_backend"] == "rectangle"

    incompatible = replace(model.config, mesh_backend="legacy_mshr")
    with pytest.raises(ValueError, match="mesh_backend"):
        run_simulation(model, incompatible)


@pytest.mark.fenics
@pytest.mark.parametrize("mode", MODES)
def test_resolution_four_one_step_run_is_finite_and_complete(mode):
    model = build_model(
        set_id=756,
        mode=mode,
        resolution=4,
        mesh_backend="rectangle",
    )
    config = replace(
        model.config,
        final_time=model.config.dt,
        snapshot_times=(0.0, model.config.dt),
        command_arguments=("pytest", mode),
    )

    result = run_simulation(model, config)

    assert result.time == pytest.approx([0.0, config.dt])
    for values in (
        result.bacteria_mucus,
        result.bacteria_tissue,
        result.macrophage_mucus,
        result.macrophage_tissue,
        result.mucus_density,
    ):
        assert values.shape == (2,)
        assert np.all(np.isfinite(values))

    assert result.snapshots is not None
    assert result.snapshots.time == pytest.approx([0.0, config.dt])
    assert result.snapshots.bacteria_mucus.shape[0] == 2
    assert result.snapshots.bacteria_mucus.shape[1] == len(
        result.snapshots.mucus_xy
    )
    assert np.all(np.isfinite(result.snapshots.macrophage_tissue))

    assert result.metadata.keys() >= {
        "set_id",
        "mode",
        "parameter_source",
        "parameter_source_kind",
        "parameter_source_sha256",
        "dt",
        "final_time",
        "requested_snapshot_times",
        "resolution",
        "mesh_backend",
        "solver_settings",
        "command_arguments",
        "package_version",
        "created_at_utc",
    }
    assert result.metadata["mesh_backend"] == "rectangle"
    assert result.metadata["command_arguments"] == ["pytest", mode]
    expected_source = "initial" if mode == "calibration" else "calibrated"
    assert Path(result.metadata["parameter_source"]).parent.name == expected_source
    assert result.metadata["parameter_source_kind"] == expected_source
    assert len(result.metadata["parameter_source_sha256"]) == 64
    if mode == "treatment":
        assert result.metadata["eta"] == pytest.approx(1.0)
        assert result.metadata["input_danti"] == pytest.approx(1.0)
        assert result.metadata["effective_danti"] == pytest.approx(
            model.abm.initial_macrophages * float(model.parameters["dB"])
        )


@pytest.mark.fenics
def test_model_can_reset_and_assign_parameter_constants():
    model = build_model(
        set_id=756,
        mode="calibration",
        resolution=4,
        mesh_backend="rectangle",
    )
    initial_state = model.w.vector().get_local().copy()
    updated_lambda = float(model.parameters["lambdaB"]) * 1.5

    model.w.vector()[:] = 0.0
    model.assign_parameters({"lambdaB": updated_lambda})
    model.reset()

    np.testing.assert_allclose(model.w.vector().get_local(), initial_state)
    assert float(model.parameters["lambdaB"]) == pytest.approx(updated_lambda)
    with pytest.raises(KeyError, match="unknown model parameter"):
        model.assign_parameters({"not_a_parameter": 1.0})
