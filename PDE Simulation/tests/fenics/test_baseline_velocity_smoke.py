from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


dolfin = pytest.importorskip("dolfin")

from lung_pde.model import build_model
from lung_pde.results import load_simulation_result
from lung_pde.simulation import run_simulation
from lung_pde.velocity_fields import MucusVelocity
from scripts import simulate, simulate_velocity


ROOT = Path(__file__).resolve().parents[2]
BASELINE_SCRIPT = Path("scripts/simulate.py")


@pytest.mark.fenics
def test_mucus_velocity_matches_test_v4_profile():
    velocity = MucusVelocity(t=0.0, degree=3)
    point = np.array([0.05, 0.2])
    values = np.empty(2)

    for time, expected_x in ((0.0, 0.0), (0.5, 0.005), (1.0, 0.01)):
        velocity.t = time
        velocity.eval(values, point)
        np.testing.assert_allclose(values, [expected_x, 0.0], atol=1.0e-15)


@pytest.mark.fenics
@pytest.mark.parametrize(
    ("mode", "script_name", "main"),
    [
        ("baseline", "scripts/simulate.py", simulate.main),
        ("velocity", "scripts/simulate_velocity.py", simulate_velocity.main),
    ],
)
def test_reduced_cli_writes_identical_finite_result_schema(
    tmp_path, mode, script_name, main
):
    output_dir = tmp_path / mode
    arguments = [
        "--set",
        "756",
        "--resolution",
        "4",
        "--mesh-backend",
        "rectangle",
        "--final-time",
        "0.02",
        "--snapshot-times",
        "0",
        "0.01",
        "0.02",
        "--output-dir",
        str(output_dir),
    ]

    assert main(arguments) == 0

    assert {path.name for path in output_dir.iterdir()} == {
        "concentrations.npz",
        "spatial.npz",
        "metadata.json",
    }
    result = load_simulation_result(output_dir)
    assert result.time == pytest.approx([0.0, 0.01, 0.02])
    for trajectory in (
        result.bacteria_mucus,
        result.bacteria_tissue,
        result.macrophage_mucus,
        result.macrophage_tissue,
        result.mucus_density,
    ):
        assert trajectory.shape == (3,)
        assert np.all(np.isfinite(trajectory))

    assert result.snapshots is not None
    assert result.snapshots.time == pytest.approx([0.0, 0.01, 0.02])
    for state in (
        result.snapshots.bacteria_mucus,
        result.snapshots.bacteria_tissue,
        result.snapshots.macrophage_mucus,
        result.snapshots.macrophage_tissue,
    ):
        assert state.shape[0] == 3
        assert np.all(np.isfinite(state))

    metadata = result.metadata
    assert metadata["mode"] == mode
    assert metadata["dt"] == pytest.approx(0.01)
    assert metadata["final_time"] == pytest.approx(0.02)
    assert metadata["requested_snapshot_times"] == pytest.approx(
        [0.0, 0.01, 0.02]
    )
    assert metadata["parameter_source_kind"] == "calibrated"
    assert Path(metadata["parameter_source"]).parent.name == "calibrated"
    assert metadata["mesh_backend"] == "rectangle"
    assert metadata["command_arguments"] == [script_name, *arguments]


@pytest.mark.fenics
def test_baseline_cli_dt_override_controls_saved_grid_and_model_metadata(
    tmp_path,
):
    output_dir = tmp_path / "baseline-dt-override"
    arguments = [
        "--set",
        "756",
        "--resolution",
        "4",
        "--mesh-backend",
        "rectangle",
        "--dt",
        "0.02",
        "--final-time",
        "0.04",
        "--snapshot-times",
        "0",
        "0.02",
        "0.04",
        "--output-dir",
        str(output_dir),
    ]

    completed = subprocess.run(
        [sys.executable, str(BASELINE_SCRIPT), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=complete" in completed.stdout
    assert f"output={output_dir}" in completed.stdout

    result = load_simulation_result(output_dir)
    assert result.time == pytest.approx([0.0, 0.02, 0.04])
    assert result.snapshots is not None
    assert result.snapshots.time == pytest.approx([0.0, 0.02, 0.04])

    metadata = result.metadata
    assert metadata["dt"] == pytest.approx(0.02)
    assert metadata["final_time"] == pytest.approx(0.04)
    assert metadata["actual_final_time"] == pytest.approx(0.04)
    assert metadata["requested_snapshot_times"] == pytest.approx(
        [0.0, 0.02, 0.04]
    )
    assert metadata["parameter_source_kind"] == "calibrated"
    parameter_source = Path(metadata["parameter_source"])
    assert metadata["parameter_source_sha256"] == hashlib.sha256(
        parameter_source.read_bytes()
    ).hexdigest()
    assert metadata["mesh_backend"] == "rectangle"
    assert metadata["command_arguments"] == [
        str(BASELINE_SCRIPT),
        *arguments,
    ]


@pytest.mark.fenics
def test_dt_override_controls_model_provenance_validation_and_metadata():
    model = build_model(
        set_id=756,
        mode="baseline",
        resolution=4,
        mesh_backend="rectangle",
        dt=0.02,
    )

    assert model.config.dt == pytest.approx(0.02)
    assert float(model.dt) == pytest.approx(0.02)
    assert model.provenance.dt == pytest.approx(0.02)

    config = replace(
        model.config,
        final_time=0.04,
        snapshot_times=(0.0, 0.02, 0.04),
    )
    result = run_simulation(model, config)

    assert result.time == pytest.approx([0.0, 0.02, 0.04])
    assert result.metadata["dt"] == pytest.approx(0.02)
    assert result.metadata["final_time"] == pytest.approx(0.04)
    assert result.metadata["requested_snapshot_times"] == pytest.approx(
        [0.0, 0.02, 0.04]
    )

    with pytest.raises(ValueError, match="dt"):
        run_simulation(model, replace(config, dt=0.01))


@pytest.mark.fenics
@pytest.mark.parametrize("dt", [0.0, -0.01, np.nan, np.inf])
def test_model_rejects_invalid_dt_before_building(dt):
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        build_model(
            set_id=756,
            mode="baseline",
            resolution=4,
            mesh_backend="rectangle",
            dt=dt,
        )
