from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest


pytest.importorskip("dolfin")

from lung_pde.model import build_model
from lung_pde.treatment import (
    CASE_SCHEMA_VERSION,
    TreatmentCase,
    case_output_path,
    compute_runtime_revision,
    iter_treatment_cases,
    load_case_result,
    load_treatment_grid,
    write_case_result,
)
from lung_pde.velocity_fields import TreatmentVelocity
from scripts import sweep_danti_eta


@pytest.mark.fenics
def test_treatment_velocity_matches_test_anti_profile_exactly():
    velocity = TreatmentVelocity(t=0.0, degree=3)
    point = np.array([0.05, 0.2])
    values = np.empty(2)

    for time, expected_x in ((0.0, 0.0), (0.25, 0.0072), (0.5, 0.0144)):
        velocity.t = time
        velocity.eval(values, point)
        np.testing.assert_allclose(values, [expected_x, 0.0], atol=1.0e-15)


@pytest.mark.fenics
def test_treatment_model_defaults_to_calibrated_parameters_and_current_db():
    model = build_model(
        set_id=756,
        mode="treatment",
        resolution=4,
        mesh_backend="rectangle",
        eta=1.0,
        danti=1.0,
    )

    assert model.config.dt == pytest.approx(0.1)
    assert model.config.final_time == pytest.approx(14.0)
    assert model.provenance.parameter_source_kind == "calibrated"
    assert model.provenance.parameter_source.parent.name == "calibrated"
    expected = (
        model.provenance.initial_macrophages * model.parameter_values["dB"]
    )
    assert model.effective_danti == pytest.approx(expected)

    model.assign_parameters({"dB": 2.0 * model.parameter_values["dB"]})
    assert model.effective_danti == pytest.approx(2.0 * expected)


@pytest.mark.fenics
def test_reduced_treatment_cli_writes_one_finite_self_validating_case(tmp_path):
    output_dir = tmp_path / "treatment"
    arguments = [
        "--set",
        "756",
        "--config",
        "configs/treatment_grid.yaml",
        "--case",
        "1.0",
        "1.0",
        "--resolution",
        "4",
        "--mesh-backend",
        "rectangle",
        "--final-time",
        "0.1",
        "--output-dir",
        str(output_dir),
    ]

    assert sweep_danti_eta.main(arguments) == 0

    case = TreatmentCase(eta=1.0, danti_input=1.0)
    destination = case_output_path(output_dir, case)
    assert {
        path.name
        for path in (output_dir / "cases").iterdir()
        if path.suffix == ".json"
    } == {
        destination.name
    }
    assert not (output_dir / "treatment.csv").exists()
    record = load_case_result(destination, expected_case=case)
    assert record["set_id"] == 756
    assert record["status"] == "success"
    assert record["dt"] == pytest.approx(0.1)
    assert record["final_time"] == pytest.approx(0.1)
    assert record["danti_input"] == pytest.approx(1.0)
    assert record["danti_effective"] == pytest.approx(
        record["M0"] * record["dB"]
    )
    assert record["parameter_source_kind"] == "calibrated"
    assert Path(record["parameter_source"]).parent.name == "calibrated"
    assert record["mesh_backend"] == "rectangle"
    assert record["resolved_mesh_backend"] == "rectangle"
    for field in ("B_MUCUS", "B_TISSUE", "M_MUCUS", "M_TISSUE"):
        assert np.isfinite(record[field])

    mismatched = dict(record, dt=0.2)
    write_case_result(destination, mismatched)
    with pytest.raises(SystemExit) as error:
        sweep_danti_eta.main(arguments)
    assert error.value.code == 2

    assert sweep_danti_eta.main([*arguments, "--overwrite"]) == 0
    restored = load_case_result(destination, expected_case=case)
    assert restored["dt"] == pytest.approx(0.1)


@pytest.mark.fenics
def test_parallel_cli_spawns_disjoint_workers_before_any_parent_dolfin_import(
    tmp_path,
):
    config = tmp_path / "two-cases.yaml"
    config.write_text(
        "eta_values: [1.0]\n"
        "positive_danti_multipliers: [0.002]\n"
        "include_zero_reference: true\n"
        "zero_reference_eta: 1.0\n",
        encoding="utf-8",
    )
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "import builtins\n"
        "import os\n"
        "original_import = builtins.__import__\n"
        "def guarded_import(name, *args, **kwargs):\n"
        "    if (name == 'dolfin' or name.startswith('dolfin.')) and "
        "os.environ.get('DOLFIN_GUARD_PID') == str(os.getpid()):\n"
        "        raise RuntimeError('parent imported DOLFIN before worker spawn')\n"
        "    return original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded_import\n",
        encoding="utf-8",
    )
    runner = tmp_path / "parallel_runner.py"
    runner.write_text(
        "import os\n"
        "import sys\n"
        "from scripts.sweep_danti_eta import main\n"
        "if __name__ == '__main__':\n"
        "    os.environ['DOLFIN_GUARD_PID'] = str(os.getpid())\n"
        "    raise SystemExit(main(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "parallel-output"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path("src").resolve()), str(Path.cwd()))
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--set",
            "756",
            "--config",
            str(config),
            "--workers",
            "2",
            "--resolution",
            "4",
            "--mesh-backend",
            "rectangle",
            "--final-time",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "parent imported DOLFIN before worker spawn" not in completed.stderr
    assert "completed=2" in completed.stdout
    expected_names = {
        f"{TreatmentCase(1.0, 0.0).key}.json",
        f"{TreatmentCase(1.0, 0.002).key}.json",
    }
    actual_names = {
        path.name
        for path in (output_dir / "cases").iterdir()
        if path.suffix == ".json"
    }
    assert actual_names == expected_names
    assert (output_dir / "treatment.csv").is_file()


@pytest.mark.fenics
def test_parallel_failure_keeps_successful_case_but_writes_no_aggregate(tmp_path):
    config = tmp_path / "partial.yaml"
    config.write_text(
        "eta_values: [1.0]\n"
        "positive_danti_multipliers: [0.002]\n"
        "include_zero_reference: true\n"
        "zero_reference_eta: 1.0\n",
        encoding="utf-8",
    )
    cases = tuple(iter_treatment_cases(load_treatment_grid(config)))
    output_dir = tmp_path / "partial"
    mismatched = cases[0].result_row(
        set_id=756,
        initial_bacteria=1.771e-06,
        initial_macrophages=0.00260375,
        dB=0.00117535292215688,
        outputs=(1.0, 2.0, 3.0, 4.0),
    )
    mismatched.update(
        {
            "schema_version": CASE_SCHEMA_VERSION,
            "parameter_source": str(
                Path("data/parameters/calibrated/parameters_756.xml").resolve()
            ),
            "parameter_source_kind": "calibrated",
            "parameter_source_sha256": (
                "64ebbf3508d16a60c1c66df3387d53724"
                "a3ddce00d1e08e7cf632a3213e7d59c"
            ),
            "runtime_revision": compute_runtime_revision(),
            "mesh_backend": "rectangle",
            "resolved_mesh_backend": "rectangle",
            "resolution": 4,
            "dt": 0.2,
            "final_time": 0.0,
            "status": "success",
        }
    )
    mismatched_path = case_output_path(output_dir, cases[0])
    write_case_result(mismatched_path, mismatched)
    mismatched_bytes = mismatched_path.read_bytes()

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/sweep_danti_eta.py",
            "--set",
            "756",
            "--config",
            str(config),
            "--workers",
            "2",
            "--resolution",
            "4",
            "--mesh-backend",
            "rectangle",
            "--final-time",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "dt" in completed.stderr
    assert mismatched_path.read_bytes() == mismatched_bytes
    successful_path = case_output_path(output_dir, cases[1])
    assert successful_path.is_file()
    load_case_result(successful_path, expected_case=cases[1])
    assert not (output_dir / "treatment.csv").exists()
