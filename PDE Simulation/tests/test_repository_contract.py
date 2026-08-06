import json
from pathlib import Path
import subprocess

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import yaml

from lung_pde.constants import CALIBRATED_KEYS, FIXED_KEYS, SUPPORTED_SET_IDS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_documented_entry_points_and_inputs_exist():
    for script in (
        "calibrate.py",
        "simulate.py",
        "simulate_velocity.py",
        "sweep_danti_eta.py",
    ):
        assert (REPOSITORY_ROOT / "scripts" / script).is_file()

    for set_id in SUPPORTED_SET_IDS:
        assert (REPOSITORY_ROOT / f"data/abm/data_{set_id}.npz").is_file()
        assert (
            REPOSITORY_ROOT
            / f"data/parameters/initial/parameters_{set_id}.xml"
        ).is_file()
        assert (
            REPOSITORY_ROOT
            / f"data/parameters/calibrated/parameters_{set_id}.xml"
        ).is_file()

    for required in (
        "README.md",
        "docs/calibration_provenance.md",
        "docs/mathematical_forms.md",
        "docs/output_schema.md",
        ".github/workflows/tests.yml",
        "scripts/verify_repository.py",
    ):
        assert (REPOSITORY_ROOT / required).is_file()


def test_readme_records_the_scientific_contract_without_machine_local_commands():
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    for name in CALIBRATED_KEYS + FIXED_KEYS:
        assert f"`{name}`" in readme
    for phrase in (
        "83",
        "1/6",
        "test_final.py",
        "legacy_mshr",
        "rectangle",
        "danti_input",
        "danti_effective",
        "No open-source license",
        "No citation metadata",
    ):
        assert phrase in readme
    assert "/Users/" not in readme


def test_runtime_code_never_references_withdrawn_parameter_branch():
    runtime = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (
            REPOSITORY_ROOT / "src",
            REPOSITORY_ROOT / "scripts",
            REPOSITORY_ROOT / "configs",
        )
        for path in root.rglob("*.*")
        if path.is_file() and path.suffix in {".py", ".yaml"}
    )
    assert "param_final-m" not in runtime


def test_withdrawn_plotting_artifacts_are_not_tracked():
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    tracked = tuple(completed.stdout.splitlines())

    assert "scripts/make_figures.py" not in tracked
    assert not any(path.startswith("src/lung_pde/plotting/") for path in tracked)
    assert not any(path.startswith("data/treatment_reference/") for path in tracked)
    assert not any(
        Path(path).suffix.lower() in {".png", ".pdf", ".tex"}
        for path in tracked
    )


def test_data_manifest_covers_exactly_30_scientific_inputs():
    manifest = json.loads(
        (REPOSITORY_ROOT / "data/manifest.json").read_text(encoding="utf-8")
    )
    scientific_inputs = [
        entry
        for entry in manifest["inputs"]
        if entry["path"].startswith("data/abm/")
        or entry["path"].startswith("data/parameters/")
    ]

    assert len(scientific_inputs) == 30
    assert len({entry["path"] for entry in scientific_inputs}) == 30


def test_simulation_only_dependencies_exclude_plotting_stack():
    project = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project_dependencies = {
        dependency.split(">", 1)[0].split("<", 1)[0].lower()
        for dependency in project["project"]["dependencies"]
    }
    environment = yaml.safe_load(
        (REPOSITORY_ROOT / "environment.yml").read_text(encoding="utf-8")
    )
    conda_dependencies = {
        dependency.split(">", 1)[0].split("<", 1)[0].split("=", 1)[0].lower()
        for dependency in environment["dependencies"]
        if isinstance(dependency, str)
    }

    assert {"numpy", "scipy", "pyyaml"} <= project_dependencies
    assert {"python", "fenics", "numpy", "scipy", "pyyaml", "pytest"} <= (
        conda_dependencies
    )
    assert {"matplotlib", "pandas"}.isdisjoint(project_dependencies)
    assert {"matplotlib", "pandas"}.isdisjoint(conda_dependencies)
