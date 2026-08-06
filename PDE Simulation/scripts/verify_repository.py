#!/usr/bin/env python3
"""Verify the portable, simulation-only repository contract."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable, NamedTuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lung_pde.calibration_config import load_pipeline
from lung_pde.constants import CALIBRATED_KEYS, SUPPORTED_SET_IDS
from lung_pde.data import load_abm
from lung_pde.parameters import read_parameter_xml
from lung_pde.treatment import iter_treatment_cases, load_treatment_grid


TERM_NAMES = ("a_B1", "a_B2", "a_M1", "a_M2", "a_m", "a")


class CheckRecord(NamedTuple):
    name: str
    status: str
    detail: str


class CommandFailure(RuntimeError):
    def __init__(self, identity: str, returncode: int, diagnostic: str) -> None:
        super().__init__(diagnostic)
        self.identity = identity
        self.returncode = int(returncode)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_data_inputs(root: Path) -> str:
    manifest_path = root / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("inputs"), list):
        raise ValueError("data manifest must contain an inputs list")

    entries = manifest["inputs"]
    paths = [entry.get("path") for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("data manifest contains duplicate paths")

    expected_scientific = {
        *(f"data/abm/data_{set_id}.npz" for set_id in SUPPORTED_SET_IDS),
        *(
            f"data/parameters/{kind}/parameters_{set_id}.xml"
            for kind in ("initial", "calibrated")
            for set_id in SUPPORTED_SET_IDS
        ),
    }
    scientific = {
        path
        for path in paths
        if isinstance(path, str)
        and (path.startswith("data/abm/") or path.startswith("data/parameters/"))
    }
    if scientific != expected_scientific:
        missing = sorted(expected_scientific - scientific)
        extra = sorted(scientific - expected_scientific)
        raise ValueError(f"scientific input mismatch; missing={missing}, extra={extra}")

    for entry in entries:
        if set(entry) != {"path", "source_path", "bytes", "sha256"}:
            raise ValueError(f"invalid manifest entry fields: {entry.get('path')}")
        path = root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"manifest input is missing: {entry['path']}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"byte-size mismatch: {entry['path']}")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {entry['path']}")

    for set_id in SUPPORTED_SET_IDS:
        load_abm(root / f"data/abm/data_{set_id}.npz")
        for kind in ("initial", "calibrated"):
            read_parameter_xml(
                root / f"data/parameters/{kind}/parameters_{set_id}.xml"
            )

    return "30 scientific inputs structurally and all manifest checksums verified"


def verify_configurations(root: Path) -> str:
    pipeline_paths = sorted(
        path
        for path in (root / "configs/calibration").rglob("*.yaml")
        if path.name != "parameters.yaml"
    )
    if len(pipeline_paths) != 4:
        raise ValueError("expected four calibration pipeline YAML files")
    for path in pipeline_paths:
        load_pipeline(path)

    import yaml

    catalog = yaml.safe_load(
        (root / "configs/calibration/parameters.yaml").read_text(encoding="utf-8")
    )
    names = tuple(entry["name"] for entry in catalog["parameters"])
    if names != CALIBRATED_KEYS:
        raise ValueError("calibration parameter catalog does not match CALIBRATED_KEYS")

    grid = load_treatment_grid(root / "configs/treatment_grid.yaml")
    cases = tuple(iter_treatment_cases(grid))
    positive_count = sum(case.danti_input > 0.0 for case in cases)
    zero_count = sum(case.danti_input == 0.0 for case in cases)
    if (len(cases), positive_count, zero_count) != (170, 169, 1):
        raise ValueError("treatment grid must contain 169 positive cases and one zero")
    return "four calibration pipelines, parameter catalog, and 170 cases verified"


def _command_identity(command: list[str], root: Path) -> str:
    normalized = []
    resolved_root = root.resolve()
    for index, raw_argument in enumerate(command):
        argument = str(raw_argument)
        path = Path(argument)
        if index == 0:
            executable = path.name
            normalized.append(
                "python" if executable.startswith("python") else executable
            )
        elif path.is_absolute():
            try:
                normalized.append(path.resolve().relative_to(resolved_root).as_posix())
            except ValueError:
                normalized.append("<absolute-path>")
        else:
            normalized.append(argument)
    return json.dumps(normalized, separators=(",", ":"), ensure_ascii=True)


def _run_command(command: list[str], root: Path) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        output = (completed.stderr or completed.stdout).strip().splitlines()
        diagnostic = output[-1] if output else "no diagnostic output"
        raise CommandFailure(
            _command_identity(command, root), completed.returncode, diagnostic
        )
    return completed


def verify_cli_surfaces(root: Path) -> str:
    python = sys.executable
    scripts = (
        "scripts/calibrate.py",
        "scripts/simulate.py",
        "scripts/simulate_velocity.py",
        "scripts/sweep_danti_eta.py",
    )
    for script in scripts:
        _run_command([python, script, "--help"], root)

    validations = (
        [
            python,
            "scripts/calibrate.py",
            "--set",
            "756",
            "--pipeline",
            "configs/calibration/legacy_reconstructed.yaml",
            "--validate-only",
        ],
        [python, "scripts/simulate.py", "--set", "756", "--validate-only"],
        [
            python,
            "scripts/simulate_velocity.py",
            "--set",
            "756",
            "--validate-only",
        ],
        [
            python,
            "scripts/sweep_danti_eta.py",
            "--set",
            "756",
            "--validate-only",
        ],
    )
    for command in validations:
        _run_command(command, root)
    return "four CLI help and validation surfaces verified"


def _assignment_name(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _protected_assignments(statements: list[ast.stmt]) -> tuple[ast.Assign, ...]:
    assignments = tuple(
        statement
        for statement in statements
        if _assignment_name(statement) in TERM_NAMES
    )
    names = tuple(_assignment_name(statement) for statement in assignments)
    if names != TERM_NAMES:
        raise ValueError(f"protected form terms have unexpected order: {names}")
    return assignments


def _fixture_trees(path: Path) -> tuple[str, ...]:
    statements = ast.parse(path.read_text(encoding="utf-8")).body
    if len(statements) != len(TERM_NAMES):
        raise ValueError(f"fixture has unexpected statements: {path}")
    return tuple(
        ast.dump(statement.value, include_attributes=False)
        for statement in _protected_assignments(statements)
    )


def _target_trees(path: Path) -> tuple[str, ...]:
    functions = [
        statement
        for statement in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == "build_form"
    ]
    if len(functions) != 1:
        raise ValueError(f"expected one build_form function: {path}")
    return tuple(
        ast.dump(statement.value, include_attributes=False)
        for statement in _protected_assignments(functions[0].body)
    )


def verify_weak_forms(root: Path) -> str:
    manifest_path = root / "docs/provenance/weak_form_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_targets = {
        "src/lung_pde/forms/baseline.py",
        "src/lung_pde/forms/calibration.py",
        "src/lung_pde/forms/velocity.py",
        "src/lung_pde/forms/treatment.py",
    }
    observed_targets = {
        target for entry in manifest for target in entry.get("targets", ())
    }
    if observed_targets != expected_targets:
        raise ValueError("weak-form manifest target coverage is incomplete")

    for entry in manifest:
        fixture = root / entry["fixture"]
        if _sha256(fixture) != entry["sha256"]:
            raise ValueError(f"weak-form fixture checksum mismatch: {entry['fixture']}")
        fixture_trees = _fixture_trees(fixture)
        for target_name in entry["targets"]:
            if _target_trees(root / target_name) != fixture_trees:
                raise ValueError(f"weak-form AST mismatch: {target_name}")
    return "three fixture checksums and four protected-form AST parity checks verified"


def verify_withdrawn_artifacts(root: Path) -> str:
    completed = _run_command(["git", "ls-files"], root)
    tracked = tuple(completed.stdout.splitlines())
    forbidden = [
        path
        for path in tracked
        if path == "scripts/make_figures.py"
        or path.startswith("src/lung_pde/plotting/")
        or path.startswith("data/treatment_reference/")
        or Path(path).suffix.lower() in {".png", ".pdf", ".tex"}
    ]
    if forbidden:
        raise ValueError("withdrawn plotting artifacts are tracked: " + ", ".join(forbidden))
    return "withdrawn plotting, heatmap, figure, and LaTeX artifacts are absent"


def _failure_record(name: str, error: Exception) -> CheckRecord:
    print(f"{name} diagnostic: {type(error).__name__}: {error}", file=sys.stderr)
    if isinstance(error, CommandFailure):
        detail = (
            f"command failed: {error.identity}; exit={error.returncode}"
        )
    else:
        detail = f"check failed ({type(error).__name__})"
    return CheckRecord(name, "fail", detail)


def _record(name: str, check: Callable[[], str]) -> CheckRecord:
    try:
        return CheckRecord(name, "pass", check())
    except Exception as error:
        return _failure_record(name, error)


def _pytest_record(root: Path, *, fenics: bool, skip_slow: bool) -> CheckRecord:
    name = "fenics_tests" if fenics else "pure_tests"
    if fenics and importlib.util.find_spec("dolfin") is None:
        return CheckRecord(name, "skip", "DOLFIN unavailable in this interpreter")

    marker = "fenics" if fenics else "not fenics"
    if skip_slow:
        marker += " and not slow"
    command = [sys.executable, "-m", "pytest", "-m", marker, "-v"]
    try:
        _run_command(command, root)
    except Exception as error:
        return _failure_record(name, error)
    backend = "rectangle smoke tests" if fenics else "import-safe tests"
    return CheckRecord(name, "pass", f"pytest passed ({backend}; marker: {marker})")


def _mshr_record() -> CheckRecord:
    if importlib.util.find_spec("mshr") is None:
        return CheckRecord(
            "legacy_mshr",
            "skip",
            "mshr unavailable; legacy_mshr numerical reproduction was not run",
        )
    return CheckRecord(
        "legacy_mshr",
        "pass",
        "mshr importable; requires_mshr tests were eligible to run",
    )


def write_report(path: Path, records: list[CheckRecord]) -> None:
    payload = {
        "schema_version": 1,
        "overall_status": (
            "fail" if any(record.status == "fail" for record in records) else "pass"
        ),
        "checks": [record._asdict() for record in records],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify scientific inputs, workflow contracts, and tests."
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="exclude tests marked slow from pure-Python and FEniCS pytest runs",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = REPOSITORY_ROOT
    records = [
        _record("data_inputs", lambda: verify_data_inputs(root)),
        _record("configurations", lambda: verify_configurations(root)),
        _record("cli_surfaces", lambda: verify_cli_surfaces(root)),
        _record("weak_forms", lambda: verify_weak_forms(root)),
        _record("withdrawn_artifacts", lambda: verify_withdrawn_artifacts(root)),
        _pytest_record(root, fenics=False, skip_slow=args.skip_slow),
        _pytest_record(root, fenics=True, skip_slow=args.skip_slow),
    ]
    records.append(_mshr_record())

    report_path = root / "outputs/verification/report.json"
    write_report(report_path, records)
    for record in records:
        print(f"{record.status.upper():4} {record.name}: {record.detail}")
    print(f"report={report_path.relative_to(root)}")
    return 1 if any(record.status == "fail" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
