import hashlib
import os
from pathlib import Path
import select
import subprocess
import sys

import pytest

from lung_pde.data import load_abm
from lung_pde.parameters import read_parameter_xml
from lung_pde.treatment import (
    CASE_SCHEMA_VERSION,
    TreatmentCase,
    case_output_path,
    compute_runtime_revision,
    iter_treatment_cases,
    load_treatment_grid,
    write_case_result,
)
from scripts import sweep_danti_eta
from scripts.sweep_danti_eta import build_parser


SCRIPT = Path("scripts/sweep_danti_eta.py")
ROOT = Path.cwd()
PARAMETER_SOURCE = Path("data/parameters/calibrated/parameters_756.xml").resolve()


def _run_cli(arguments, *, env=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
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
        "        raise RuntimeError('treatment utility imported DOLFIN')\n"
        "    return original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded_import\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path("src").resolve()))
    )
    return env


def _lock_contender_env(tmp_path, attempt_fd):
    guard_dir = tmp_path / "lock-guard"
    guard_dir.mkdir()
    (guard_dir / "sitecustomize.py").write_text(
        "import builtins\n"
        "import fcntl\n"
        "import os\n"
        "original_flock = fcntl.flock\n"
        "lock_attempt_signaled = False\n"
        "def signaling_flock(fd, operation):\n"
        "    global lock_attempt_signaled\n"
        "    if operation & fcntl.LOCK_EX and not lock_attempt_signaled:\n"
        "        os.write(int(os.environ['LOCK_ATTEMPT_FD']), b'lock-attempt\\n')\n"
        "        lock_attempt_signaled = True\n"
        "    return original_flock(fd, operation)\n"
        "fcntl.flock = signaling_flock\n"
        "original_import = builtins.__import__\n"
        "def guarded_import(name, *args, **kwargs):\n"
        "    if name == 'dolfin' or name.startswith('dolfin.'):\n"
        "        raise RuntimeError('lock contender imported DOLFIN')\n"
        "    return original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded_import\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["LOCK_ATTEMPT_FD"] = str(attempt_fd)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(guard_dir), str(Path("src").resolve()))
    )
    return env


def _wait_for_pipe_signal(fd, expected, process, timeout=10.0):
    readable, _, _ = select.select([fd], [], [], timeout)
    assert readable, f"timed out waiting for {expected!r}; exit={process.poll()}"
    assert os.read(fd, 4096) == expected


def _stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_cli_behind_holder(tmp_path, record):
    output_dir = tmp_path / "held-case"
    case = TreatmentCase(eta=1.0, danti_input=1.0)
    destination = case_output_path(output_dir, case)
    staged = tmp_path / "staged-case.json"
    write_case_result(staged, record)
    expected_bytes = staged.read_bytes()

    holder_script = tmp_path / "lock_holder.py"
    holder_script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "from lung_pde.treatment import (\n"
        "    exclusive_case_lock, load_case_result, write_case_result,\n"
        ")\n"
        "destination = Path(sys.argv[1])\n"
        "record = load_case_result(sys.argv[2])\n"
        "with exclusive_case_lock(destination):\n"
        "    print('holder-ready', flush=True)\n"
        "    if sys.stdin.readline() != 'release\\n':\n"
        "        raise RuntimeError('holder release signal missing')\n"
        "    write_case_result(destination, record)\n"
        "    print('holder-committed', flush=True)\n",
        encoding="utf-8",
    )
    holder_env = dict(os.environ)
    holder_env["PYTHONPATH"] = str(Path("src").resolve())
    holder = subprocess.Popen(
        [sys.executable, str(holder_script), str(destination), str(staged)],
        cwd=ROOT,
        env=holder_env,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    contender = None
    attempt_read, attempt_write = os.pipe()
    try:
        _wait_for_pipe_signal(
            holder.stdout.fileno(), b"holder-ready\n", holder
        )
        contender = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "--set",
                "756",
                "--case",
                "1.0",
                "1.0",
                "--resolution",
                "4",
                "--mesh-backend",
                "rectangle",
                "--output-dir",
                str(output_dir),
            ],
            cwd=ROOT,
            env=_lock_contender_env(tmp_path, attempt_write),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(attempt_write,),
        )
        os.close(attempt_write)
        attempt_write = None
        _wait_for_pipe_signal(attempt_read, b"lock-attempt\n", contender)

        assert contender.poll() is None
        assert not destination.exists()

        holder_stdout, holder_stderr = holder.communicate(
            "release\n", timeout=10
        )
        contender_stdout, contender_stderr = contender.communicate(timeout=10)
        assert holder.returncode == 0, holder_stderr
        assert "holder-committed" in holder_stdout
        assert destination.read_bytes() == expected_bytes
        assert Path(f"{destination}.lock").read_bytes() == b""
        return contender.returncode, contender_stdout, contender_stderr
    finally:
        os.close(attempt_read)
        if attempt_write is not None:
            os.close(attempt_write)
        _stop_process(contender)
        _stop_process(holder)


def _real_identity():
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "parameter_source": str(PARAMETER_SOURCE),
        "parameter_source_kind": "calibrated",
        "parameter_source_sha256": hashlib.sha256(
            PARAMETER_SOURCE.read_bytes()
        ).hexdigest(),
        "runtime_revision": compute_runtime_revision(),
        "mesh_backend": "rectangle",
        "resolved_mesh_backend": "rectangle",
        "resolution": 4,
        "dt": 0.1,
        "final_time": 14.0,
        "status": "success",
    }


def _real_record(case):
    abm = load_abm("data/abm/data_756.npz")
    parameters = read_parameter_xml(PARAMETER_SOURCE)
    record = case.result_row(
        set_id=756,
        initial_bacteria=abm.initial_bacteria,
        initial_macrophages=abm.initial_macrophages,
        dB=parameters["dB"],
        outputs=(1.0, 2.0, 3.0, 4.0),
    )
    record.update(_real_identity())
    return record


def _write_complete_grid(output_dir):
    grid = load_treatment_grid("configs/treatment_grid.yaml")
    for case in iter_treatment_cases(grid):
        write_case_result(case_output_path(output_dir, case), _real_record(case))


def test_parser_preserves_treatment_defaults_and_calibrated_only_contract():
    args = build_parser().parse_args(["--set", "756"])

    assert args.config == Path("configs/treatment_grid.yaml")
    assert args.workers == 1
    assert args.dt == 0.1
    assert args.final_time == 14.0
    assert args.parameter_source == "calibrated"


def test_validate_only_resolves_full_grid_without_importing_dolfin(tmp_path):
    output_dir = tmp_path / "validated"
    completed = _run_cli(
        [
            "--set",
            "756",
            "--resolution",
            "4",
            "--mesh-backend",
            "rectangle",
            "--output-dir",
            str(output_dir),
            "--validate-only",
        ],
        env=_dolfin_guard_env(tmp_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert "validation=ok" in completed.stdout
    assert "mode=treatment" in completed.stdout
    assert "authoritative_source=test_anti.py" in completed.stdout
    assert "parameter_source_kind=calibrated" in completed.stdout
    assert "data/parameters/calibrated/parameters_756.xml" in completed.stdout
    assert "dt=0.1" in completed.stdout
    assert "final_time=14.0" in completed.stdout
    assert "case_count=170" in completed.stdout
    assert f"output_dir={output_dir}" in completed.stdout
    assert "treatment utility imported DOLFIN" not in completed.stderr
    assert not output_dir.exists()


def test_single_case_does_not_load_or_expand_the_grid(tmp_path):
    output_dir = tmp_path / "slurm-case"
    completed = _run_cli(
        [
            "--set",
            "756",
            "--config",
            str(tmp_path / "does-not-exist.yaml"),
            "--case",
            "1.0",
            "1.0",
            "--output-dir",
            str(output_dir),
            "--validate-only",
        ],
        env=_dolfin_guard_env(tmp_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert "case_count=1" in completed.stdout
    assert f"case_keys={TreatmentCase(1.0, 1.0).key}" in completed.stdout
    assert not output_dir.exists()


def test_matching_successful_single_case_is_skipped_without_dolfin(tmp_path):
    output_dir = tmp_path / "resume"
    case = next(
        case
        for case in iter_treatment_cases(
            load_treatment_grid("configs/treatment_grid.yaml")
        )
        if case.eta == 1.0 and case.danti_input == 0.002
    )
    write_case_result(case_output_path(output_dir, case), _real_record(case))

    completed = _run_cli(
        [
            "--set",
            "756",
            "--case",
            "1.0",
            "0.002",
            "--resolution",
            "4",
            "--mesh-backend",
            "rectangle",
            "--output-dir",
            str(output_dir),
        ],
        env=_dolfin_guard_env(tmp_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=skipped" in completed.stdout
    assert "skipped=1" in completed.stdout
    assert "treatment utility imported DOLFIN" not in completed.stderr


def test_resume_uses_parameter_checksum_not_absolute_source_path(tmp_path):
    output_dir = tmp_path / "relocated"
    case = next(
        case
        for case in iter_treatment_cases(
            load_treatment_grid("configs/treatment_grid.yaml")
        )
        if case.eta == 1.0 and case.danti_input == 0.002
    )
    record = _real_record(case)
    record["parameter_source"] = (
        "/relocated/repository/data/parameters/calibrated/parameters_756.xml"
    )
    write_case_result(case_output_path(output_dir, case), record)

    completed = _run_cli(
        [
            "--set",
            "756",
            "--case",
            "1.0",
            "0.002",
            "--resolution",
            "4",
            "--mesh-backend",
            "rectangle",
            "--output-dir",
            str(output_dir),
        ],
        env=_dolfin_guard_env(tmp_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=skipped" in completed.stdout
    assert "treatment utility imported DOLFIN" not in completed.stderr


@pytest.mark.parametrize("field", ["B0", "M0", "dB"])
def test_resume_rejects_internally_consistent_scientific_metadata_corruption(
    tmp_path, field
):
    output_dir = tmp_path / field
    case = next(
        case
        for case in iter_treatment_cases(
            load_treatment_grid("configs/treatment_grid.yaml")
        )
        if case.eta == 1.0 and case.danti_input == 0.002
    )
    record = _real_record(case)
    record[field] *= 2.0
    record["danti_effective"] = (
        record["danti_input"] * record["M0"] * record["dB"]
    )
    write_case_result(case_output_path(output_dir, case), record)

    completed = _run_cli(
        [
            "--set",
            "756",
            "--case",
            "1.0",
            "0.002",
            "--resolution",
            "4",
            "--mesh-backend",
            "rectangle",
            "--output-dir",
            str(output_dir),
        ],
        env=_dolfin_guard_env(tmp_path),
    )

    assert completed.returncode == 2
    assert field in completed.stderr
    assert "--overwrite" in completed.stderr
    assert "treatment utility imported DOLFIN" not in completed.stderr


def test_resume_rejects_auto_case_when_current_resolution_changes(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "auto-changed"
    monkeypatch.setattr(
        sweep_danti_eta,
        "resolve_requested_mesh_backend",
        lambda name: "legacy_mshr",
        raising=False,
    )
    args = build_parser().parse_args(
        [
            "--set",
            "756",
            "--case",
            "1.0",
            "0.002",
            "--resolution",
            "4",
            "--mesh-backend",
            "auto",
            "--output-dir",
            str(output_dir),
        ]
    )
    inputs = sweep_danti_eta.resolve_treatment_inputs(args)
    case = inputs.cases[0]
    record = _real_record(case)
    record["mesh_backend"] = "auto"
    record["resolved_mesh_backend"] = "rectangle"
    write_case_result(case_output_path(output_dir, case), record)

    with pytest.raises(RuntimeError, match="resolved_mesh_backend"):
        sweep_danti_eta._run_cases(
            inputs,
            overwrite=False,
            command_arguments=("scripts/sweep_danti_eta.py",),
        )


def test_locked_matching_contender_blocks_then_revalidates_and_skips(tmp_path):
    case = TreatmentCase(eta=1.0, danti_input=1.0)

    returncode, stdout, stderr = _run_cli_behind_holder(
        tmp_path, _real_record(case)
    )

    assert returncode == 0, stderr
    assert "status=skipped" in stdout
    assert "completed=0" in stdout
    assert "skipped=1" in stdout
    assert "lock contender imported DOLFIN" not in stderr


def test_locked_mismatched_contender_blocks_then_fails_without_overwrite(
    tmp_path,
):
    case = TreatmentCase(eta=1.0, danti_input=1.0)
    mismatched = dict(_real_record(case), dt=0.2)

    returncode, stdout, stderr = _run_cli_behind_holder(
        tmp_path, mismatched
    )

    assert returncode == 2
    assert stdout == ""
    assert "dt" in stderr
    assert "--overwrite" in stderr
    assert "lock contender imported DOLFIN" not in stderr


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("set_id", 48),
        ("parameter_source_sha256", "b" * 64),
        ("runtime_revision", "b" * 64),
        ("mesh_backend", "auto"),
        ("resolution", 8),
        ("dt", 0.2),
        ("final_time", 7.0),
        ("status", "failed"),
        ("eta", 2.0),
    ],
)
def test_resume_mismatch_requires_overwrite(tmp_path, field, value):
    output_dir = tmp_path / field
    case = next(
        case
        for case in iter_treatment_cases(
            load_treatment_grid("configs/treatment_grid.yaml")
        )
        if case.eta == 1.0 and case.danti_input == 0.002
    )
    record = _real_record(case)
    record[field] = value
    destination = case_output_path(output_dir, case)
    destination.parent.mkdir(parents=True)
    destination.write_text(__import__("json").dumps(record), encoding="utf-8")

    completed = _run_cli(
        [
            "--set",
            "756",
            "--case",
            "1.0",
            "0.002",
            "--resolution",
            "4",
            "--mesh-backend",
            "rectangle",
            "--output-dir",
            str(output_dir),
        ],
        env=_dolfin_guard_env(tmp_path),
    )

    assert completed.returncode == 2
    assert "--overwrite" in completed.stderr
    assert "treatment utility imported DOLFIN" not in completed.stderr


def test_aggregate_only_validates_and_writes_without_importing_dolfin(tmp_path):
    output_dir = tmp_path / "aggregate"
    _write_complete_grid(output_dir)

    completed = _run_cli(
        [
            "--set",
            "756",
            "--aggregate-only",
            "--resolution",
            "4",
            "--mesh-backend",
            "rectangle",
            "--output-dir",
            str(output_dir),
        ],
        env=_dolfin_guard_env(tmp_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=aggregated" in completed.stdout
    assert "case_count=170" in completed.stdout
    assert (output_dir / "treatment.csv").is_file()
    assert "treatment utility imported DOLFIN" not in completed.stderr


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ("--validate-only", "--aggregate-only"),
            "not allowed with argument",
        ),
        (("--aggregate-only", "--overwrite"), "--overwrite cannot be combined"),
        (("--aggregate-only", "--workers", "2"), "--workers cannot be combined"),
        (("--aggregate-only", "--workers", "1"), "--workers cannot be combined"),
        (("--aggregate-only", "--work", "1"), "unrecognized arguments: --work 1"),
        (
            ("--aggregate-only", "--case", "1", "0"),
            "--case cannot be combined with --aggregate-only",
        ),
        (("--validate-only", "--overwrite"), "--overwrite cannot be combined"),
    ],
)
def test_cli_rejects_conflicting_or_ignored_modes(tmp_path, arguments, message):
    completed = _run_cli(
        ["--set", "756", *arguments],
        env=_dolfin_guard_env(tmp_path),
    )

    assert completed.returncode == 2
    assert message in completed.stderr
    assert "treatment utility imported DOLFIN" not in completed.stderr


def test_spawned_failures_are_reported_in_selected_case_order(tmp_path):
    config = tmp_path / "three-cases.yaml"
    config.write_text(
        "eta_values: [1.0]\n"
        "positive_danti_multipliers: [0.002, 0.01]\n"
        "include_zero_reference: true\n"
        "zero_reference_eta: 1.0\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "ordered-failures"
    cases = tuple(iter_treatment_cases(load_treatment_grid(config)))
    records = [_real_record(case) for case in cases]
    records[0]["dt"] = 0.2
    records[1]["final_time"] = 7.0
    for case, record in zip(cases, records):
        write_case_result(case_output_path(output_dir, case), record)

    completed = _run_cli(
        [
            "--set",
            "756",
            "--config",
            str(config),
            "--workers",
            "3",
            "--resolution",
            "4",
            "--mesh-backend",
            "rectangle",
            "--output-dir",
            str(output_dir),
        ],
        env=_dolfin_guard_env(tmp_path),
    )

    assert completed.returncode == 2
    first_position = completed.stderr.find(cases[0].key)
    second_position = completed.stderr.find(cases[1].key)
    assert 0 <= first_position < second_position
    assert "dt" in completed.stderr
    assert "final_time" in completed.stderr
    assert not (output_dir / "treatment.csv").exists()
    assert "treatment utility imported DOLFIN" not in completed.stderr


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--workers", "0"), "workers must be positive"),
        (("--dt", "0"), "dt must be finite and positive"),
        (("--dt", "nan"), "dt must be finite and positive"),
        (("--final-time", "-1"), "final time must be finite and non-negative"),
        (("--case", "0", "1"), "eta must be finite and positive"),
        (("--case", "1", "-1"), "danti must be finite and non-negative"),
    ],
)
def test_validation_rejects_invalid_execution_inputs(tmp_path, arguments, message):
    completed = _run_cli(
        ["--set", "756", *arguments, "--validate-only"],
        env=_dolfin_guard_env(tmp_path),
    )

    assert completed.returncode == 2
    assert message in completed.stderr


def test_resume_rejects_legacy_case_missing_runtime_revision(tmp_path):
    output_dir = tmp_path / "missing-revision"
    case = TreatmentCase(eta=1.0, danti_input=0.002)
    record = _real_record(case)
    record.pop("runtime_revision")
    destination = case_output_path(output_dir, case)
    destination.parent.mkdir(parents=True)
    destination.write_text(__import__("json").dumps(record), encoding="utf-8")

    completed = _run_cli(
        [
            "--set",
            "756",
            "--case",
            "1.0",
            "0.002",
            "--resolution",
            "4",
            "--mesh-backend",
            "rectangle",
            "--output-dir",
            str(output_dir),
        ],
        env=_dolfin_guard_env(tmp_path),
    )

    assert completed.returncode == 2
    assert "runtime_revision" in completed.stderr
    assert "--overwrite" in completed.stderr
    assert "treatment utility imported DOLFIN" not in completed.stderr


def test_parent_propagates_runtime_revision_to_worker_request(
    tmp_path, monkeypatch
):
    args = build_parser().parse_args(
        [
            "--set",
            "756",
            "--case",
            "1.0",
            "0.002",
            "--resolution",
            "4",
            "--mesh-backend",
            "rectangle",
            "--output-dir",
            str(tmp_path),
        ]
    )
    inputs = sweep_danti_eta.resolve_treatment_inputs(args)
    captured = []

    def capture_requests(requests, workers):
        captured.extend(requests)
        return [(request.case.key, "skipped", None) for request in requests]

    monkeypatch.setattr(sweep_danti_eta, "_run_pending_cases", capture_requests)

    sweep_danti_eta._run_cases(
        inputs,
        overwrite=False,
        command_arguments=("scripts/sweep_danti_eta.py",),
    )

    assert len(captured) == 1
    assert inputs.runtime_revision == compute_runtime_revision()
    assert captured[0].runtime_revision == inputs.runtime_revision
    assert captured[0].expected_identity["runtime_revision"] == inputs.runtime_revision
