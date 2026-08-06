import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


SCRIPT = Path("scripts/verify_repository.py")


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_repository", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_static_checks_validate_inputs_configs_forms_clis_and_scope():
    verifier = _load_verifier()

    assert "30 scientific inputs" in verifier.verify_data_inputs(Path.cwd())
    assert "170 cases" in verifier.verify_configurations(Path.cwd())
    assert "four CLI" in verifier.verify_cli_surfaces(Path.cwd())
    assert "AST parity" in verifier.verify_weak_forms(Path.cwd())
    assert "absent" in verifier.verify_withdrawn_artifacts(Path.cwd())


def test_report_writer_is_deterministic_and_distinguishes_statuses(tmp_path):
    verifier = _load_verifier()
    records = [
        verifier.CheckRecord("data_inputs", "pass", "ok"),
        verifier.CheckRecord("fenics_smoke", "skip", "DOLFIN unavailable"),
        verifier.CheckRecord("pure_tests", "fail", "exit 1"),
    ]
    output = tmp_path / "report.json"

    verifier.write_report(output, records)
    first = output.read_bytes()
    verifier.write_report(output, records)

    assert output.read_bytes() == first
    report = json.loads(first)
    assert report == {
        "checks": [
            {"detail": "ok", "name": "data_inputs", "status": "pass"},
            {
                "detail": "DOLFIN unavailable",
                "name": "fenics_smoke",
                "status": "skip",
            },
            {"detail": "exit 1", "name": "pure_tests", "status": "fail"},
        ],
        "overall_status": "fail",
        "schema_version": 1,
    }


def _copy_data_contract(tmp_path):
    shutil.copytree("data", tmp_path / "data")
    fixture_dir = tmp_path / "tests/fixtures"
    fixture_dir.mkdir(parents=True)
    shutil.copy2(
        "tests/fixtures/legacy_756_conc.npz",
        fixture_dir / "legacy_756_conc.npz",
    )
    return tmp_path


def _refresh_manifest_entry(root, relative_path):
    manifest_path = root / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = root / relative_path
    entry = next(
        item for item in manifest["inputs"] if item["path"] == relative_path
    )
    entry["bytes"] = source.stat().st_size
    entry["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_form_contract(tmp_path):
    shutil.copytree("docs/provenance", tmp_path / "docs/provenance")
    shutil.copytree("src/lung_pde/forms", tmp_path / "src/lung_pde/forms")
    return tmp_path


def test_corrupted_scientific_input_becomes_a_failed_check(tmp_path):
    verifier = _load_verifier()
    root = _copy_data_contract(tmp_path)
    input_path = root / "data/abm/data_48.npz"
    payload = bytearray(input_path.read_bytes())
    payload[-1] ^= 1
    input_path.write_bytes(payload)

    record = verifier._record(
        "data_inputs", lambda: verifier.verify_data_inputs(root)
    )

    assert record == verifier.CheckRecord(
        "data_inputs", "fail", "check failed (ValueError)"
    )


@pytest.mark.parametrize("kind", ["abm", "xml"])
def test_structural_input_corruption_fails_even_with_refreshed_manifest_hash(
    tmp_path, kind
):
    verifier = _load_verifier()
    root = _copy_data_contract(tmp_path)
    if kind == "abm":
        relative_path = "data/abm/data_48.npz"
        source = root / relative_path
        with np.load(source, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        arrays["t"] = arrays["t"] + 1.0 / 6.0
        np.savez(source, **arrays)
    else:
        relative_path = "data/parameters/initial/parameters_48.xml"
        source = root / relative_path
        encoded = source.read_text(encoding="utf-8")
        source.write_text(
            encoded.replace('key="DB1" type="double" value="1e-6"',
                            'key="DB1" type="double" value="nan"', 1),
            encoding="utf-8",
        )
    _refresh_manifest_entry(root, relative_path)

    with pytest.raises(ValueError):
        verifier.verify_data_inputs(root)


@pytest.mark.parametrize("corruption", ["fixture", "form"])
def test_corrupted_protected_fixture_or_form_becomes_a_failed_check(
    tmp_path, corruption
):
    verifier = _load_verifier()
    root = _copy_form_contract(tmp_path)
    if corruption == "fixture":
        target = root / "docs/provenance/legacy_form_blocks/test_final.py"
        target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        target = root / "src/lung_pde/forms/baseline.py"
        source = target.read_text(encoding="utf-8")
        target.write_text(source.replace("a = a_B1 +", "a = a_B2 +", 1), encoding="utf-8")

    record = verifier._record(
        "weak_forms", lambda: verifier.verify_weak_forms(root)
    )

    assert record == verifier.CheckRecord(
        "weak_forms", "fail", "check failed (ValueError)"
    )


def test_subprocess_failure_detail_is_normalized_and_byte_stable(
    tmp_path, monkeypatch
):
    verifier = _load_verifier()
    outputs = iter(
        (
            "/private/tmp/run-a/test.py: failed in 0.91s",
            "/different/machine/run-b/test.py: failed in 8.31s",
        )
    )

    def fail(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            7,
            stdout="",
            stderr=next(outputs),
        )

    monkeypatch.setattr(verifier.subprocess, "run", fail)
    records = []
    for interpreter, target in (
        ("/Users/example/env/bin/python", "/private/tmp/run-a/test.py"),
        ("/opt/conda/env/bin/python", "/different/machine/run-b/test.py"),
    ):
        records.append(
            verifier._record(
                "pure_tests",
                lambda interpreter=interpreter, target=target: verifier._run_command(
                    [interpreter, "-m", "pytest", target], tmp_path
                ),
            )
        )

    expected = (
        'command failed: ["python","-m","pytest","<absolute-path>"]; exit=7'
    )
    assert [record.detail for record in records] == [expected, expected]
    assert "/Users/" not in expected
    assert "/private/" not in expected


def test_record_converts_arbitrary_exception_to_stable_failure(
    capsys,
):
    verifier = _load_verifier()

    def fail():
        raise RuntimeError("temporary failure at /private/tmp/random-123")

    record = verifier._record("example", fail)

    assert record == verifier.CheckRecord(
        "example", "fail", "check failed (RuntimeError)"
    )
    assert "temporary failure" in capsys.readouterr().err


def test_main_writes_fail_report_and_returns_nonzero_for_required_failure(
    tmp_path, monkeypatch
):
    verifier = _load_verifier()
    monkeypatch.setattr(verifier, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        verifier,
        "verify_data_inputs",
        lambda root: (_ for _ in ()).throw(ValueError("machine path /tmp/a")),
    )
    for name in (
        "verify_configurations",
        "verify_cli_surfaces",
        "verify_weak_forms",
        "verify_withdrawn_artifacts",
    ):
        monkeypatch.setattr(verifier, name, lambda root, name=name: f"{name} ok")
    monkeypatch.setattr(
        verifier,
        "_pytest_record",
        lambda root, *, fenics, skip_slow: verifier.CheckRecord(
            "fenics_tests" if fenics else "pure_tests",
            "skip" if fenics else "pass",
            "DOLFIN unavailable" if fenics else "pytest passed",
        ),
    )
    monkeypatch.setattr(
        verifier,
        "_mshr_record",
        lambda: verifier.CheckRecord("legacy_mshr", "skip", "mshr unavailable"),
    )

    exit_code = verifier.main(["--skip-slow"])

    report = json.loads(
        (tmp_path / "outputs/verification/report.json").read_text(encoding="utf-8")
    )
    assert exit_code == 1
    assert report["overall_status"] == "fail"
    assert report["checks"][0] == {
        "detail": "check failed (ValueError)",
        "name": "data_inputs",
        "status": "fail",
    }
    assert "/tmp/a" not in json.dumps(report)


def test_unavailable_fenics_and_mshr_are_explicit_skips(tmp_path, monkeypatch):
    verifier = _load_verifier()
    monkeypatch.setattr(verifier.importlib.util, "find_spec", lambda name: None)

    fenics = verifier._pytest_record(tmp_path, fenics=True, skip_slow=True)
    mshr = verifier._mshr_record()

    assert fenics.status == "skip"
    assert fenics.name == "fenics_tests"
    assert "DOLFIN unavailable" in fenics.detail
    assert mshr.status == "skip"
    assert mshr.name == "legacy_mshr"
    assert "mshr unavailable" in mshr.detail
