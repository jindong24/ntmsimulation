import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

import lung_pde.treatment as treatment_module
from lung_pde.treatment import (
    CASE_SCHEMA_VERSION,
    RUNTIME_REVISION_SOURCES,
    TREATMENT_CSV_FIELDS,
    TreatmentCase,
    aggregate_treatment_cases,
    case_output_path,
    compute_runtime_revision,
    iter_treatment_cases,
    load_case_result,
    load_treatment_grid,
    write_case_result,
)


EXPECTED_ETA = (
    0.4,
    0.5,
    0.6,
    0.8,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
)
EXPECTED_DANTI = (
    0.002,
    0.01,
    0.05,
    0.25,
    1.25,
    6.25,
    31.25,
    156.25,
    500.0,
    2500.0,
    12500.0,
    62500.0,
    312500.0,
)


def _record(case, *, set_id=756, dt=0.1, status="success"):
    row = case.result_row(
        set_id=set_id,
        initial_bacteria=1.8865e-6,
        initial_macrophages=0.0012498,
        dB=0.0018954991640019,
        outputs=(
            case.eta + case.danti_input,
            case.eta + 2.0 * case.danti_input,
            case.eta + 3.0 * case.danti_input,
            case.eta + 4.0 * case.danti_input,
        ),
    )
    row.update(
        {
            "schema_version": CASE_SCHEMA_VERSION,
            "parameter_source": (
                "/repository/data/parameters/calibrated/parameters_756.xml"
            ),
            "parameter_source_kind": "calibrated",
            "parameter_source_sha256": "a" * 64,
            "runtime_revision": compute_runtime_revision(),
            "mesh_backend": "rectangle",
            "resolved_mesh_backend": "rectangle",
            "resolution": 4,
            "dt": dt,
            "final_time": 14.0,
            "status": status,
        }
    )
    return row


def _write_all_cases(case_dir):
    grid = load_treatment_grid("configs/treatment_grid.yaml")
    for case in iter_treatment_cases(grid):
        write_case_result(case_output_path(case_dir.parent, case), _record(case))
    return grid


def test_publication_grid_has_exact_169_positive_cases_plus_one_reference():
    grid = load_treatment_grid("configs/treatment_grid.yaml")
    cases = list(iter_treatment_cases(grid))

    assert grid.eta_values == EXPECTED_ETA
    assert grid.positive_danti_multipliers == EXPECTED_DANTI
    assert len(cases) == 170
    assert sum(case.danti_input == 0.0 for case in cases) == 1
    assert cases[0] == TreatmentCase(eta=1.0, danti_input=0.0)
    assert cases[1] == TreatmentCase(eta=0.4, danti_input=0.002)
    assert cases[-1] == TreatmentCase(eta=5.0, danti_input=312500.0)
    assert len({case.key for case in cases}) == 170


def test_effective_danti_keeps_input_and_legacy_value():
    grid = load_treatment_grid("configs/treatment_grid.yaml")
    case = next(
        case for case in iter_treatment_cases(grid) if case.danti_input == 0.002
    )
    row = case.result_row(
        set_id=48,
        initial_bacteria=1.8865e-6,
        initial_macrophages=0.0012498,
        dB=0.0018954991640019,
        outputs=(1.0, 2.0, 3.0, 4.0),
    )

    assert row["danti_input"] == 0.002
    assert row["danti_effective"] == (
        0.002 * 0.0012498 * 0.0018954991640019
    )
    assert row["B_MUCUS"] == 1.0
    assert row["B_TISSUE"] == 2.0
    assert row["M_MUCUS"] == 3.0
    assert row["M_TISSUE"] == 4.0


def test_case_key_and_output_path_are_stable_and_collision_free(tmp_path):
    case = TreatmentCase(eta=1.0, danti_input=0.002)

    assert case.key == (
        "eta_3ff0000000000000__danti_3f60624dd2f1a9fc"
    )
    assert case_output_path(tmp_path, case) == (
        tmp_path
        / "cases"
        / "eta_3ff0000000000000__danti_3f60624dd2f1a9fc.json"
    )


def test_adjacent_ieee_floats_have_distinct_lossless_filename_safe_keys(tmp_path):
    first = TreatmentCase(eta=1.0, danti_input=0.002)
    adjacent = TreatmentCase(
        eta=math.nextafter(1.0, math.inf),
        danti_input=0.002,
    )

    assert first.key != adjacent.key
    assert adjacent.key.startswith("eta_3ff0000000000001__danti_")
    assert set(first.key + adjacent.key) <= set(
        "eta_dnti_0123456789abcdef"
    )
    assert case_output_path(tmp_path, first) != case_output_path(
        tmp_path, adjacent
    )


@pytest.mark.parametrize(
    ("eta", "danti", "message"),
    [
        (float("nan"), 1.0, "eta must be a finite number"),
        (float("inf"), 1.0, "eta must be a finite number"),
        (0.0, 1.0, "eta must be finite and positive"),
        (1.0, float("nan"), "danti must be a finite number"),
        (1.0, float("inf"), "danti must be a finite number"),
        (1.0, -1.0, "danti must be finite and non-negative"),
        (True, 1.0, "eta must be a finite number"),
        (1.0, False, "danti must be a finite number"),
    ],
)
def test_case_rejects_invalid_domains_before_key_or_path(eta, danti, message):
    with pytest.raises(ValueError, match=message):
        TreatmentCase(eta=eta, danti_input=danti)


def test_grid_rejects_duplicate_values_that_would_share_case_files(tmp_path):
    config = tmp_path / "duplicate.yaml"
    config.write_text(
        "eta_values: [1.0, 1.0]\n"
        "positive_danti_multipliers: [0.002]\n"
        "include_zero_reference: true\n"
        "zero_reference_eta: 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="eta_values must be unique"):
        load_treatment_grid(config)


def test_grid_rejects_duplicate_yaml_mapping_keys(tmp_path):
    config = tmp_path / "duplicate-key.yaml"
    config.write_text(
        "eta_values: [1.0]\n"
        "eta_values: [2.0]\n"
        "positive_danti_multipliers: [0.002]\n"
        "include_zero_reference: true\n"
        "zero_reference_eta: 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate YAML key: eta_values"):
        load_treatment_grid(config)


def test_case_json_round_trip_is_atomic_and_self_validating(tmp_path):
    case = TreatmentCase(eta=1.0, danti_input=0.002)
    destination = case_output_path(tmp_path, case)
    record = _record(case)

    write_case_result(destination, record)

    assert load_case_result(destination, expected_case=case) == record
    assert not list(destination.parent.glob(f".{destination.name}.*"))


def test_case_json_rejects_duplicate_properties_before_validation(tmp_path):
    case = TreatmentCase(eta=1.0, danti_input=0.002)
    destination = case_output_path(tmp_path, case)
    destination.parent.mkdir()
    encoded = json.dumps(_record(case))
    destination.write_text(
        encoded.replace("{", '{"dt": 0.2,', 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON property: dt"):
        load_case_result(destination, expected_case=case)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_case_json_rejects_nonstandard_numeric_constants(tmp_path, constant):
    case = TreatmentCase(eta=1.0, danti_input=0.002)
    destination = case_output_path(tmp_path, case)
    destination.parent.mkdir()
    encoded = json.dumps(_record(case))
    destination.write_text(
        encoded.replace('"B_MUCUS": 1.002', f'"B_MUCUS": {constant}'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-standard JSON constant"):
        load_case_result(destination, expected_case=case)


@pytest.mark.parametrize("schema_version", [True, 1.0, "2", 1])
def test_case_json_schema_version_requires_exact_integer_two(
    tmp_path, schema_version
):
    case = TreatmentCase(eta=1.0, danti_input=0.002)
    destination = case_output_path(tmp_path, case)
    destination.parent.mkdir()
    record = _record(case)
    record["schema_version"] = schema_version
    destination.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version must be integer 2"):
        load_case_result(destination, expected_case=case)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("case_key", "eta_1__danti_0.01", "case identity"),
        ("eta", True, "eta must be a JSON number"),
        ("danti_input", False, "danti_input must be a JSON number"),
        ("danti_effective", 42.0, "danti_effective"),
        ("B0", True, "B0 must be a JSON number"),
        ("B_MUCUS", float("nan"), "non-standard JSON constant: NaN"),
        ("dt", True, "dt must be a JSON number"),
        ("dt", "0.1", "dt must be a JSON number"),
        ("set_id", True, "set_id must be an integer"),
        ("resolution", True, "resolution must be an integer"),
        ("status", "failed", "successful status"),
        ("parameter_source_kind", "initial", "calibrated"),
        ("runtime_revision", "not-a-digest", "runtime_revision"),
        (
            "resolved_mesh_backend",
            "legacy_mshr",
            "resolved_mesh_backend",
        ),
    ],
)
def test_case_json_rejects_invalid_scientific_or_identity_fields(
    tmp_path, field, value, message
):
    case = TreatmentCase(eta=1.0, danti_input=0.002)
    destination = case_output_path(tmp_path, case)
    record = _record(case)
    record[field] = value
    destination.parent.mkdir()
    destination.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_case_result(destination, expected_case=case)


def test_aggregation_validates_exact_set_and_writes_stable_sorted_csv(tmp_path):
    case_dir = tmp_path / "cases"
    _write_all_cases(case_dir)
    output_csv = tmp_path / "treatment.csv"

    aggregate_treatment_cases(case_dir, output_csv)
    first_bytes = output_csv.read_bytes()
    aggregate_treatment_cases(case_dir, output_csv)

    assert output_csv.read_bytes() == first_bytes
    assert not list(tmp_path.glob(".treatment.csv.*"))
    with output_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames) == TREATMENT_CSV_FIELDS
    assert len(rows) == 170
    assert (float(rows[0]["eta"]), float(rows[0]["danti_input"])) == (
        1.0,
        0.0,
    )
    positive_pairs = [
        (float(row["eta"]), float(row["danti_input"])) for row in rows[1:]
    ]
    assert positive_pairs == sorted(positive_pairs)
    assert all(row["status"] == "success" for row in rows)


def test_aggregation_rejects_missing_duplicate_and_mismatched_cases(tmp_path):
    case_dir = tmp_path / "cases"
    grid = _write_all_cases(case_dir)
    cases = list(iter_treatment_cases(grid))
    missing_path = case_output_path(tmp_path, cases[-1])
    missing_record = load_case_result(missing_path, expected_case=cases[-1])
    missing_path.unlink()

    with pytest.raises(ValueError, match="missing case keys"):
        aggregate_treatment_cases(case_dir, tmp_path / "missing.csv")

    write_case_result(missing_path, missing_record)
    duplicate_path = case_dir / "duplicate.json"
    duplicate_path.write_text(missing_path.read_text(), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case identity"):
        aggregate_treatment_cases(case_dir, tmp_path / "duplicate.csv")

    duplicate_path.unlink()
    mismatched = load_case_result(missing_path, expected_case=cases[-1])
    mismatched["parameter_source_sha256"] = "b" * 64
    write_case_result(missing_path, mismatched)
    with pytest.raises(ValueError, match="parameter_source_sha256"):
        aggregate_treatment_cases(case_dir, tmp_path / "mismatch.csv")


def test_aggregation_can_enforce_the_requested_run_identity(tmp_path):
    case_dir = tmp_path / "cases"
    _write_all_cases(case_dir)
    expected_identity = {
        "set_id": 756,
        "parameter_source_sha256": "a" * 64,
        "mesh_backend": "rectangle",
        "resolution": 4,
        "dt": 0.1,
        "final_time": 14.0,
        "B0": 1.8865e-6,
        "M0": 0.0012498,
        "dB": 0.0018954991640019,
    }

    aggregate_treatment_cases(
        case_dir,
        tmp_path / "treatment.csv",
        expected_identity=expected_identity,
    )

    expected_identity["dt"] = 0.2
    with pytest.raises(ValueError, match="dt"):
        aggregate_treatment_cases(
            case_dir,
            tmp_path / "wrong-dt.csv",
            expected_identity=expected_identity,
        )


@pytest.mark.parametrize("field", ["B0", "M0", "dB"])
def test_aggregation_rejects_authoritative_scientific_metadata_corruption(
    tmp_path, field
):
    case_dir = tmp_path / "cases"
    grid = _write_all_cases(case_dir)
    cases = tuple(iter_treatment_cases(grid))
    target = cases[5]
    destination = case_output_path(tmp_path, target)
    record = load_case_result(destination, expected_case=target)
    record[field] *= 2.0
    record["danti_effective"] = (
        record["danti_input"] * record["M0"] * record["dB"]
    )
    write_case_result(destination, record)

    expected_identity = {
        "set_id": 756,
        "parameter_source_sha256": "a" * 64,
        "mesh_backend": "rectangle",
        "resolved_mesh_backend": "rectangle",
        "resolution": 4,
        "dt": 0.1,
        "final_time": 14.0,
        "B0": 1.8865e-6,
        "M0": 0.0012498,
        "dB": 0.0018954991640019,
    }
    with pytest.raises(ValueError) as error:
        aggregate_treatment_cases(
            case_dir,
            tmp_path / "corrupt.csv",
            expected_identity=expected_identity,
        )
    assert str(error.value) == f"case identity mismatch: {field}"


def test_common_identity_includes_scientific_initial_values_and_db(tmp_path):
    case_dir = tmp_path / "cases"
    grid = _write_all_cases(case_dir)
    target = tuple(iter_treatment_cases(grid))[5]
    destination = case_output_path(tmp_path, target)
    record = load_case_result(destination, expected_case=target)
    record["M0"] *= 2.0
    record["danti_effective"] = (
        record["danti_input"] * record["M0"] * record["dB"]
    )
    write_case_result(destination, record)

    with pytest.raises(ValueError, match="M0"):
        aggregate_treatment_cases(case_dir, tmp_path / "corrupt.csv")


def test_auto_backend_resolution_is_pure_and_tracks_current_availability(
    monkeypatch,
):
    resolver = getattr(
        treatment_module,
        "resolve_requested_mesh_backend",
        None,
    )
    assert callable(resolver)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "mshr" else None,
    )
    assert resolver("auto") == "legacy_mshr"

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None,
    )
    assert resolver("auto") == "rectangle"
    assert resolver("rectangle") == "rectangle"


def test_calibrated_parameter_checksum_used_by_cases_is_sha256():
    source = Path("data/parameters/calibrated/parameters_756.xml")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    assert len(digest) == 64
    assert digest == (
        "64ebbf3508d16a60c1c66df3387d53724a3ddce00d1e08e7cf632a3213e7d59c"
    )


def _write_runtime_revision_tree(root, line_ending):
    for relative_name in RUNTIME_REVISION_SOURCES:
        destination = root / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = f"# deterministic fixture for {relative_name}{line_ending}"
        destination.write_bytes(content.encode("utf-8"))


def test_runtime_revision_is_portable_across_roots_and_line_endings(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_runtime_revision_tree(first_root, "\n")
    _write_runtime_revision_tree(second_root, "\r\n")

    first = compute_runtime_revision(repository_root=first_root)
    second = compute_runtime_revision(repository_root=second_root)

    assert first == second
    assert len(first) == 64


def test_runtime_revision_changes_with_runtime_source(tmp_path):
    repository_root = tmp_path / "repository"
    _write_runtime_revision_tree(repository_root, "\n")
    initial_revision = compute_runtime_revision(repository_root=repository_root)
    solver_source = repository_root / "src" / "lung_pde" / "solver.py"
    solver_source.write_text(
        solver_source.read_text(encoding="utf-8") + "SOLVER_CHANGE = True\n",
        encoding="utf-8",
    )
    assert compute_runtime_revision(repository_root=repository_root) != initial_revision


def test_runtime_revision_changes_with_sweep_entry_point(tmp_path):
    repository_root = tmp_path / "repository"
    _write_runtime_revision_tree(repository_root, "\n")
    sweep_source = repository_root / "scripts" / "sweep_danti_eta.py"
    initial_revision = compute_runtime_revision(repository_root=repository_root)

    sweep_source.write_text(
        "# deterministic sweep fixture\nSWEEP_CHANGE = True\n",
        encoding="utf-8",
    )

    assert compute_runtime_revision(repository_root=repository_root) != initial_revision


def test_case_json_rejects_legacy_record_without_runtime_revision(tmp_path):
    case = TreatmentCase(eta=1.0, danti_input=0.002)
    destination = case_output_path(tmp_path, case)
    destination.parent.mkdir(parents=True)
    record = _record(case)
    record.pop("runtime_revision")
    destination.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="missing runtime_revision"):
        load_case_result(destination, expected_case=case)


def test_case_identity_rejects_changed_runtime_revision(tmp_path):
    case = TreatmentCase(eta=1.0, danti_input=0.002)
    destination = case_output_path(tmp_path, case)
    record = _record(case)
    write_case_result(destination, record)

    with pytest.raises(ValueError, match="runtime_revision"):
        load_case_result(
            destination,
            expected_case=case,
            expected_identity={"runtime_revision": "b" * 64},
        )
