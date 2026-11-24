from collections.abc import Iterable, Mapping
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import tempfile

import yaml


CASE_SCHEMA_VERSION = 2
CASE_STATUS_SUCCESS = "success"
DEFAULT_TREATMENT_GRID = (
    Path(__file__).resolve().parents[2] / "configs" / "treatment_grid.yaml"
)

RUNTIME_REVISION_SOURCES = (
    "scripts/sweep_danti_eta.py",
    "src/lung_pde/constants.py",
    "src/lung_pde/data.py",
    "src/lung_pde/form_context.py",
    "src/lung_pde/forms/__init__.py",
    "src/lung_pde/forms/baseline.py",
    "src/lung_pde/forms/calibration.py",
    "src/lung_pde/forms/treatment.py",
    "src/lung_pde/forms/velocity.py",
    "src/lung_pde/initial_conditions.py",
    "src/lung_pde/mesh.py",
    "src/lung_pde/model.py",
    "src/lung_pde/parameters.py",
    "src/lung_pde/results.py",
    "src/lung_pde/simulation.py",
    "src/lung_pde/solver.py",
    "src/lung_pde/treatment.py",
    "src/lung_pde/validation.py",
    "src/lung_pde/velocity_fields.py",
)

CASE_JSON_FIELDS = (
    "schema_version",
    "case_key",
    "set_id",
    "eta",
    "danti_input",
    "danti_effective",
    "B0",
    "M0",
    "dB",
    "B_MUCUS",
    "B_TISSUE",
    "M_MUCUS",
    "M_TISSUE",
    "parameter_source",
    "parameter_source_kind",
    "parameter_source_sha256",
    "runtime_revision",
    "mesh_backend",
    "resolved_mesh_backend",
    "resolution",
    "dt",
    "final_time",
    "status",
)

TREATMENT_CSV_FIELDS = tuple(
    field for field in CASE_JSON_FIELDS if field != "schema_version"
)

RUN_IDENTITY_FIELDS = (
    "set_id",
    "parameter_source_sha256",
    "runtime_revision",
    "B0",
    "M0",
    "dB",
    "mesh_backend",
    "resolved_mesh_backend",
    "resolution",
    "dt",
    "final_time",
)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _finite_float(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _json_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    return _finite_float(value, name)


def _float_bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON property: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")


def _runtime_revision_repository_root() -> Path:
    module_path = Path(__file__).resolve()
    module_source = Path("src/lung_pde/treatment.py")
    for candidate in module_path.parents:
        if (candidate / module_source).resolve() == module_path:
            return candidate
    raise RuntimeError("unable to locate repository root for runtime revision")


def compute_runtime_revision(repository_root: str | Path | None = None) -> str:
    root = (
        _runtime_revision_repository_root()
        if repository_root is None
        else Path(repository_root)
    )
    digest = hashlib.sha256()
    digest.update(b"lung-pde-runtime-revision-v1\0")
    for relative_name in RUNTIME_REVISION_SOURCES:
        encoded_name = relative_name.encode("utf-8")
        try:
            content = (root / relative_name).read_bytes()
        except OSError as error:
            raise RuntimeError(
                f"runtime revision source is unavailable: {relative_name}"
            ) from error
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def resolve_requested_mesh_backend(name: str) -> str:
    if name not in {"auto", "legacy_mshr", "rectangle"}:
        raise ValueError(
            f"Unknown mesh backend {name!r}; expected auto, legacy_mshr, or rectangle"
        )
    if name != "auto":
        return name
    return "legacy_mshr" if importlib.util.find_spec("mshr") else "rectangle"


@dataclass(frozen=True)
class TreatmentGrid:
    eta_values: tuple[float, ...]
    positive_danti_multipliers: tuple[float, ...]
    include_zero_reference: bool
    zero_reference_eta: float


@dataclass(frozen=True)
class TreatmentCase:
    eta: float
    danti_input: float

    def __post_init__(self):
        eta = _finite_float(self.eta, "eta")
        danti = _finite_float(self.danti_input, "danti")
        if eta <= 0.0:
            raise ValueError("eta must be finite and positive")
        if danti < 0.0:
            raise ValueError("danti must be finite and non-negative")
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "danti_input", danti)

    @property
    def key(self) -> str:
        return f"eta_{_float_bits(self.eta)}__danti_{_float_bits(self.danti_input)}"

    def result_row(
        self,
        *,
        set_id: int,
        initial_bacteria: float,
        initial_macrophages: float,
        dB: float,
        outputs,
    ) -> dict:
        values = tuple(outputs)
        if len(values) != 4:
            raise ValueError("treatment outputs must contain four concentrations")
        B0 = _finite_float(initial_bacteria, "B0")
        M0 = _finite_float(initial_macrophages, "M0")
        current_dB = _finite_float(dB, "dB")
        concentrations = tuple(
            _finite_float(value, name)
            for value, name in zip(
                values,
                ("B_MUCUS", "B_TISSUE", "M_MUCUS", "M_TISSUE"),
            )
        )
        return {
            "case_key": self.key,
            "set_id": int(set_id),
            "eta": self.eta,
            "danti_input": self.danti_input,
            "danti_effective": self.danti_input * M0 * current_dB,
            "B0": B0,
            "M0": M0,
            "dB": current_dB,
            "B_MUCUS": concentrations[0],
            "B_TISSUE": concentrations[1],
            "M_MUCUS": concentrations[2],
            "M_TISSUE": concentrations[3],
        }


def _number_sequence(value, name: str, *, positive: bool) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(_finite_float(item, name) for item in value)
    if positive and any(item <= 0.0 for item in result):
        raise ValueError(f"{name} values must be positive")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    return result


def load_treatment_grid(path: str | Path) -> TreatmentGrid:
    source = Path(path)
    try:
        payload = yaml.load(
            source.read_text(encoding="utf-8"),
            Loader=_UniqueKeyLoader,
        )
    except OSError as error:
        raise FileNotFoundError(f"treatment grid does not exist: {source}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid treatment grid YAML: {source}") from error

    if not isinstance(payload, dict):
        raise ValueError("treatment grid must be a mapping")
    expected_fields = {
        "eta_values",
        "positive_danti_multipliers",
        "include_zero_reference",
        "zero_reference_eta",
    }
    missing = sorted(expected_fields - set(payload))
    extra = sorted(set(payload) - expected_fields)
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError("invalid treatment grid fields: " + "; ".join(details))

    include_zero_reference = payload["include_zero_reference"]
    if not isinstance(include_zero_reference, bool):
        raise ValueError("include_zero_reference must be boolean")
    zero_reference_eta = _finite_float(
        payload["zero_reference_eta"], "zero_reference_eta"
    )
    if zero_reference_eta <= 0.0:
        raise ValueError("zero_reference_eta must be positive")

    grid = TreatmentGrid(
        eta_values=_number_sequence(payload["eta_values"], "eta_values", positive=True),
        positive_danti_multipliers=_number_sequence(
            payload["positive_danti_multipliers"],
            "positive_danti_multipliers",
            positive=True,
        ),
        include_zero_reference=include_zero_reference,
        zero_reference_eta=zero_reference_eta,
    )
    list(iter_treatment_cases(grid))
    return grid


def iter_treatment_cases(grid: TreatmentGrid):
    seen = set()
    if grid.include_zero_reference:
        reference = TreatmentCase(grid.zero_reference_eta, 0.0)
        seen.add(reference.key)
        yield reference
    for eta in grid.eta_values:
        for danti in grid.positive_danti_multipliers:
            case = TreatmentCase(eta, danti)
            if case.key in seen:
                raise ValueError(f"treatment grid has colliding case key: {case.key}")
            seen.add(case.key)
            yield case


def case_output_path(root: str | Path, case: TreatmentCase) -> Path:
    return Path(root) / "cases" / f"{case.key}.json"


def case_lock_path(case_path: str | Path) -> Path:
    return Path(f"{Path(case_path)}.lock")


@contextmanager
def exclusive_case_lock(case_path: str | Path):
    lock_path = case_lock_path(case_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        os.fsync(lock_file.fileno())
        try:
            yield lock_path
        finally:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.flush()
            os.fsync(lock_file.fileno())
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _require_exact_fields(record: Mapping) -> None:
    missing = sorted(set(CASE_JSON_FIELDS) - set(record))
    extra = sorted(set(record) - set(CASE_JSON_FIELDS))
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError("invalid case fields: " + "; ".join(details))


def _validate_sha256(value, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_identity_values(record: Mapping, expected_identity: Mapping) -> None:
    unknown = sorted(set(expected_identity) - set(RUN_IDENTITY_FIELDS))
    if unknown:
        raise ValueError("unknown expected identity fields: " + ", ".join(unknown))
    mismatches = [
        name
        for name, expected in expected_identity.items()
        if record[name] != expected
    ]
    if mismatches:
        raise ValueError("case identity mismatch: " + ", ".join(mismatches))


def validate_case_result(
    record,
    *,
    expected_case: TreatmentCase | None = None,
    expected_identity: Mapping | None = None,
) -> dict:
    if not isinstance(record, dict):
        raise ValueError("case JSON must contain an object")
    _require_exact_fields(record)

    if (
        type(record["schema_version"]) is not int
        or record["schema_version"] != CASE_SCHEMA_VERSION
    ):
        raise ValueError("schema_version must be integer 2")
    if isinstance(record["set_id"], bool) or not isinstance(record["set_id"], int):
        raise ValueError("set_id must be an integer")
    if record["set_id"] <= 0:
        raise ValueError("set_id must be positive")

    case = TreatmentCase(
        _json_number(record["eta"], "eta"),
        _json_number(record["danti_input"], "danti_input"),
    )
    if record["case_key"] != case.key:
        raise ValueError("case identity does not match eta and danti_input")
    if expected_case is not None and case != expected_case:
        raise ValueError("case identity does not match the expected case")

    numeric_fields = (
        "danti_effective",
        "B0",
        "M0",
        "dB",
        "B_MUCUS",
        "B_TISSUE",
        "M_MUCUS",
        "M_TISSUE",
        "dt",
        "final_time",
    )
    numeric = {
        name: _json_number(record[name], name) for name in numeric_fields
    }
    if numeric["dt"] <= 0.0:
        raise ValueError("dt must be finite and positive")
    if numeric["final_time"] < 0.0:
        raise ValueError("final_time must be finite and non-negative")

    expected_effective = case.danti_input * numeric["M0"] * numeric["dB"]
    if not math.isclose(
        numeric["danti_effective"],
        expected_effective,
        rel_tol=1.0e-12,
        abs_tol=1.0e-18,
    ):
        raise ValueError("danti_effective does not match danti_input * M0 * dB")
    if record["status"] != CASE_STATUS_SUCCESS:
        raise ValueError("case JSON does not have successful status")
    if record["parameter_source_kind"] != "calibrated":
        raise ValueError("case parameter source must be calibrated")
    if not isinstance(record["parameter_source"], str) or not record[
        "parameter_source"
    ]:
        raise ValueError("parameter_source must be a non-empty string")
    _validate_sha256(
        record["parameter_source_sha256"], "parameter_source_sha256"
    )
    _validate_sha256(record["runtime_revision"], "runtime_revision")
    if record["mesh_backend"] not in {"auto", "legacy_mshr", "rectangle"}:
        raise ValueError("mesh_backend is invalid")
    if record["resolved_mesh_backend"] not in {"legacy_mshr", "rectangle"}:
        raise ValueError("resolved_mesh_backend is invalid")
    if (
        record["mesh_backend"] != "auto"
        and record["resolved_mesh_backend"] != record["mesh_backend"]
    ):
        raise ValueError(
            "resolved_mesh_backend must match an explicitly selected mesh_backend"
        )
    if isinstance(record["resolution"], bool) or not isinstance(
        record["resolution"], int
    ):
        raise ValueError("resolution must be an integer")
    if record["resolution"] <= 0:
        raise ValueError("resolution must be positive")

    if expected_identity is not None:
        _validate_identity_values(record, expected_identity)
    return record


def load_case_result(
    path: str | Path,
    *,
    expected_case: TreatmentCase | None = None,
    expected_identity: Mapping | None = None,
) -> dict:
    source = Path(path)
    try:
        record = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid case JSON {source}: {error}") from error
    return validate_case_result(
        record,
        expected_case=expected_case,
        expected_identity=expected_identity,
    )


def write_case_result(path: str | Path, record: Mapping) -> Path:
    destination = Path(path)
    validated = validate_case_result(dict(record))
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
            json.dump(validated, temporary, sort_keys=True, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _case_sort_key(record: Mapping):
    if record["danti_input"] == 0.0:
        return (0, 0.0, 0.0)
    return (1, record["eta"], record["danti_input"])


def _atomic_write_csv(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(
                temporary,
                fieldnames=TREATMENT_CSV_FIELDS,
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for record in rows:
                writer.writerow(
                    {field: record[field] for field in TREATMENT_CSV_FIELDS}
                )
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def aggregate_treatment_cases(
    case_dir: str | Path,
    output_csv: str | Path,
    *,
    expected_cases: Iterable[TreatmentCase] | None = None,
    expected_identity: Mapping | None = None,
) -> Path:
    directory = Path(case_dir)
    if expected_cases is None:
        expected_cases = iter_treatment_cases(
            load_treatment_grid(DEFAULT_TREATMENT_GRID)
        )
    expected_cases = tuple(expected_cases)
    expected_keys = {case.key for case in expected_cases}
    if len(expected_keys) != len(expected_cases):
        raise ValueError("expected treatment cases contain duplicate keys")

    by_key = {}
    source_paths = {}
    for source in sorted(directory.glob("*.json")):
        record = load_case_result(source)
        key = record["case_key"]
        if key in by_key:
            raise ValueError(
                "duplicate case identity "
                f"{key}: {source_paths[key].name}, {source.name}"
            )
        by_key[key] = record
        source_paths[key] = source

    actual_keys = set(by_key)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing case keys: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected case keys: " + ", ".join(unexpected))
        raise ValueError("invalid treatment case set; " + "; ".join(details))

    for key, source in source_paths.items():
        expected_name = f"{key}.json"
        if source.name != expected_name:
            raise ValueError(
                f"case filename mismatch: {source.name} must be {expected_name}"
            )

    ordered_records = [by_key[case.key] for case in expected_cases]
    if expected_identity is not None:
        for record in ordered_records:
            _validate_identity_values(record, expected_identity)
    if ordered_records:
        common_identity = {
            field: ordered_records[0][field] for field in RUN_IDENTITY_FIELDS
        }
        for record in ordered_records[1:]:
            _validate_identity_values(record, common_identity)

    ordered_records.sort(key=_case_sort_key)
    destination = Path(output_csv)
    _atomic_write_csv(destination, ordered_records)
    return destination
