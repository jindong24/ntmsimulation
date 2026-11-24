from dataclasses import dataclass, replace
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from .constants import REQUIRED_PARAMETER_KEYS
from .data import validate_set_id


LEGACY_PARAMETER_KEYS = (
    "dt",
    "T",
    "lambdaB1",
    "lambdaB2",
    "dm1",
    "dm2",
    "loss",
    "Kbi",
)
_ALLOWED_PARAMETER_KEYS = set(REQUIRED_PARAMETER_KEYS + LEGACY_PARAMETER_KEYS)


@dataclass(frozen=True)
class ParameterSet:
    values: dict[str, float]

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def updated(self, updates: dict[str, float]) -> "ParameterSet":
        return replace(self, values={**self.values, **updates})


def read_parameter_xml(path: str | Path) -> ParameterSet:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise ValueError("invalid parameter XML") from error

    if root.tag == "dolfin":
        if root.attrib:
            raise ValueError("parameter XML dolfin root must not have attributes")
        children = list(root)
        if len(children) != 1 or children[0].tag != "parameters":
            raise ValueError(
                "parameter XML dolfin root must contain exactly one parameters node"
            )
        container = children[0]
    elif root.tag == "parameters":
        container = root
    else:
        raise ValueError("parameter XML root must be dolfin or parameters")

    if set(container.attrib) != {"name"}:
        raise ValueError("parameters attributes must contain exactly name")
    if container.attrib["name"] != "problem_parameters":
        raise ValueError("parameters name must be problem_parameters")

    values = {}
    for node in list(container):
        if node.tag != "parameter" or list(node):
            raise ValueError("parameter XML contains an unexpected child node")
        if node.text is not None and node.text.strip():
            raise ValueError("parameter node contains unexpected text")
        if set(node.attrib) != {"key", "type", "value"}:
            raise ValueError(
                "parameter attributes must contain exactly key, type, and value"
            )
        key = node.attrib["key"]
        if key in values:
            raise ValueError(f"duplicate parameter key: {key}")
        if key not in _ALLOWED_PARAMETER_KEYS:
            raise ValueError(f"unexpected parameter key: {key}")
        if node.attrib["type"] != "double":
            raise ValueError("parameter type must be double")
        try:
            value = float(node.attrib["value"].replace("\u2212", "-"))
        except ValueError as error:
            raise ValueError(f"parameter {key} must be a finite number") from error
        if not math.isfinite(value):
            raise ValueError(f"parameter {key} must be finite")
        values[key] = value

    missing = sorted(set(REQUIRED_PARAMETER_KEYS) - set(values))
    if missing:
        raise ValueError(f"missing required parameters: {', '.join(missing)}")
    return ParameterSet(values)


def write_parameter_xml(parameters: ParameterSet, path: str | Path) -> None:
    destination = Path(path)
    root = ET.Element("parameters", {"name": "problem_parameters"})
    for key, value in parameters.values.items():
        ET.SubElement(root, "parameter", {
            "key": key,
            "type": "double",
            "value": repr(float(value)),
        })

    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        ET.ElementTree(root).write(temporary, encoding="UTF-8", xml_declaration=True)

    try:
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def parameter_path(root: str | Path, set_id: int, calibrated: bool) -> Path:
    input_kind = "calibrated" if calibrated else "initial"
    return (
        Path(root)
        / "data"
        / "parameters"
        / input_kind
        / f"parameters_{validate_set_id(set_id)}.xml"
    )
