from copy import deepcopy
import xml.etree.ElementTree as ET

import pytest

from lung_pde.constants import CALIBRATED_KEYS, FIXED_KEYS, REQUIRED_PARAMETER_KEYS
from lung_pde.parameters import (
    LEGACY_PARAMETER_KEYS,
    parameter_path,
    read_parameter_xml,
    write_parameter_xml,
)


def test_unicode_minus_and_all_required_keys_are_read():
    params = read_parameter_xml("data/parameters/initial/parameters_756.xml")
    assert params["DB2"] == 8.64e-7
    assert set(CALIBRATED_KEYS + FIXED_KEYS) <= set(params.values)


def test_xml_round_trip_preserves_values_and_extra_metadata(tmp_path):
    source = read_parameter_xml("data/parameters/calibrated/parameters_756.xml")
    output = tmp_path / "parameters.xml"
    write_parameter_xml(source, output)
    restored = read_parameter_xml(output)
    assert restored.values == source.values
    assert "loss" in restored.values


def test_parameter_path_selects_requested_input_kind(tmp_path):
    assert parameter_path(tmp_path, 756, calibrated=False) == (
        tmp_path / "data" / "parameters" / "initial" / "parameters_756.xml"
    )
    assert parameter_path(tmp_path, 756, calibrated=True) == (
        tmp_path / "data" / "parameters" / "calibrated" / "parameters_756.xml"
    )


def _temporary_tree(tmp_path):
    tree = ET.parse("data/parameters/calibrated/parameters_756.xml")
    destination = tmp_path / "parameters.xml"
    return tree, destination


def _parameter_container(tree):
    root = tree.getroot()
    return root if root.tag == "parameters" else root.find("parameters")


def test_supplied_legacy_extra_parameter_keys_are_explicitly_supported():
    parameters = read_parameter_xml(
        "data/parameters/calibrated/parameters_756.xml"
    )

    assert set(parameters.values) - set(REQUIRED_PARAMETER_KEYS) == set(
        LEGACY_PARAMETER_KEYS
    )


def test_parameter_xml_rejects_duplicate_keys(tmp_path):
    tree, destination = _temporary_tree(tmp_path)
    container = _parameter_container(tree)
    container.append(deepcopy(container.find("parameter")))
    tree.write(destination, encoding="UTF-8", xml_declaration=True)

    with pytest.raises(ValueError, match="duplicate parameter key: dt"):
        read_parameter_xml(destination)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_parameter_xml_rejects_nonfinite_values(tmp_path, value):
    tree, destination = _temporary_tree(tmp_path)
    container = _parameter_container(tree)
    container.find("parameter[@key='DB1']").set("value", value)
    tree.write(destination, encoding="UTF-8", xml_declaration=True)

    with pytest.raises(ValueError, match="parameter DB1 must be finite"):
        read_parameter_xml(destination)


@pytest.mark.parametrize(
    "mutation",
    ["wrong_root", "unexpected_child", "nested_parameter"],
)
def test_parameter_xml_rejects_unexpected_hierarchy(tmp_path, mutation):
    tree, destination = _temporary_tree(tmp_path)
    root = tree.getroot()
    container = _parameter_container(tree)
    if mutation == "wrong_root":
        root.tag = "unexpected"
    elif mutation == "unexpected_child":
        ET.SubElement(container, "metadata")
    else:
        wrapper = ET.SubElement(container, "group")
        wrapper.append(deepcopy(container.find("parameter")))
    tree.write(destination, encoding="UTF-8", xml_declaration=True)

    with pytest.raises(ValueError, match="parameter XML"):
        read_parameter_xml(destination)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_parameter_attribute", "parameter attributes"),
        ("missing_parameter_attribute", "parameter attributes"),
        ("wrong_parameter_type", "parameter type must be double"),
        ("parameter_text", "parameter node contains unexpected text"),
        ("extra_container_attribute", "parameters attributes"),
        ("wrong_container_name", "parameters name must be problem_parameters"),
    ],
)
def test_parameter_xml_rejects_unexpected_attributes(
    tmp_path, mutation, message
):
    tree, destination = _temporary_tree(tmp_path)
    container = _parameter_container(tree)
    parameter = container.find("parameter")
    if mutation == "extra_parameter_attribute":
        parameter.set("unit", "unknown")
    elif mutation == "missing_parameter_attribute":
        parameter.attrib.pop("type")
    elif mutation == "wrong_parameter_type":
        parameter.set("type", "float")
    elif mutation == "parameter_text":
        parameter.text = "unexpected"
    elif mutation == "extra_container_attribute":
        container.set("version", "1")
    else:
        container.set("name", "other_parameters")
    tree.write(destination, encoding="UTF-8", xml_declaration=True)

    with pytest.raises(ValueError, match=message):
        read_parameter_xml(destination)


def test_parameter_xml_rejects_unknown_legacy_key(tmp_path):
    tree, destination = _temporary_tree(tmp_path)
    container = _parameter_container(tree)
    parameter = deepcopy(container.find("parameter"))
    parameter.set("key", "undocumented_legacy_key")
    container.append(parameter)
    tree.write(destination, encoding="UTF-8", xml_declaration=True)

    with pytest.raises(ValueError, match="unexpected parameter key"):
        read_parameter_xml(destination)


def test_parameter_xml_rejects_malformed_xml_with_stable_error(tmp_path):
    destination = tmp_path / "malformed.xml"
    destination.write_text("<dolfin><parameters>", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid parameter XML"):
        read_parameter_xml(destination)
