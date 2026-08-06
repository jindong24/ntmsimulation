import numpy as np
import pytest

from lung_pde.data import load_abm, validate_set_id


SOURCE = "data/abm/data_756.npz"


def _source_arrays():
    with np.load(SOURCE, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def test_abm_contract_splits_initial_value_from_83_targets():
    data = load_abm("data/abm/data_756.npz")
    assert data.times.shape == (83,)
    assert data.bacteria.shape == (83,)
    assert data.macrophages.shape == (83,)
    assert data.initial_bacteria > 0
    assert data.initial_macrophages > 0
    np.testing.assert_allclose(np.diff(data.times), 1.0 / 6.0)


def test_invalid_set_is_rejected():
    with pytest.raises(ValueError, match="supported parameter set"):
        validate_set_id(999)


def test_abm_rejects_shifted_time_grid_even_when_spacing_is_one_sixth(tmp_path):
    arrays = _source_arrays()
    arrays["t"] = arrays["t"] + 1.0 / 6.0
    destination = tmp_path / "shifted.npz"
    np.savez(destination, **arrays)

    with pytest.raises(ValueError, match="target times must equal"):
        load_abm(destination)


@pytest.mark.parametrize("field", ["B", "M", "t"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf], ids=("nan", "infinity"))
def test_abm_rejects_nonfinite_values_in_every_array(tmp_path, field, invalid):
    arrays = _source_arrays()
    arrays[field] = arrays[field].copy()
    arrays[field][0] = invalid
    destination = tmp_path / f"nonfinite-{field}.npz"
    np.savez(destination, **arrays)

    with pytest.raises(ValueError, match=f"ABM {field} must contain finite real numbers"):
        load_abm(destination)


@pytest.mark.parametrize("field", ["B", "M", "t"])
@pytest.mark.parametrize(
    "invalid_kind",
    ["complex", "string", "boolean"],
)
def test_abm_rejects_nonreal_or_nonnumeric_array_types(
    tmp_path, field, invalid_kind
):
    arrays = _source_arrays()
    if invalid_kind == "complex":
        arrays[field] = arrays[field].astype(complex) + 1.0j
    elif invalid_kind == "string":
        arrays[field] = arrays[field].astype(str)
    else:
        arrays[field] = np.ones(arrays[field].shape, dtype=bool)
    destination = tmp_path / f"{invalid_kind}-{field}.npz"
    np.savez(destination, **arrays)

    with pytest.raises(ValueError, match=f"ABM {field} must contain finite real numbers"):
        load_abm(destination)
