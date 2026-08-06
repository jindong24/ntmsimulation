import pytest


pytest.importorskip("dolfin")
from dolfin import assemble

from lung_pde.model import build_model


@pytest.mark.fenics
@pytest.mark.parametrize("mode", ["calibration", "baseline", "velocity", "treatment"])
def test_each_protected_form_assembles(mode):
    model = build_model(set_id=756, mode=mode, resolution=4, mesh_backend="rectangle")
    assembled_terms = [assemble(term) for term in model.form.terms]
    residual = assemble(model.form.residual)
    jacobian = assemble(model.form.jacobian)

    assert len(assembled_terms) == 5
    assert all(term.size() == residual.size() for term in assembled_terms)
    assert residual.size() > 0
    assert jacobian.size(0) == residual.size()
    assert jacobian.size(1) == residual.size()
