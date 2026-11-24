import os
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

dolfin = pytest.importorskip("dolfin")

from lung_pde.model import build_model
from lung_pde.ufc_batch import BatchAssembler


@pytest.mark.fenics
def test_cached_scatter_matches_dolfin_on_two_states():
    if "NTM_ORIGINAL_MESH" not in os.environ or "NTM_UFC_LIBRARY" not in os.environ:
        pytest.skip("original mesh and compiled UFC library are required")
    model = build_model(
        756, "treatment", mesh_backend="original_mshr",
        original_mesh=Path(os.environ["NTM_ORIGINAL_MESH"]), conservative_exchange=True,
        polynomial_velocity=True,
    )
    model.assign_parameters(dict(tauB=.015, tauM=.025, delta=.006))
    model.eta.assign(.5)
    model.danti.assign(2.5)
    owner = BatchAssembler(model, os.environ["NTM_UFC_LIBRARY"])
    a = owner.active
    state = model.w.vector().get_local()
    previous = model.w_n.vector().get_local()
    rng = np.random.default_rng(48)
    for t in (.11, 1.37):
        model.velocity.t = t
        state[a] += 1e-8 * rng.random(len(a))
        previous[a] += 1e-9 * rng.random(len(a))
        for f, x in ((model.w, state), (model.w_n, previous)):
            f.vector().set_local(x)
            f.vector().apply("insert")
        expected = dolfin.assemble(model.form.residual).get_local()[a]
        np.testing.assert_allclose(owner.f(state, previous), expected, rtol=1e-10, atol=1e-15)
        standard = dolfin.as_backend_type(dolfin.assemble(model.form.jacobian)).mat()
        ip, ix, data = standard.getValuesCSR()
        expected = csr_matrix((data, ix, ip), shape=standard.getSize())[a, :][:, a]
        actual = owner.j(state, previous)
        actual.eliminate_zeros()
        np.testing.assert_allclose((actual - expected).data, 0., rtol=0., atol=1e-13)
