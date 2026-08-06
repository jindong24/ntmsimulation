import pytest

dolfin = pytest.importorskip("dolfin")

from lung_pde.initial_conditions import initialize_state
from lung_pde.mesh import MeshConfig, build_mesh


@pytest.mark.fenics
def test_initial_mucus_integrals_equal_abm_initial_values():
    bundle = build_mesh(MeshConfig(resolution=8, backend="rectangle"))
    state, previous = initialize_state(bundle, 1.8865e-6, 0.0012498)
    bacteria = 50.0 * dolfin.assemble(state.sub(0) * bundle.dx(0))
    macrophages = 50.0 * dolfin.assemble(state.sub(2) * bundle.dx(0))
    assert bacteria == pytest.approx(1.8865e-6, rel=2e-2)
    assert macrophages == pytest.approx(0.0012498, rel=2e-2)
    assert state.vector().get_local().tolist() == previous.vector().get_local().tolist()


@pytest.mark.fenics
def test_inactive_components_are_initialized_to_legacy_values():
    bundle = build_mesh(MeshConfig(resolution=4, backend="rectangle"))
    state, _ = initialize_state(bundle, 1.8865e-6, 0.0012498)

    bacteria_tissue = state.sub(1, deepcopy=True).vector().get_local()
    macrophages_tissue = state.sub(3, deepcopy=True).vector().get_local()
    mucus = state.sub(4, deepcopy=True).vector().get_local()
    assert max(abs(bacteria_tissue)) == 0.0
    assert max(abs(macrophages_tissue)) == 0.0
    assert max(mucus) == pytest.approx(1.0e-3)
