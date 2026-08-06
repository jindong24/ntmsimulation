import pytest

dolfin = pytest.importorskip("dolfin")

import lung_pde.mesh as mesh_module
from lung_pde.mesh import (
    INTERFACE_FLAG,
    MUCUS_FLAG,
    RIGHT_MUCUS_FLAG,
    TISSUE_FLAG,
    MeshConfig,
    build_mesh,
    resolve_mesh_backend,
)


@pytest.mark.fenics
def test_rectangle_mesh_marks_mucus_tissue_and_interface():
    bundle = build_mesh(
        MeshConfig(resolution=4, backend="rectangle", treatment_boundary=True)
    )
    assert bundle.backend == "rectangle"
    assert set(bundle.cell_mark.array()) == {0, 1}
    assert 1 in set(bundle.boundary_mark.array())
    assert 2 in set(bundle.boundary_mark.array())
    assert 3 in set(bundle.boundary_mark.array())
    assert 4 in set(bundle.boundary_mark.array())
    assert bundle.function_space.num_sub_spaces() == 5


@pytest.mark.fenics
def test_rectangle_backend_rejects_nonconforming_interface():
    with pytest.raises(ValueError, match="align with a mesh row"):
        build_mesh(
            MeshConfig(resolution=4, backend="rectangle", interface_y=0.09)
        )


@pytest.mark.fenics
def test_interface_marks_exactly_all_facets_separating_cell_domains():
    bundle = build_mesh(MeshConfig(resolution=4, backend="rectangle"))
    dimension = bundle.mesh.topology().dim()
    bundle.mesh.init(dimension - 1, dimension)
    separating_facets = []
    marked_facets = []

    for facet in dolfin.facets(bundle.mesh):
        adjacent_cells = facet.entities(dimension)
        if len(adjacent_cells) == 2:
            markers = {bundle.cell_mark[cell] for cell in adjacent_cells}
            if markers == {MUCUS_FLAG, TISSUE_FLAG}:
                separating_facets.append(facet.index())
        if bundle.boundary_mark[facet] == INTERFACE_FLAG:
            marked_facets.append(facet.index())

    assert marked_facets
    assert set(marked_facets) == set(separating_facets)
    assert dolfin.assemble(dolfin.Constant(1.0) * bundle.dS(INTERFACE_FLAG)) == (
        pytest.approx(0.2)
    )


@pytest.mark.fenics
def test_treatment_marker_is_absent_when_disabled():
    bundle = build_mesh(
        MeshConfig(resolution=4, backend="rectangle", treatment_boundary=False)
    )
    assert RIGHT_MUCUS_FLAG not in set(bundle.boundary_mark.array())
    assert dolfin.assemble(dolfin.Constant(1.0) * bundle.ds(RIGHT_MUCUS_FLAG)) == 0.0


@pytest.mark.fenics
def test_treatment_marker_covers_only_the_right_mucus_boundary():
    config = MeshConfig(resolution=4, backend="rectangle", treatment_boundary=True)
    bundle = build_mesh(config)
    marked_facets = [
        facet
        for facet in dolfin.facets(bundle.mesh)
        if bundle.boundary_mark[facet] == RIGHT_MUCUS_FLAG
    ]

    assert marked_facets
    assert all(facet.exterior() for facet in marked_facets)
    assert all(
        dolfin.near(facet.midpoint().x(), config.width) for facet in marked_facets
    )
    assert all(facet.midpoint().y() > config.interface_y for facet in marked_facets)
    assert dolfin.assemble(dolfin.Constant(1.0) * bundle.ds(RIGHT_MUCUS_FLAG)) == (
        pytest.approx(config.height - config.interface_y)
    )


@pytest.mark.fenics
def test_submesh_spaces_and_measures_cover_the_domain():
    config = MeshConfig(resolution=4, backend="rectangle")
    bundle = build_mesh(config)
    mucus_cells = sum(
        bundle.cell_mark[cell] == MUCUS_FLAG for cell in dolfin.cells(bundle.mesh)
    )
    tissue_cells = sum(
        bundle.cell_mark[cell] == TISSUE_FLAG for cell in dolfin.cells(bundle.mesh)
    )

    assert bundle.mucus_mesh.num_cells() == mucus_cells
    assert bundle.tissue_mesh.num_cells() == tissue_cells
    assert bundle.mucus_space.ufl_element().family() == "Lagrange"
    assert bundle.tissue_space.ufl_element().family() == "Lagrange"
    assert bundle.mucus_space.ufl_element().degree() == 1
    assert bundle.tissue_space.ufl_element().degree() == 1
    mucus_area = dolfin.assemble(dolfin.Constant(1.0) * bundle.dx(MUCUS_FLAG))
    tissue_area = dolfin.assemble(dolfin.Constant(1.0) * bundle.dx(TISSUE_FLAG))
    assert mucus_area == pytest.approx(
        config.width * (config.height - config.interface_y)
    )
    assert tissue_area == pytest.approx(config.width * config.interface_y)
    assert mucus_area + tissue_area == pytest.approx(config.width * config.height)


@pytest.mark.fenics
def test_auto_fallback_is_recorded_on_bundle(monkeypatch):
    def unavailable(name):
        assert name == "mshr"
        raise ModuleNotFoundError("No module named 'mshr'")

    monkeypatch.setattr(mesh_module, "import_module", unavailable)
    bundle = build_mesh(MeshConfig(resolution=4, backend="auto"))
    assert bundle.backend == "rectangle"


@pytest.mark.fenics
def test_explicit_unavailable_mshr_has_installation_error(monkeypatch):
    def unavailable(name):
        assert name == "mshr"
        raise ModuleNotFoundError("No module named 'mshr'")

    monkeypatch.setattr(mesh_module, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="requires the optional mshr package"):
        resolve_mesh_backend("legacy_mshr")


@pytest.mark.fenics
def test_legacy_mshr_backend_is_explicitly_available_or_skipped():
    try:
        pytest.importorskip("mshr", reason="mshr unavailable")
    except ImportError:
        pytest.skip("mshr unavailable")

    bundle = build_mesh(MeshConfig(resolution=4, backend="legacy_mshr"))
    assert bundle.backend == "legacy_mshr"
    assert set(bundle.cell_mark.array()) == {0, 1}


@pytest.mark.fenics
def test_unknown_mesh_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown mesh backend"):
        resolve_mesh_backend("triangles")
