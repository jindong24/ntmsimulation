from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from dolfin import (
    Constant,
    DOLFIN_EPS,
    FacetNormal,
    FiniteElement,
    FunctionSpace,
    HDF5File,
    Measure,
    Mesh,
    MeshFunction,
    MPI,
    MixedElement,
    Point,
    RectangleMesh,
    SubDomain,
    SubMesh,
    cells,
    near,
)


MUCUS_FLAG = 0
TISSUE_FLAG = 1
INTERFACE_FLAG = 1
LOWER_FLAG = 2
UPPER_FLAG = 3
RIGHT_MUCUS_FLAG = 4


@dataclass(frozen=True)
class MeshConfig:
    resolution: int = 32
    backend: str = "auto"
    treatment_boundary: bool = False
    width: float = 0.2
    height: float = 0.2
    interface_y: float = 0.1
    original_mesh: Path | None = None


@dataclass
class MeshBundle:
    mesh: Any
    cell_mark: Any
    boundary_mark: Any
    dx: Any
    ds: Any
    dS: Any
    normal: Any
    down_normal: Any
    function_space: Any
    mucus_mesh: Any
    tissue_mesh: Any
    mucus_space: Any
    tissue_space: Any
    backend: str


def resolve_mesh_backend(name: str) -> str:
    if name not in {"auto", "legacy_mshr", "original_mshr", "rectangle"}:
        raise ValueError(
            f"Unknown mesh backend {name!r}; expected auto, legacy_mshr, original_mshr, or rectangle"
        )

    if name in {"rectangle", "original_mshr"}:
        return name

    try:
        import_module("mshr")
    except ImportError as exc:
        if name == "legacy_mshr":
            raise RuntimeError(
                "The legacy_mshr backend requires the optional mshr package; "
                "install mshr or select backend='rectangle'."
            ) from exc
        return "rectangle"
    return "legacy_mshr"


class _LowerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] < DOLFIN_EPS


class _UpperBoundary(SubDomain):
    def __init__(self, height):
        super().__init__()
        self.height = height

    def inside(self, x, on_boundary):
        return on_boundary and x[1] > self.height - DOLFIN_EPS


class _Interface(SubDomain):
    def __init__(self, interface_y):
        super().__init__()
        self.interface_y = interface_y

    def inside(self, x, on_boundary):
        return near(x[1], self.interface_y, DOLFIN_EPS)


class _RightMucusBoundary(SubDomain):
    def __init__(self, width, height, interface_y):
        super().__init__()
        self.width = width
        self.height = height
        self.interface_y = interface_y

    def inside(self, x, on_boundary):
        return (
            on_boundary
            and x[1] > self.interface_y - DOLFIN_EPS
            and x[1] < self.height + DOLFIN_EPS
            and x[0] > self.width - DOLFIN_EPS
        )


def _build_legacy_mshr_mesh(config):
    mshr = import_module("mshr")
    domain = mshr.Rectangle(Point(0, 0), Point(config.width, config.height))
    tissue = mshr.Rectangle(
        Point(0, 0), Point(config.width, config.interface_y)
    )
    domain.set_subdomain(TISSUE_FLAG, tissue)
    mesh = mshr.generate_mesh(domain, config.resolution)
    dimension = mesh.geometry().dim()
    cell_mark = MeshFunction("size_t", mesh, dimension, mesh.domains())
    return mesh, cell_mark


def _build_rectangle_mesh(config):
    interface_row = config.interface_y / config.height * config.resolution
    aligned_interface_y = config.height * round(interface_row) / config.resolution
    if not near(config.interface_y, aligned_interface_y, DOLFIN_EPS):
        raise ValueError(
            "Rectangle backend interface_y must align with a mesh row; "
            "adjust interface_y or resolution."
        )
    mesh = RectangleMesh(
        Point(0, 0),
        Point(config.width, config.height),
        config.resolution,
        config.resolution,
    )
    dimension = mesh.geometry().dim()
    cell_mark = MeshFunction("size_t", mesh, dimension, MUCUS_FLAG)
    for cell in cells(mesh):
        if cell.midpoint().y() < config.interface_y:
            cell_mark[cell] = TISSUE_FLAG
    return mesh, cell_mark


def _load_original_mesh(config):
    if config.original_mesh is None or config.resolution != 32:
        raise ValueError("original_mshr requires the archived mesh file and resolution 32")
    mesh = Mesh()
    reader = HDF5File(MPI.comm_world, str(config.original_mesh), "r")
    try:
        reader.read(mesh, "/mesh", False)
        mark = MeshFunction("size_t", mesh, mesh.topology().dim())
        reader.read(mark, "/cell_mark")
    finally:
        reader.close()
    if (mesh.num_vertices(), mesh.num_cells()) != (1643, 3142):
        raise ValueError("unexpected geometry in the original mshr mesh")
    return mesh, mark


def build_mesh(config: MeshConfig) -> MeshBundle:
    if config.resolution <= 0:
        raise ValueError("Mesh resolution must be positive")
    if not 0.0 < config.interface_y < config.height:
        raise ValueError("Mesh interface_y must lie strictly inside the domain")
    if config.width <= 0.0 or config.height <= 0.0:
        raise ValueError("Mesh width and height must be positive")

    backend = resolve_mesh_backend(config.backend)
    if backend == "original_mshr":
        mesh, cell_mark = _load_original_mesh(config)
    elif backend == "legacy_mshr":
        mesh, cell_mark = _build_legacy_mshr_mesh(config)
    else:
        mesh, cell_mark = _build_rectangle_mesh(config)

    dimension = mesh.geometry().dim()
    boundary_mark = MeshFunction("size_t", mesh, dimension - 1, 0)
    _UpperBoundary(config.height).mark(boundary_mark, UPPER_FLAG)
    _LowerBoundary().mark(boundary_mark, LOWER_FLAG)
    _Interface(config.interface_y).mark(boundary_mark, INTERFACE_FLAG)
    if config.treatment_boundary:
        _RightMucusBoundary(
            config.width, config.height, config.interface_y
        ).mark(boundary_mark, RIGHT_MUCUS_FLAG)

    mucus_mesh = SubMesh(mesh, cell_mark, MUCUS_FLAG)
    tissue_mesh = SubMesh(mesh, cell_mark, TISSUE_FLAG)
    mucus_space = FunctionSpace(mucus_mesh, "CG", 1)
    tissue_space = FunctionSpace(tissue_mesh, "CG", 1)

    element = FiniteElement("CG", mesh.ufl_cell(), 1)
    mixed_element = MixedElement((element, element, element, element, element))
    function_space = FunctionSpace(mesh, mixed_element)

    metadata = {"quadrature_degree": 3}
    dx = Measure("dx", domain=mesh, subdomain_data=cell_mark, metadata=metadata)
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_mark, metadata=metadata)
    dS = Measure("dS", domain=mesh, subdomain_data=boundary_mark, metadata=metadata)

    return MeshBundle(
        mesh=mesh,
        cell_mark=cell_mark,
        boundary_mark=boundary_mark,
        dx=dx,
        ds=ds,
        dS=dS,
        normal=FacetNormal(mesh),
        down_normal=Constant(("0.", "-1")),
        function_space=function_space,
        mucus_mesh=mucus_mesh,
        tissue_mesh=tissue_mesh,
        mucus_space=mucus_space,
        tissue_space=tissue_space,
        backend=backend,
    )
