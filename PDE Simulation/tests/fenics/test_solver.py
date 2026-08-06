import pytest

dolfin = pytest.importorskip("dolfin")

from lung_pde.mesh import MeshConfig, build_mesh
from lung_pde.solver import (
    ConstrainedNonlinearProblem,
    SolverSettings,
    compute_fixed_dofs,
    make_newton_solver,
)


@pytest.mark.fenics
def test_compute_fixed_dofs_constrains_components_outside_their_domains():
    bundle = build_mesh(MeshConfig(resolution=4, backend="rectangle"))
    fixed_dofs = set(compute_fixed_dofs(bundle))
    enabled_dofs = set()

    assert fixed_dofs
    for cell in dolfin.cells(bundle.mesh):
        marker = bundle.cell_mark[cell]
        enabled = (0, 2, 4) if marker == 0 else (1, 3)
        for component in enabled:
            dofs = bundle.function_space.sub(component).dofmap().cell_dofs(cell.index())
            enabled_dofs.update(dofs)

    assert fixed_dofs == set(range(bundle.function_space.dim())) - enabled_dofs


@pytest.mark.fenics
def test_constrained_problem_zeroes_residual_and_finalizes_identity_rows():
    bundle = build_mesh(MeshConfig(resolution=4, backend="rectangle"))
    state = dolfin.Function(bundle.function_space)
    components = dolfin.split(state)
    tests = dolfin.TestFunctions(bundle.function_space)
    residual = sum(
        (component - float(index + 1)) * test * bundle.dx
        for index, (component, test) in enumerate(zip(components, tests))
    )
    jacobian = dolfin.derivative(
        residual, state, dolfin.TrialFunction(bundle.function_space)
    )
    fixed_dofs = compute_fixed_dofs(bundle)
    problem = ConstrainedNonlinearProblem(residual, jacobian, fixed_dofs)
    fixed_dof_set = set(fixed_dofs)

    residual_vector = dolfin.assemble(residual)
    problem.F(residual_vector, state.vector())
    residual_values = residual_vector.get_local()
    assert fixed_dofs
    assert all(residual_values[dof] == 0.0 for dof in fixed_dofs)
    assert any(
        residual_values[dof] != 0.0
        for dof in range(bundle.function_space.dim())
        if dof not in fixed_dof_set
    )

    jacobian_matrix = dolfin.assemble(jacobian, keep_diagonal=True)
    problem.J(jacobian_matrix, state.vector())
    for dof in fixed_dofs:
        columns, values = jacobian_matrix.getrow(dof)
        row = dict(zip(columns, values))
        assert row[dof] == pytest.approx(1.0)
        assert all(
            value == pytest.approx(0.0)
            for column, value in row.items()
            if column != dof
        )


@pytest.mark.fenics
def test_newton_solver_applies_only_requested_linear_settings():
    default_solver = make_newton_solver(SolverSettings())
    source_default = dolfin.NewtonSolver().parameters["linear_solver"]
    assert default_solver.parameters["linear_solver"] == source_default
    assert default_solver.parameters["relative_tolerance"] == pytest.approx(1.0e-6)
    assert default_solver.parameters["report"] is False

    gmres_solver = make_newton_solver(
        SolverSettings(linear_solver="gmres", preconditioner="ilu")
    )
    assert gmres_solver.parameters["linear_solver"] == "gmres"
    assert gmres_solver.parameters["preconditioner"] == "ilu"

    extended_solver = make_newton_solver(SolverSettings(maximum_iterations=100))
    assert extended_solver.parameters["maximum_iterations"] == 100
