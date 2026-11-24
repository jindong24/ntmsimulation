from dataclasses import dataclass

from dolfin import NonlinearProblem, NewtonSolver, assemble, cells

from .mesh import MUCUS_FLAG, TISSUE_FLAG


@dataclass(frozen=True)
class SolverSettings:
    relative_tolerance: float = 1.0e-6
    report: bool = False
    linear_solver: str | None = None
    preconditioner: str | None = None
    maximum_iterations: int | None = None


def compute_fixed_dofs(bundle):
    function_space = bundle.function_space
    dofmaps = [function_space.sub(index).dofmap() for index in range(5)]
    is_constrained = [True] * function_space.dim()

    for cell in cells(bundle.mesh):
        marker = bundle.cell_mark[cell]
        if marker == MUCUS_FLAG:
            enabled_components = (0, 2, 4)
        elif marker == TISSUE_FLAG:
            enabled_components = (1, 3)
        else:
            continue
        for component in enabled_components:
            for dof in dofmaps[component].cell_dofs(cell.index()):
                is_constrained[dof] = False

    return [dof for dof, constrained in enumerate(is_constrained) if constrained]


class ConstrainedNonlinearProblem(NonlinearProblem):
    def __init__(self, residual, jacobian, fixed_dofs, update_state=None):
        super().__init__()
        self.residual = residual
        self.jacobian = jacobian
        self.fixed_dofs = fixed_dofs
        self.update_state = update_state

    def F(self, vector, state):
        if self.update_state is not None and not self.update_state(state):
            raise RuntimeError("Invalid mucus mean viscosity in residual.")
        assemble(self.residual, tensor=vector)
        vector[self.fixed_dofs] = 0.0
        vector.apply("insert")

    def J(self, matrix, state):
        if self.update_state is not None and not self.update_state(state):
            raise RuntimeError("Invalid mucus mean viscosity in Jacobian.")
        assemble(self.jacobian, tensor=matrix, keep_diagonal=True)
        matrix.ident(self.fixed_dofs)
        matrix.apply("insert")


def make_newton_solver(settings: SolverSettings) -> NewtonSolver:
    solver = NewtonSolver()
    solver.parameters["relative_tolerance"] = settings.relative_tolerance
    solver.parameters["report"] = settings.report
    if settings.linear_solver is not None:
        solver.parameters["linear_solver"] = settings.linear_solver
    if settings.preconditioner is not None:
        solver.parameters["preconditioner"] = settings.preconditioner
    if settings.maximum_iterations is not None:
        solver.parameters["maximum_iterations"] = settings.maximum_iterations
    return solver
