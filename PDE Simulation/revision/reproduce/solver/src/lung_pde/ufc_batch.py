"""Assemble the unchanged FFC kernels with cached serial P1 scatter indices."""

from ctypes import CDLL, c_int, c_void_p
from pathlib import Path

import numpy as np
from dolfin import Constant, MPI, cells, facets
from dolfin.jit.jit import ffc_jit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu


class _Kernel:
    def __init__(self, owner, form, offsets):
        self.owner = owner
        ptr, self.module, _ = ffc_jit(form, mpi_comm=owner.mesh.mpi_comm())
        self.handle = owner.lib.create_assembly(ptr)
        ordered = form.coefficients()
        self.kinds = []
        self.constants = []
        for i in range(owner.lib.coefficient_count(self.handle)):
            coefficient = ordered[owner.lib.coefficient_position(self.handle, i)]
            if coefficient is owner.model.w:
                kind = 1
            elif coefficient is owner.model.w_n:
                kind = 2
            elif isinstance(coefficient, Constant) and coefficient.ufl_shape == ():
                kind = 0
            else:
                raise TypeError(f"unsupported UFC batch coefficient: {type(coefficient)}")
            self.kinds.append(kind)
            self.constants.append(coefficient if kind == 0 else None)
        self.kinds = np.array(self.kinds, dtype=np.intc)
        self.offsets = np.asarray(offsets, dtype=np.intc)

    def __del__(self):
        if hasattr(self, "handle"):
            self.owner.lib.destroy_assembly(self.handle)

    def run(self, state, previous, size):
        scalars = np.array([float(c) if c is not None else 0. for c in self.constants])
        result = np.zeros(size)
        owner = self.owner
        owner.lib.assemble_batch(
            self.handle, len(owner.tasks),
            *(a.ctypes.data for a in (owner.tasks, owner.coords, owner.dofs, self.kinds,
                                     scalars, state, previous, self.offsets, result))
        )
        return result


class BatchAssembler:
    """Supports two marked cell regions and the original P1 transmission facets."""

    def __init__(self, model, library):
        self.model = model
        self.update_state = getattr(model.velocity, "update_state", None)
        self.mesh = mesh = model.mesh.mesh
        if MPI.size(mesh.mpi_comm()) != 1 or mesh.topology().dim() != 2:
            raise ValueError("UFC batch requires a serial triangular mesh")
        self.lib = CDLL(str(Path(library).resolve()))
        self.lib.create_assembly.argtypes = [c_void_p]
        self.lib.create_assembly.restype = c_void_p
        self.lib.destroy_assembly.argtypes = [c_void_p]
        self.lib.destroy_assembly.restype = None
        self.lib.coefficient_count.argtypes = [c_void_p]
        self.lib.coefficient_position.argtypes = [c_void_p, c_int]
        self.lib.assemble_batch.argtypes = [c_void_p, c_int] + [c_void_p] * 9
        self.lib.assemble_batch.restype = None

        space = model.w.function_space()
        n = space.dim()
        active_mask = np.ones(n, dtype=bool)
        active_mask[model.fixed_dofs] = False
        self.active = np.flatnonzero(active_mask)
        ix = np.full(n, -1, dtype=np.intc)
        ix[self.active] = np.arange(len(self.active))
        dim = len(self.active)
        self.dofs = np.array([space.dofmap().cell_dofs(i) for i in range(mesh.num_cells())],
                             dtype=np.intc)
        if self.dofs.shape[1] != 15:
            raise ValueError("expected five mixed P1 fields")
        self.coords = np.array([c.get_vertex_coordinates() for c in cells(mesh)], dtype=float)
        mesh.init(1, 2)
        mesh.init(2, 1)
        cell_marks = model.mesh.cell_mark.array()
        if set(cell_marks) != {0, 1}:
            raise ValueError("unexpected UFC batch cell markers")
        facet_marks = model.mesh.boundary_mark.array()
        local = [list(c.entities(1)) for c in cells(mesh)]
        tasks = [(0, i, -1, 0, 0, int(flag)) for i, flag in enumerate(cell_marks)]
        task_dofs = list(ix[self.dofs])
        for f in facets(mesh):
            fi = f.index()
            flag = int(facet_marks[fi])
            if flag not in (1, 2, 4):
                continue
            near = f.entities(2)
            a = int(near[0])
            if flag == 1 and len(near) == 2:
                b = int(near[1])
                tasks.append((2, a, b, local[a].index(fi), local[b].index(fi), flag))
                task_dofs.append(ix[np.r_[self.dofs[a], self.dofs[b]]])
            elif flag in (2, 4) and len(near) == 1:
                tasks.append((1, a, -1, local[a].index(fi), 0, flag))
                task_dofs.append(ix[self.dofs[a]])
        self.tasks = np.array(tasks, dtype=np.intc)
        keys = []
        for dofs in task_dofs:
            valid = (dofs[:, None] >= 0) & (dofs[None, :] >= 0)
            key = dofs.astype(np.int64)[:, None] * dim + dofs[None, :]
            keys.append(np.where(valid, key, -1).ravel())
        scatter = np.concatenate(keys)
        valid = scatter >= 0
        indices, inverse = np.unique(scatter[valid], return_inverse=True)
        matrix_offsets = np.full(len(scatter), -1, dtype=np.intc)
        matrix_offsets[valid] = inverse
        self.indptr = np.r_[0, np.cumsum(np.bincount(indices // dim, minlength=dim))].astype(np.intc)
        self.indices = (indices % dim).astype(np.intc)
        self.residual = _Kernel(self, model.form.residual, np.concatenate(task_dofs))
        self.jacobian = _Kernel(self, model.form.jacobian, matrix_offsets)

    def f(self, state, previous):
        if self.update_state is not None and not self.update_state(state):
            return np.full(len(self.active), np.nan)
        return self.residual.run(state, previous, len(self.active))

    def j(self, state, previous):
        if self.update_state is not None and not self.update_state(state):
            raise RuntimeError("Invalid mucus mean viscosity in Jacobian.")
        # Keep the global viscosity scalar fixed in the local Jacobian.
        dim = len(self.active)
        values = self.jacobian.run(state, previous, len(self.indices))
        return csr_matrix((values, self.indices, self.indptr), shape=(dim, dim), copy=True)


class BatchNewton:
    """Use the same residual and Jacobian with the cached scatter assembly."""

    def __init__(self, model, library):
        self.model = model
        self.assembly = BatchAssembler(model, library)
        self.active = self.assembly.active
        self.fallbacks = 0
        self.factorizations = 0
        self.factor = None
        self.atol = float(model.solver.parameters["absolute_tolerance"])
        self.rtol = float(model.solver.parameters["relative_tolerance"])

    def step(self, time, previous, guess=None, *, lag=False):
        model = self.model
        model.velocity.t = float(time)
        now = previous.copy() if guess is None else guess.copy()
        residual = self.assembly.f(now, previous)
        norm = np.linalg.norm(residual)
        limit = max(self.atol, self.rtol * norm) if np.isfinite(norm) else self.atol
        if not lag:
            self.factor = None
        for _ in range(30):
            if not np.isfinite(norm):
                break
            if norm <= limit:
                return now
            fresh = self.factor is None
            if fresh:
                matrix = self.assembly.j(now, previous)
                matrix.eliminate_zeros()
                try:
                    self.factor = splu(matrix.tocsc())
                    self.factorizations += 1
                except RuntimeError:
                    break
            updated = now.copy()
            updated[self.active] -= self.factor.solve(residual)
            candidate = self.assembly.f(updated, previous)
            next_norm = np.linalg.norm(candidate)
            if next_norm <= limit or (np.isfinite(next_norm) and next_norm < (.7 if lag and not fresh else 1.) * norm):
                now, residual, norm = updated, candidate, next_norm
            elif fresh:
                break
            else:
                self.factor = None
            if not lag:
                self.factor = None
        self.factor = None
        self.fallbacks += 1
        now = self.damped_step(previous)
        if now is None:
            now = self.homotopy_step(previous)
        if now is not None:
            return now
        for f in (model.w, model.w_n):
            f.vector().set_local(previous)
            f.vector().apply("insert")
        try:
            model.step(time)
            return model.w.vector().get_local()
        except RuntimeError:
            from dolfin import NewtonSolver

            old = model.solver
            direct = NewtonSolver()
            direct.parameters["report"] = False
            direct.parameters["linear_solver"] = "lu"
            direct.parameters["relative_tolerance"] = self.rtol
            direct.parameters["absolute_tolerance"] = self.atol
            direct.parameters["maximum_iterations"] = 80
            direct.parameters["relaxation_parameter"] = .7
            model.solver = direct
            try:
                for f in (model.w, model.w_n):
                    f.vector().set_local(previous)
                    f.vector().apply("insert")
                model.step(time)
                return model.w.vector().get_local()
            finally:
                model.solver = old

    def damped_step(self, previous, guess=None):
        state = previous.copy() if guess is None else guess.copy()
        residual = self.assembly.f(state, previous)
        norm = np.linalg.norm(residual)
        limit = max(self.atol, self.rtol * norm) if np.isfinite(norm) else self.atol
        for _ in range(45):
            if not np.isfinite(norm):
                return None
            if norm <= limit:
                return state
            matrix = self.assembly.j(state, previous)
            matrix.eliminate_zeros()
            try:
                direction = splu(matrix.tocsc()).solve(residual)
            except RuntimeError:
                return None
            size = 1.
            for _ in range(10):
                trial = state.copy()
                trial[self.active] -= size * direction
                trial_residual = self.assembly.f(trial, previous)
                trial_norm = np.linalg.norm(trial_residual)
                if np.isfinite(trial_norm) and trial_norm < (1. - 1e-4 * size) * norm:
                    state, residual, norm = trial, trial_residual, trial_norm
                    break
                size *= .5
            else:
                return None
        return None

    def homotopy_step(self, previous):
        """Use auxiliary mass weights only to find the same target step root."""
        dt = float(self.model.dt)
        guess = previous
        try:
            for fraction in (.125, .25, .5, .75, 1.):
                self.model.dt.assign(dt * fraction)
                guess = self.damped_step(previous, guess)
                if guess is None:
                    return None
            return guess
        finally:
            self.model.dt.assign(dt)

    def endpoint(self, time, weights):
        self.model.reset()
        state = self.model.w.vector().get_local()
        for t in time[1:]:
            state = self.step(t, state)
        return weights @ state

    def slice_endpoints(self, time, weights, multipliers):
        self.model.reset()
        self.factor = None
        state = self.model.w.vector().get_local()
        states = np.repeat(state[None, :], len(multipliers), axis=0)
        for t in time[1:]:
            for j, multiplier in enumerate(multipliers):
                self.model.danti.assign(float(multiplier))
                states[j] = self.step(t, states[j], states[j - 1] if j else None, lag=True)
        return states @ weights.T
