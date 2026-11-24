"""Opt-in discrete tangent for the serial DOLFIN calibration model."""

import numpy as np
from dolfin import Constant, Form, Function, MPI, PETScMatrix, PETScVector, TrialFunction, assemble
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from ufl import as_ufl, derivative
from ufl.algorithms import expand_derivatives


def _assemble_csr(form, tensor):
    assemble(form, tensor=tensor)
    mat = tensor.mat()
    indptr, indices, data = mat.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=mat.getSize())


class CalibrationTangent:
    """Differentiate converged backward Euler steps, not Newton iterations.

    ``names`` sets optimizer column order. ``times`` and ``weights`` match
    calibrate_conservative.predict; one model step is taken per time entry.
    The model must have fixed dt and parameter-independent initial state and
    constrained DOF values. The previous state must enter only through the
    backward Euler mass term. Use one instance per model and training grid.
    """

    def __init__(self, model, names, times, weights):
        if model.config.mode != "calibration":
            raise ValueError("discrete tangents are only supported in calibration mode")
        space = model.w.function_space()
        comm = space.mesh().mpi_comm()
        if MPI.size(comm) != 1:
            raise ValueError("calibration tangents require a single MPI rank")

        self.names = tuple(names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("parameter names must be nonempty and unique")
        self._log = np.array([name != "alpha1" for name in self.names])
        self._model = model
        self._times = np.array(times, dtype=float, copy=True)
        if self._times.ndim != 1 or not self._times.size or not np.isfinite(self._times).all():
            raise ValueError("times must be a nonempty finite vector")
        self._weights = np.array(weights, dtype=float, copy=True)
        if (self._weights.ndim != 2 or self._weights.shape[1] != space.dim()
                or not np.isfinite(self._weights).all()):
            raise ValueError("weights must be finite with one column per state DOF")

        active = np.ones(space.dim(), dtype=bool)
        fixed = np.asarray(model.fixed_dofs, dtype=np.intp)
        if np.any((fixed < 0) | (fixed >= active.size)):
            raise ValueError("fixed DOF indices are outside the state space")
        active[fixed] = False
        self._active = np.flatnonzero(active)
        if not self._active.size:
            raise ValueError("the calibration model has no active DOFs")
        self._active_weights = self._weights[:, self._active]

        residual = model.form.residual
        self._partials = []
        for name in self.names:
            constant = model.parameters[name]
            if not isinstance(constant, Constant) or constant.ufl_shape != ():
                raise TypeError(f"parameter {name} must be a scalar DOLFIN Constant")
            partial = expand_derivatives(derivative(residual, constant, as_ufl(1.0)))
            if not partial.integrals():
                raise ValueError(f"parameter {name} is inactive in this calibration residual")
            self._partials.append(Form(partial))

        self._dt = float(model.dt)
        self._jacobian = Form(model.form.jacobian)
        history = Form(expand_derivatives(derivative(residual, model.w_n, TrialFunction(space))))
        history_matrix = _assemble_csr(history, PETScMatrix(comm))
        self._mass_over_dt = -self._restrict(history_matrix)
        self._matrix = PETScMatrix(comm)
        self._vectors = [PETScVector(comm) for _ in self.names]
        self._previous = Function(space)

    def _restrict(self, matrix):
        return matrix[self._active, :][:, self._active]

    def _physical(self, vector):
        physical = np.array(vector, dtype=float, copy=True)
        if physical.shape != (len(self.names),) or not np.isfinite(physical).all():
            raise ValueError("optimizer vector has wrong shape or non-finite entries")
        with np.errstate(over="ignore", under="ignore"):
            physical[self._log] = np.power(10.0, physical[self._log])
        if not np.isfinite(physical).all() or np.any(physical[self._log] <= 0):
            raise ValueError("log10 parameters must decode to positive finite values")
        return physical

    def unpack(self, vector):
        """Return the physical parameter mapping accepted by the primal predictor."""
        return dict(zip(self.names, map(float, self._physical(vector))))

    def predict(self, vector):
        """Return (values[time, channel], Jacobian[time, channel, parameter])."""
        model = self._model
        if float(model.dt) != self._dt:
            raise ValueError("dt changed; construct a new calibration tangent")
        physical = self._physical(vector)
        scale = np.ones_like(physical)
        scale[self._log] = np.log(10.0) * physical[self._log]
        model.reset()
        model.assign_parameters(dict(zip(self.names, map(float, physical))))

        shape = (self._times.size, self._weights.shape[0])
        values = np.empty(shape)
        jacobian = np.empty((*shape, len(self.names)))
        sensitivities = np.zeros((self._active.size, len(self.names)))
        for i, time in enumerate(self._times):
            self._previous.assign(model.w_n)
            model.step(time)
            # step() commits w_n; evaluate all derivatives at the solved pair.
            model.w_n.assign(self._previous)
            try:
                matrix = self._restrict(_assemble_csr(self._jacobian, self._matrix))
                rhs = self._mass_over_dt @ sensitivities
                for j, (form, tensor) in enumerate(zip(self._partials, self._vectors)):
                    assemble(form, tensor=tensor)
                    rhs[:, j] -= tensor.get_local()[self._active] * scale[j]
                sensitivities = splu(matrix.tocsc()).solve(rhs)
            finally:
                model.w_n.assign(model.w)
            values[i] = self._weights @ model.w.vector().get_local()
            jacobian[i] = self._active_weights @ sensitivities

        if not np.isfinite(values).all() or not np.isfinite(jacobian).all():
            raise RuntimeError("non-finite PDE trajectory or calibration tangent")
        return values, jacobian

    def relative_objective(self, observations, channels=(0, 2)):
        """Return SciPy ``fun, jac`` sharing the most recent successful solve."""
        channels = tuple(channels)
        observed = np.array(observations, dtype=float, copy=True)
        if (observed.shape != (self._times.size, len(channels))
                or not np.isfinite(observed).all() or np.any(observed <= 0)):
            raise ValueError("observations must be finite and positive with shape (time, channel)")
        if not channels or any(not 0 <= channel < self._weights.shape[0] for channel in channels):
            raise ValueError("objective channels are outside the prediction weights")
        last = None

        def evaluate(vector):
            nonlocal last
            if last is None or not np.array_equal(vector, last[0]):
                values, jacobian = self.predict(vector)
                last = (np.array(vector, copy=True), values, jacobian)
            return last[1:]

        def fun(vector):
            values, _ = evaluate(vector)
            return ((values[:, channels] - observed) / observed).ravel()

        def jac(vector):
            _, jacobian = evaluate(vector)
            return (jacobian[:, channels, :] / observed[..., None]).reshape(-1, len(self.names))

        return fun, jac
