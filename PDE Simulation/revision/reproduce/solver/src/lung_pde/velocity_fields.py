from math import cos, pi

import numpy as np
from dolfin import Constant, SpatialCoordinate, UserExpression, as_vector, cells

from .mesh import MUCUS_FLAG


class _TimedVelocity(UserExpression):
    def __init__(self, t=0.0, **kwargs):
        super().__init__(**kwargs)
        self.t = float(t)

    def value_shape(self):
        return (2,)


class ZeroVelocity(_TimedVelocity):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0


class MucusVelocity(_TimedVelocity):
    def eval(self, values, x):
        factor = 0.5 * (np.sin(np.pi * self.t - np.pi / 2.0) + 1.0)
        values[0] = factor * (x[1] - 0.1) ** 2
        values[1] = 0.0


class TreatmentVelocity(_TimedVelocity):
    def eval(self, values, x):
        factor = 0.5 * (np.sin(2.0 * np.pi * self.t - np.pi / 2.0) + 1.0)
        values[0] = 1.44 * factor * (x[1] - 0.1) ** 2
        values[1] = 0.0


class PolynomialVelocity:
    """The legacy polynomial velocity evaluated directly at quadrature points."""

    def __init__(self, mesh, treatment):
        self.scale = 1.44 if treatment else 1.
        self.frequency = (2. if treatment else 1.) * pi
        self.factor = Constant(0.)
        y = SpatialCoordinate(mesh)[1]
        self.form = as_vector((self.factor * (y - .1)**2, 0.))
        self.t = 0.

    @property
    def t(self):
        return self._time

    @t.setter
    def t(self, value):
        self._time = value
        self.factor.assign(.5 * self.scale * (1. - cos(self.frequency * value)))


class MeanViscosityVelocity(PolynomialVelocity):
    """Preserve the baseline shear profile and update its spatially uniform scale."""

    def __init__(self, bundle, eta0, m_ref):
        self.eta0 = eta0
        self.m_ref = m_ref
        self.mean_eta = 1.
        super().__init__(bundle.mesh, treatment=True)
        mucus = [c for c in cells(bundle.mesh) if bundle.cell_mark[c] == MUCUS_FLAG]
        dofmap = bundle.function_space.sub(4).dofmap()
        self.mucin_dofs = np.asarray([dofmap.cell_dofs(c.index()) for c in mucus])
        areas = np.asarray([c.volume() for c in mucus])
        # Positive seven-point degree-five quadrature on each P1 triangle.
        a, b = .059715871789770, .470142064105115
        c, d = .797426985353087, .101286507323456
        self.barycentric = np.array([
            (1/3, 1/3, 1/3), (a, b, b), (b, a, b), (b, b, a),
            (c, d, d), (d, c, d), (d, d, c),
        ])
        weights = np.array([.225, *([.132394152788506] * 3), *([.125939180544827] * 3)])
        self.weights = ((areas / areas.sum())[:, None] * weights).ravel()

    @PolynomialVelocity.t.setter
    def t(self, value):
        self._time = value
        self.baseline_factor = .5 * self.scale * (1. - cos(self.frequency * value))
        self.factor.assign(self.baseline_factor / self.mean_eta)

    def update_state(self, state):
        if hasattr(state, "get_local"):
            state = state.get_local()
        ref, eta = float(self.m_ref), float(self.eta0)
        if not (np.isfinite(ref) and np.isfinite(eta) and ref > 0. and eta > 0.):
            return False
        nodal = state[self.mucin_dofs]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            relative = np.exp(nodal @ (self.barycentric.T / ref) - 1.)
            mean = eta * float(self.weights @ relative.ravel())
        if not np.isfinite(mean) or mean <= 0.:
            return False
        self.mean_eta = mean
        self.factor.assign(self.baseline_factor / mean)
        return True
