import numpy as np
from dolfin import UserExpression


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
