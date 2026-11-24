"""Verify the mucus-domain average, velocity invariance in x, and bulk-form preservation."""

import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "solver/src"))

import numpy as np
from dolfin import Constant, Expression, assemble, exp, grad, interpolate, split
from lung_pde.mesh import MUCUS_FLAG
from lung_pde.model import build_model
from lung_pde.velocity_fields import MeanViscosityVelocity


def model(mean):
    return build_model(756, "treatment", 32, "original_mshr", eta=2.,
                       parameter_source="initial", dt=.1, conservative_exchange=True,
                       original_mesh=Path(os.environ["NTM_ORIGINAL_MESH"]),
                       polynomial_velocity=True, mean_viscosity_velocity=mean)


def main():
    updated, local = model(True), model(False)
    assert isinstance(updated.velocity, MeanViscosityVelocity)
    mref = float(updated.parameters["m0"])
    for obj in (updated, local):
        obj.velocity.t = .5
        state = interpolate(Constant((2e-6, 1e-6, .002, .001, 1.3 * mref)),
                            obj.w.function_space())
        obj.w.assign(state)
        obj.w_n.assign(state)
    assert updated.velocity.update_state(updated.w.vector())
    np.testing.assert_allclose(updated.velocity.mean_eta, 2 * np.exp(.3), rtol=2e-14)
    np.testing.assert_allclose(assemble(updated.form.residual).get_local(),
                               assemble(local.form.residual).get_local(),
                               rtol=1e-12, atol=1e-16)

    nonuniform = Expression(("0", "0", "0", "0", "ref*(.3+3*x[0]+x[1])"),
                            ref=mref, degree=1)
    updated.w.assign(interpolate(nonuniform, updated.w.function_space()))
    assert updated.velocity.update_state(updated.w.vector())
    dx = updated.mesh.dx(MUCUS_FLAG, metadata={"quadrature_degree": 12})
    area = assemble(Constant(1.) * dx)
    m = split(updated.w)[4]
    mean = assemble(updated.eta * exp(m / updated.parameters["m0"] - 1.) * dx) / area
    np.testing.assert_allclose(updated.velocity.mean_eta, mean, rtol=2e-13)
    average_inverse = assemble(exp(1. - m / updated.parameters["m0"]) / updated.eta * dx) / area
    assert abs(1. / mean - average_inverse) > 1e-4
    derivative = grad(updated.velocity.form)[0, 0]
    np.testing.assert_allclose(assemble(derivative**2 * dx), 0., atol=1e-25)
    print("Uniform and varying mucus viscosity, x-independence, and unchanged bulk terms verified.")


if __name__ == "__main__":
    main()
