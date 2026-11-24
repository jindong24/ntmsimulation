import numpy as np
import pytest

dolfin = pytest.importorskip("dolfin")

from lung_pde.mesh import RIGHT_MUCUS_FLAG
from lung_pde.model import build_model
from lung_pde.velocity_fields import MucusVelocity, TreatmentVelocity


@pytest.mark.fenics
@pytest.mark.parametrize("mode,old_expression", (
    ("velocity", MucusVelocity), ("treatment", TreatmentVelocity),
))
def test_conservative_modes_keep_the_legacy_velocity(mode, old_expression):
    model = build_model(756, mode, resolution=8, mesh_backend="rectangle",
                        conservative_exchange=True)
    model.assign_parameters(dict(tauB=.004, tauM=.008))
    old = old_expression(t=0., degree=3)
    for t in (0., .21, 1., 1.8):
        model.velocity.t = old.t = t
        for point in ((.02, .15), (.17, .2)):
            np.testing.assert_allclose(model.velocity(point), old(point), atol=2e-17)


@pytest.mark.fenics
def test_treatment_mass_balance_with_no_reactions():
    dt = .1
    model = build_model(756, "treatment", resolution=8, mesh_backend="rectangle",
                        danti=0., eta=.7, dt=dt, conservative_exchange=True,
                        initial_loads=(1.4091e-6, .001974684))
    model.assign_parameters(dict(
        tauB=.005, tauM=.02, DB1=.01, DB2=.003, DM=.002, delta=.01,
        lambdaB=0., dMb=0., dB=0., dM=0., alpha0=0., alpha1=0.,
    ))
    model.solver.parameters["absolute_tolerance"] = 1e-13
    model.solver.parameters["relative_tolerance"] = 1e-8
    before = np.asarray(model.concentrations()[:4]).reshape(2, 2).sum(axis=1)
    clearance = np.zeros(2)
    for i in range(1, 4):
        model.step(i * dt)
        b, _, m, _, _ = dolfin.split(model.w)
        right = model.mesh.ds(RIGHT_MUCUS_FLAG)
        clearance += 50 * .01 * dt * np.array([dolfin.assemble(f * right) for f in (b, m)])
    after = np.asarray(model.concentrations()[:4]).reshape(2, 2)
    assert np.all(after[:, 1] > 0)
    np.testing.assert_allclose(after.sum(axis=1) + clearance, before, atol=2e-11, rtol=0.)
