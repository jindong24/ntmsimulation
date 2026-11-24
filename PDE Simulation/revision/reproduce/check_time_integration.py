"""Cross-check the cached assembly against direct DOLFIN with the new velocity."""

import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "solver/scripts"), str(HERE / "solver/src")]

import numpy as np
from dolfin import Constant, assemble, exp, split
from lung_pde.mesh import MUCUS_FLAG
from lung_pde.ufc_batch import BatchNewton
from revision_runs import PARAMETERS, make_model, read_rows
from revision_sensitivity import INITIAL_LOADS


def main(design):
    row = read_rows(design)[0]
    params = {k: float(row[k]) for k in PARAMETERS}
    model = make_model(756, "treatment", params, 32, .1, initial_loads=INITIAL_LOADS,
                       interpolate_profile=True, polynomial_velocity=True,
                       mean_viscosity_velocity=True)
    model.eta.assign(float(row["eta"]))
    model.danti.assign(float(row["danti"]))
    model.reset()
    solver = BatchNewton(model, os.environ["NTM_UFC_LIBRARY"])
    state = model.w.vector().get_local()
    times = np.arange(1, 13) * .1
    for time in times:
        previous = state
        state = solver.step(time, previous, lag=True)
        residual = solver.assembly.f(state, previous)
        assert np.linalg.norm(residual) < 1e-11
    model.reset()
    for time in times:
        model.step(time)
    direct = model.w.vector().get_local()
    np.testing.assert_allclose(state, direct, rtol=5e-7, atol=5e-11)
    m = split(model.w)[4]
    dx = model.mesh.dx(MUCUS_FLAG, metadata={"quadrature_degree": 12})
    direct_mean = assemble(model.eta * exp(m/model.parameters["m0"] - 1.) * dx) / assemble(Constant(1.) * dx)
    assert model.velocity.update_state(model.w.vector())
    # The spatially varying solved field is not a polynomial under the exponential.
    np.testing.assert_allclose(direct_mean, model.velocity.mean_eta, rtol=5e-8)
    rel = abs(model.velocity.mean_eta / direct_mean - 1.)
    print(f"12 implicit steps: accelerated and direct solutions agree. "
          f"Current-state mean viscosity differs from higher-order quadrature by {rel:.2e}.")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
