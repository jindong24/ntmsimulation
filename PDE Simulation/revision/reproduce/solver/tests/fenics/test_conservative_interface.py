import numpy as np
import pytest

pytest.importorskip('dolfin')

from lung_pde.model import build_model


@pytest.mark.fenics
def test_interface_exchanges_both_species_without_changing_their_totals():
    model = build_model(
        756, 'calibration', resolution=8, mesh_backend='rectangle',
        conservative_exchange=True,
    )
    model.assign_parameters(dict(
        tauB=0.005, tauM=0.01, DB1=0.01, DB2=0.003, DM=0.002, delta=0.01,
        lambdaB=0.0, dMb=0.0, dB=0.0, dM=0.0, alpha0=0.0, alpha1=0.0,
    ))
    before = np.array(model.concentrations()[:4]).reshape(2, 2)
    for i in range(1, 4):
        model.step(i * model.config.dt)
    after = np.array(model.concentrations()[:4]).reshape(2, 2)
    assert np.all(after[:, 1] > 0)
    assert np.all(after[:, 0] < before[:, 0])
    assert after.sum(axis=1) == pytest.approx(before.sum(axis=1), abs=1e-11)
