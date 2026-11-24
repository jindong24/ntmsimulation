import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dolfin")

from lung_pde.mesh import MeshConfig, build_mesh
from lung_pde.model import build_model


@pytest.mark.fenics
def test_original_mesh_and_active_tau_parameters():
    file = Path(os.environ["NTM_ORIGINAL_MESH"])
    grid = build_mesh(MeshConfig(backend="original_mshr", original_mesh=file))
    assert (grid.mesh.num_vertices(), grid.mesh.num_cells()) == (1643, 3142)
    assert (grid.mucus_mesh.num_vertices(), grid.tissue_mesh.num_vertices()) == (838, 842)
    model = build_model(756, "calibration", mesh_backend="original_mshr", original_mesh=file,
                        conservative_exchange=True)
    assert "tauB" in model.parameters and "tauM" in model.parameters
    assert not any(k.startswith("kappa") for k in model.parameters)
    with pytest.raises(ValueError, match="tauB"):
        model.assign_parameters({"tauB": -1})


@pytest.mark.fenics
def test_original_mesh_conserves_both_species_with_no_reactions():
    file = Path(os.environ["NTM_ORIGINAL_MESH"])
    model = build_model(756, "calibration", mesh_backend="original_mshr", original_mesh=file,
                        conservative_exchange=True)
    model.assign_parameters(dict(tauB=.005, tauM=.02, DB1=.01, DB2=.003, DM=.002,
                                 delta=.01, lambdaB=0., dMb=0., dB=0., dM=0.,
                                 alpha0=0., alpha1=0.))
    start = np.asarray(model.concentrations()[:4]).reshape(2, 2)
    for step in range(1, 4):
        model.step(step / 6)
    end = np.asarray(model.concentrations()[:4]).reshape(2, 2)
    assert np.all(end[:, 1] > 0.)
    np.testing.assert_allclose(end.sum(axis=1), start.sum(axis=1), atol=5e-11, rtol=0.)
