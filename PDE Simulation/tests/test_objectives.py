import numpy as np

from lung_pde.objectives import OBJECTIVES


def test_named_legacy_objectives_match_literal_formulas():
    sim_b = np.array([2.0, 4.0])
    sim_m = np.array([3.0, 8.0])
    obs_b = np.array([1.0, 2.0])
    obs_m = np.array([2.0, 4.0])
    assert OBJECTIVES["relative_mse_sum"](sim_b, sim_m, obs_b, obs_m) == (
        np.sum((sim_b - obs_b) ** 2 / obs_b ** 2)
        + np.sum((sim_m - obs_m) ** 2 / obs_m ** 2)
    )
    assert OBJECTIVES["relative_mse_mean"](sim_b, sim_m, obs_b, obs_m) == (
        np.mean((sim_b - obs_b) ** 2 / obs_b ** 2)
        + np.mean((sim_m - obs_m) ** 2 / obs_m ** 2)
    )
    assert OBJECTIVES["relative_l2_target"](sim_b, sim_m, obs_b, obs_m) == (
        np.linalg.norm(sim_b - obs_b) / np.linalg.norm(obs_b)
        + np.linalg.norm(sim_m - obs_m) / np.linalg.norm(obs_m)
    )
    assert OBJECTIVES["relative_l2_prediction"](sim_b, sim_m, obs_b, obs_m) == (
        np.linalg.norm(sim_b - obs_b) / np.linalg.norm(sim_b)
        + np.linalg.norm(sim_m - obs_m) / np.linalg.norm(sim_m)
    )
    assert OBJECTIVES["bacteria_relative_mse_mean"](sim_b, sim_m, obs_b, obs_m) == np.mean(
        (sim_b - obs_b) ** 2 / obs_b ** 2
    )
