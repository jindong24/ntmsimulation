import numpy as np


def relative_mse_sum(sb, sm, ob, om):
    return float(np.sum((sb - ob) ** 2 / ob ** 2) + np.sum((sm - om) ** 2 / om ** 2))


def relative_mse_mean(sb, sm, ob, om):
    return float(np.mean((sb - ob) ** 2 / ob ** 2) + np.mean((sm - om) ** 2 / om ** 2))


def relative_l2_target(sb, sm, ob, om):
    return float(
        np.linalg.norm(sb - ob) / np.linalg.norm(ob)
        + np.linalg.norm(sm - om) / np.linalg.norm(om)
    )


def relative_l2_prediction(sb, sm, ob, om):
    return float(
        np.linalg.norm(sb - ob) / np.linalg.norm(sb)
        + np.linalg.norm(sm - om) / np.linalg.norm(sm)
    )


def bacteria_relative_mse_mean(sb, sm, ob, om):
    return float(np.mean((sb - ob) ** 2 / ob ** 2))


OBJECTIVES = {
    "relative_mse_sum": relative_mse_sum,
    "relative_mse_mean": relative_mse_mean,
    "relative_l2_target": relative_l2_target,
    "relative_l2_prediction": relative_l2_prediction,
    "bacteria_relative_mse_mean": bacteria_relative_mse_mean,
}
