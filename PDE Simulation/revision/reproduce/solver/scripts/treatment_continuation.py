"""Solve neighboring doses using the preceding dose as a Newton initial guess."""

import numpy as np


def dose_trajectory(model, time, weights, guesses=None):
    """Keep the new dose's actual previous time step separate from its guess."""
    model.reset()
    dim = model.w.function_space().dim()
    states = np.empty((len(time) - 1, dim))
    for i, moment in enumerate(time[1:]):
        if guesses is not None:
            vector = model.w.vector()
            vector.set_local(guesses[i])
            vector.apply("insert")
        try:
            model.step(float(moment))
        except RuntimeError:
            if guesses is None:
                raise
            model.w.assign(model.w_n)
            model.step(float(moment))
        states[i] = model.w.vector().get_local()
    endpoint = weights @ states[-1]
    if not np.isfinite(states).all() or not np.isfinite(endpoint).all():
        raise RuntimeError("non-finite treatment trajectory")
    return endpoint, states
