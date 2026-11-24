"""Paired diagnostic: restore only the historical macrophage interface terms."""

import argparse
import csv
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / 'calibration'
TREAT = ROOT / 'inputs/treatment/recalibrated/applied_parameters.csv'
SETS = (48, 83, 182, 188, 335, 586, 646, 756, 808, 843)
FIELDS = ('B_mucus', 'B_tissue', 'M_mucus', 'M_tissue', 'B_total', 'M_total')


def inputs(sid, mode):
    base = (CAL / 'multistart_182_335/selected' if sid in (182, 335)
            else CAL / 'full')
    path = base / f'set_{sid:03d}/metrics.json'
    report = json.loads(path.read_text())
    assert report['set'] == sid and report['mesh_backend'] == 'original_mshr'
    fitted = report['fitted_parameters']
    if mode == 'calibration':
        return report['fixed_parameters'] | fitted, str(path)
    with TREAT.open(newline='') as stream:
        treatment = next(row for row in csv.DictReader(stream) if int(row['set_id']) == sid)
    fixed = {k: float(treatment[k]) for k in report['fixed_parameters']}
    return fixed | fitted, str(path)


def restore_interface_chemotaxis(model):
    from dolfin import Form, TestFunctions, TrialFunction, as_vector, avg, derivative, dot, grad, split
    from lung_pde.form_context import FormBundle
    from lung_pde.mesh import INTERFACE_FLAG

    b, bt, m, mt, _ = split(model.w)
    _, _, f, ft, _ = TestFunctions(model.w.function_space())
    n = model.mesh.normal
    down = as_vector((0., -1.))
    measure = model.mesh.dS(INTERFACE_FLAG)
    delta = model.parameters['delta']

    def normal_derivative(u, orientation):
        return sum(dot(grad(u(side)), n(side)) *
                   (1. + dot(n(side), orientation)) / 2. for side in ('+', '-'))

    # The old law differs by these side-specific chemotaxis terms when DM*tauM
    # in the old parameterization is matched to the same effective coefficient.
    corrections = (delta * normal_derivative(b, down) * avg(m) * avg(f) * measure,
                   delta * normal_derivative(bt, -down) * avg(mt) * avg(ft) * measure)
    b1, b2, m1, m2, mucin = model.form.terms
    terms = (b1, b2, m1 + corrections[0], m2 + corrections[1], mucin)
    residual = sum(terms)
    jacobian = derivative(residual, model.w, TrialFunction(model.w.function_space()))
    model.form = FormBundle(residual, jacobian, terms)
    model.problem.residual = Form(residual)
    model.problem.jacobian = Form(jacobian)


def solve(model, time, weights):
    from lung_pde.ufc_batch import BatchNewton

    model.reset()
    state = model.w.vector().get_local()
    kernel = BatchNewton(model, os.environ['NTM_UFC_LIBRARY'])
    rows = [(weights @ state)[:4]]
    for t in time[1:]:
        state = kernel.step(float(t), state, lag=True)
        rows.append((weights @ state)[:4])
    values = np.asarray(rows)
    assert np.isfinite(values).all()
    return np.c_[values, values[:, :2].sum(axis=1), values[:, 2:4].sum(axis=1)], kernel.fallbacks


def one(job):
    from calibrate_conservative import concentration_weights
    from revision_runs import make_model

    sid, mode, output, warmup = job
    params, source = inputs(sid, mode)
    treatment = mode == 'treatment'
    dt = .1 if treatment else 1/6
    model = make_model(
        sid, mode, params, 32, dt, interpolate_profile=treatment,
        polynomial_velocity=treatment, mean_viscosity_velocity=treatment,
        macrophage_diffusivity_in_flux=False,
    )
    if treatment:
        model.eta.assign(1.)
        model.danti.assign(1.)
    time = np.arange(1 + round(14/dt)) * dt
    weights = concentration_weights(model)
    new, f_new = solve(model, time, weights)
    restore_interface_chemotaxis(model)
    old, f_old = solve(model, time, weights)
    np.testing.assert_array_equal(old[0], new[0])
    if warmup:
        effective_delta = float(model.parameters['delta'])
        model.parameters['delta'].assign(0.)
        zero_old, _ = solve(model, time[:4], weights)
        zero_new_model = make_model(
            sid, mode, params | {'delta': 0.}, 32, dt, interpolate_profile=treatment,
            polynomial_velocity=treatment, mean_viscosity_velocity=treatment,
            macrophage_diffusivity_in_flux=False,
        )
        if treatment:
            zero_new_model.eta.assign(1.)
            zero_new_model.danti.assign(1.)
        zero_new, _ = solve(zero_new_model, time[:4], concentration_weights(zero_new_model))
        np.testing.assert_allclose(zero_old, zero_new, rtol=1e-10, atol=1e-15)
        model.parameters['delta'].assign(effective_delta)
    result = {'dataset': sid, 'mode': mode, 'dt': dt, 'calibration_source': source,
              'delta': params['delta'], 'tauM_effective': params['tauM'],
              'new_fallbacks': f_new, 'old_fallbacks': f_old, 'metrics': {}}
    for i, name in enumerate(FIELDS):
        a, b = old[:, i], new[:, i]
        result['metrics'][name] = {
            'old_day14': float(a[-1]), 'new_day14': float(b[-1]),
            'day14_change_pct': float(100 * (b[-1] - a[-1]) / abs(a[-1])),
            'trajectory_relative_L2_pct': float(100 * np.linalg.norm(b-a) / np.linalg.norm(a)),
        }
    if not treatment:
        np.testing.assert_allclose(time[1:-1], model.abm.times, atol=1e-12, rtol=0.)
        targets = np.c_[model.abm.bacteria, model.abm.macrophages]
        result['mucus_msre'] = {
            label: dict(zip(('bacteria', 'macrophages'),
                            np.mean(((series[1:-1, [0, 2]] - targets) / targets)**2, axis=0).tolist()))
            for label, series in [('old', old), ('new', new)]
        }
    np.savez(output / f'{mode}_{sid}_trajectories.npz', time=time, old=old, new=new, fields=FIELDS)
    (output / f'{mode}_{sid}.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'{mode} {sid}: day-14 M mucus/tissue/total changes '
          + ', '.join(f'{result["metrics"][k]["day14_change_pct"]:+.3g}%'
                      for k in ('M_mucus', 'M_tissue', 'M_total')), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / 'results'
    root.mkdir(exist_ok=True)
    modes = ('calibration', 'treatment')
    if args.pilot:
        for mode in modes:
            one((756, mode, root, True))
        return
    jobs = [(sid, mode, root, False) for mode in modes for sid in SETS
            if not (root / f'{mode}_{sid}.json').exists()]
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            list(pool.map(one, jobs))
    results = [json.loads((root / f'{mode}_{sid}.json').read_text())
               for mode in modes for sid in SETS]
    (root.parent / 'comparison.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
