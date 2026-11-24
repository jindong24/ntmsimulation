"""Same-design PRCC comparison with bacterial exchange held fixed."""

import argparse
import math
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np

from compare_interface import ROOT, restore_interface_chemotaxis
from revision_runs import FIELDS, PARAMETERS, make_model, read_rows, write_rows
from revision_sensitivity import INITIAL_LOADS, TRANSFER, flux_forms, prcc
from calibrate_conservative import concentration_weights


SOURCE = Path(os.environ.get('NTM_GLOBAL_SENSITIVITY', ROOT / 'runs/sensitivity/range_rescaled'))
RESULTS = Path(__file__).resolve().parent / 'prcc'
THRESHOLD = 'p_Bonferroni_16_per_outcome'
MFLUX = ('M_M_TO_T', 'M_T_TO_M', 'M_NET_TO_T')
TIME = np.arange(141) * .1


def extra_flux_forms(model):
    from dolfin import Form, as_vector, avg, dot, grad, split
    from lung_pde.conservative_interface import interface_fluxes
    from lung_pde.mesh import INTERFACE_FLAG
    from ufl import max_value

    _, bt, _, mt, _ = split(model.w)
    n = model.mesh.normal
    up = as_vector((0., 1.))
    dn_bt = sum(dot(grad(bt(s)), n(s)) * (1. + dot(n(s), up)) / 2.
                for s in ('+', '-'))
    _, shared = interface_fluxes(model.w, model.parameters, model.eta)
    # Tissue residual is -shared + delta * M^t * dn B^t.
    incoming = shared - model.parameters['delta'] * avg(mt) * dn_bt
    interface = model.mesh.dS(INTERFACE_FLAG)
    return tuple(Form(50 * max_value(j, 0.) * interface) for j in (incoming, -incoming))


def setup(first, hybrid):
    from lung_pde.ufc_batch import BatchNewton
    global model, kernel, weights, forms
    model = make_model(756, 'treatment', {k: float(first[k]) for k in PARAMETERS}, 32, .1,
                       initial_loads=INITIAL_LOADS, interpolate_profile=True,
                       polynomial_velocity=True, mean_viscosity_velocity=True,
                       macrophage_diffusivity_in_flux=True)
    if hybrid:
        restore_interface_chemotaxis(model)
    kernel = BatchNewton(model, os.environ['NTM_UFC_LIBRARY'])
    weights = concentration_weights(model)
    forms = flux_forms(model) + (extra_flux_forms(model) if hybrid else ())


def simulate(row):
    from dolfin import NewtonSolver, assemble

    sid = int(row['sample_id'])
    model.assign_parameters({k: float(row[k]) for k in PARAMETERS})
    model.eta.assign(float(row['eta']))
    model.danti.assign(float(row['danti']))
    assert math.isclose(model.effective_danti, float(row['danti_effective']), rel_tol=1e-12)

    def run(accelerated):
        model.reset()
        kernel.factor = None
        w = model.w.vector()
        state = w.get_local()
        integral = np.zeros(len(forms))
        for t in TIME[1:]:
            if accelerated:
                state = kernel.step(float(t), state, lag=True)
                w.set_local(state)
                w.apply('insert')
            else:
                model.step(float(t))
                state = w.get_local()
            integral += TIME[1] * np.array([assemble(form) for form in forms])
        c = (weights @ state)[:4]
        assert np.isfinite(c).all() and np.isfinite(integral).all()
        return c, integral

    try:
        c, q = run(True)
    except RuntimeError as error:
        print(f'Sample {sid}: retry with direct solver ({error})', flush=True)
        old_solver = model.solver
        direct = NewtonSolver()
        direct.parameters['report'] = False
        direct.parameters['linear_solver'] = 'lu'
        direct.parameters['absolute_tolerance'] = 1e-13
        direct.parameters['relative_tolerance'] = 1e-7
        direct.parameters['maximum_iterations'] = 70
        model.solver = direct
        try:
            c, q = run(False)
        finally:
            model.solver = old_solver

    measured = dict(zip(TRANSFER, [*q[:6], *(q[2:4] - q[4:6])]))
    result = {'sample_id': sid, 'time_day': 14., **dict(zip(FIELDS, c)),
              'B_TOTAL': c[0]+c[1], 'M_TOTAL': c[2]+c[3], **measured}
    if len(q) > 6:
        result.update(dict(zip(('M_TISSUE_IN', 'M_TISSUE_OUT', 'M_TISSUE_NET'),
                               (q[6], q[7], q[6]-q[7]))))
    return result


def verify_archive(design, new):
    setup(design[0], hybrid=False)
    for sid in (1, 182, 400):
        row = simulate(design[sid-1])
        target = new[sid-1]
        keys = tuple(row)[2:]
        a = np.array([row[k] for k in keys])
        b = np.array([target[k] for k in keys])
        np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-8, err_msg=f'sample {sid}')
        print(f'Conservative archive sample {sid} reproduced: max abs error {max(abs(a-b)):.3g}', flush=True)
    existing = read_rows(SOURCE / 'prcc_day14_solver_inputs.csv')
    recomputed = prcc(design, new)
    assert len(existing) == len(recomputed) == 224
    for a, b in zip(existing, recomputed):
        assert (a['outcome'], a['parameter']) == (b['outcome'], b['parameter'])
        np.testing.assert_allclose(float(a['PRCC']), b['PRCC'], atol=2e-12, rtol=0.)
    print('Archived conservative PRCC reproduced for all 224 combinations', flush=True)


def compare(design, old, new):
    base = prcc(design, new)
    separate = ('M_TISSUE_IN', 'M_TISSUE_OUT', 'M_TISSUE_NET')
    for kind in ('same_diagnostic', 'physical_tissue_flux'):
        previous = []
        for values in old:
            record = {k: values[k] for k in new[0]}
            if kind == 'physical_tissue_flux':
                record.update({k: values[j] for k, j in zip(MFLUX, separate)})
            previous.append(record)
        prior_prcc = prcc(design, previous)
        pairs = []
        for a, b in zip(prior_prcc, base):
            assert (a['outcome'], a['parameter'], a['n']) == (b['outcome'], b['parameter'], 400)
            p1, p2 = a[THRESHOLD], b[THRESHOLD]
            pairs.append({'outcome': a['outcome'], 'parameter': a['parameter'], 'n': a['n'],
                          'old_M_PRCC': a['PRCC'], 'conservative_M_PRCC': b['PRCC'],
                          'delta_PRCC': b['PRCC']-a['PRCC'], 'old_p_adjusted': p1,
                          'new_p_adjusted': p2, 'significance_flip': (p1 < .05) != (p2 < .05),
                          'significant_sign_reversal': p1 < .05 and p2 < .05 and a['PRCC']*b['PRCC'] < 0})
        ordered = sorted(pairs, key=lambda r: abs(r['delta_PRCC']), reverse=True)
        write_rows(RESULTS / f'{kind}.csv', ordered)
        print(kind, 'largest:', ordered[:3], 'significance flips:',
              sum(r['significance_flip'] for r in pairs), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-archive', action='store_true')
    parser.add_argument('--warmup-hybrid', action='store_true')
    parser.add_argument('--workers', type=int, default=7)
    args = parser.parse_args()
    design_file = SOURCE / 'design.csv'
    design = read_rows(design_file if design_file.exists() else SOURCE / 'parameter_set_samples.csv')
    raw = read_rows(SOURCE / 'endpoints.csv')
    assert [int(r['sample_id']) for r in design] == [int(r['sample_id']) for r in raw] == list(range(1, 401))
    new = [{k: float(v) for k, v in d.items()} for d in raw]
    if args.verify_archive:
        verify_archive(design, new)
        return
    per_sample = RESULTS / 'cases'
    per_sample.mkdir(parents=True, exist_ok=True)
    if args.warmup_hybrid:
        setup(design[0], hybrid=True)
        initial = simulate(design[0])
        write_rows(per_sample / '001.csv', [initial])
        print('Hybrid replay warmup completed.', flush=True)
        return
    jobs = [d for d in design if not (per_sample / f"{int(d['sample_id']):03d}.csv").exists()]
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'),
                                 initializer=setup, initargs=(design[0], True)) as pool:
            futures = [pool.submit(simulate, row) for row in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                r = future.result()
                write_rows(per_sample / f"{r['sample_id']:03d}.csv", [r])
                if i % 25 == 0 or i == len(jobs):
                    print(f'{i}/{len(jobs)} matched hybrid replays completed', flush=True)
    old = [{k: float(v) for k, v in read_rows(per_sample / f'{i:03d}.csv')[0].items()}
           for i in range(1, 401)]
    assert [r['sample_id'] for r in old] == [r['sample_id'] for r in new]
    write_rows(RESULTS / 'hybrid_endpoints.csv', old)
    compare(design, old, new)


if __name__ == '__main__':
    main()
