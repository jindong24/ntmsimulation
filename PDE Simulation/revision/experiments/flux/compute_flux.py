"""Replay range-rescaled samples and export the original lung flux diagnostic."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import math
import multiprocessing as mp
import os
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np


COLUMNS = ('time (day)', 'B mucus2tissue', 'M mucus2tissue', 'B tissue2mucus',
           'M tissue2mucus', 'B right', 'M right')
TIME = np.arange(1, 141) / 10.
FIXED = {'delta', 'Dm', 'dm', 'm0', 'beta0', 'beta1'}
ALIASES = {'eta0': 'eta', 'xi': 'danti'}


def sample_parameters(path):
    with ZipFile(path) as book:
        root = ET.fromstring(book.read('xl/worksheets/sheet1.xml'))
        tabs = ET.fromstring(book.read('xl/workbook.xml'))
        shared = ([''.join(s.itertext()) for s in ET.fromstring(book.read('xl/sharedStrings.xml'))]
                  if 'xl/sharedStrings.xml' in book.namelist() else [])
    if tabs.find('.//{*}sheet').get('name') != 'variable':
        raise ValueError(f'{path}: expected variable as first sheet')
    def value(cell):
        cell_type, raw = cell.get('t'), cell.findtext('{*}v')
        if cell_type == 's':
            return shared[int(raw)]
        return raw or cell.findtext('{*}is/{*}t')

    cells = [[value(c) for c in row]
             for row in root.findall('.//{*}sheetData/{*}row')]
    if cells[0] != ['Variable', 'Value']:
        raise ValueError(f'{path}: unexpected parameter header')
    values = {ALIASES.get(k, k): float(v) for k, v in cells[1:]}
    if len(values) != len(cells) - 1 or not all(map(math.isfinite, values.values())):
        raise ValueError(f'{path}: invalid or duplicate parameter')
    return values


def flux_rows(raw):
    if raw.shape != (len(TIME), 4) or not np.isfinite(raw).all():
        raise ValueError('expected 140 finite time steps of four assembled fluxes')
    return np.column_stack((TIME, raw[:, :2], -raw[:, :2], raw[:, 2:]))


def read_csv(path):
    with path.open(newline='') as stream:
        return {int(row.pop('sample_id')): {k: float(v) for k, v in row.items()}
                for row in csv.DictReader(stream)}


def setup(first):
    from dolfin import Form, avg, dot, grad, split
    from calibrate_conservative import concentration_weights
    from lung_pde.mesh import INTERFACE_FLAG, RIGHT_MUCUS_FLAG
    from lung_pde.ufc_batch import BatchNewton
    from lung_pde.velocity_fields import PolynomialVelocity
    from revision_runs import PARAMETERS, make_model
    from revision_sensitivity import INITIAL_LOADS

    global model, solver, weights, velocity, forms
    model = make_model(756, 'treatment', {k: first[k] for k in PARAMETERS}, 32, .1,
                       initial_loads=INITIAL_LOADS, interpolate_profile=True,
                       polynomial_velocity=True, mean_viscosity_velocity=True)
    solver = BatchNewton(model, os.environ['NTM_UFC_LIBRARY'])
    weights = concentration_weights(model)
    b, bt, m, mt, _ = split(model.w)
    p, mesh = model.parameters, model.mesh
    interface, right = mesh.dS(INTERFACE_FLAG), mesh.ds(RIGHT_MUCUS_FLAG)
    # The historical output used unscaled u0 and DB1 in both right-edge columns.
    velocity = PolynomialVelocity(mesh.mesh, treatment=True)
    forms = tuple(Form(form) for form in (
        p['tauB'] * (avg(bt) - avg(b)) * interface,
        p['tauM'] * (avg(mt) - avg(m)) * interface,
        dot(mesh.normal, velocity.form * b - p['DB1'] * grad(b)) * right,
        dot(mesh.normal, velocity.form * m - p['DB1'] * grad(m)) * right,
    ))


def run_sample(job):
    from dolfin import NewtonSolver, assemble
    from revision_runs import FIELDS, PARAMETERS

    sid, params, endpoints, root = job
    model.assign_parameters({k: params[k] for k in PARAMETERS})
    model.eta.assign(params['eta'])
    model.danti.assign(params['danti'])
    if not math.isclose(model.effective_danti, params['danti_effective'], rel_tol=1e-12):
        raise ValueError(f'effective antibiotic dose mismatch: sample {sid}')

    def solve(backend):
        model.reset()
        vector = model.w.vector()
        state = vector.get_local()
        raw = np.empty((len(TIME), 4))
        for i in range(len(TIME)):
            moment = (i + 1) * .1  # Same stepping sequence as the archived run.
            if backend is None:
                model.step(moment)
                state = vector.get_local()
            else:
                state = backend.step(moment, state, lag=True)
                vector.set_local(state)
                vector.apply('insert')
            velocity.t = moment
            raw[i] = [assemble(form) for form in forms]
        return flux_rows(raw), (weights @ state)[:4]

    solver.factor = None
    try:
        rows, last = solve(solver)
    except RuntimeError as error:
        print(f'sample {sid}: retrying with direct solve: {error}', flush=True)
        old = model.solver
        direct = NewtonSolver()
        direct.parameters['report'] = False
        direct.parameters['linear_solver'] = 'lu'
        direct.parameters['absolute_tolerance'] = 1e-13
        direct.parameters['relative_tolerance'] = 1e-7
        direct.parameters['maximum_iterations'] = 70
        model.solver = direct
        try:
            rows, last = solve(None)
        finally:
            model.solver = old
    reference = np.array([endpoints[k] for k in FIELDS])
    np.testing.assert_allclose(last, reference, rtol=1e-6, atol=2e-14,
                               err_msg=f'sample {sid} day-14 concentrations')
    temp = root / f'sample_{sid}.tmp'
    np.savetxt(temp, rows, delimiter=',', header=','.join(COLUMNS), comments='', fmt='%.17g')
    temp.replace(root / f'sample_{sid}.csv')
    return sid, float(np.max(np.abs(last - reference)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--ids', type=int, nargs='+')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    design_path = args.samples.parent / 'parameter_set_samples.csv'
    design = read_csv(design_path if design_path.is_file() else args.archive / 'design.csv')
    endpoints = read_csv(args.archive / 'endpoints.csv')
    ids = range(1, 401)
    assert set(design) == set(endpoints) == set(ids)
    args.output.mkdir(exist_ok=True)
    jobs = []
    for sid in args.ids or ids:
        params = sample_parameters(args.samples / f'sample_{sid}.xlsx')
        reference = design[sid]
        assert params.keys() == reference.keys() - FIXED - {'danti_effective'}, sid
        for key, value in params.items():
            assert math.isclose(value, reference[key], rel_tol=2e-15), (sid, key)
        if not (args.output / f'sample_{sid}.csv').is_file():
            jobs.append((sid, reference | params, endpoints[sid], args.output))
    if not jobs:
        print('Requested samples already computed.', flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=setup, initargs=(jobs[0][1],)) as pool:
        pending = [pool.submit(run_sample, job) for job in jobs]
        for i, future in enumerate(as_completed(pending), 1):
            sid, error = future.result()
            if i <= 5 or i % 20 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)}: sample {sid}, day-14 max abs error {error:.3g}', flush=True)


if __name__ == '__main__':
    main()
