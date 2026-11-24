"""Run the local diffusivity design using the revised mean-viscosity PDE."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import cos, pi
import multiprocessing as mp
import os
from pathlib import Path
from time import monotonic
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np

from local_design import BRANCHES, COORDINATES, SETS, prepare


FIELDS = ('bacteria_mucus', 'bacteria_tissue', 'macrophage_mucus', 'macrophage_tissue')
HEADER = ('time (day)', 'B mucus2tissue', 'M mucus2tissue', 'B tissue2mucus',
          'M tissue2mucus', 'B right', 'M right')
TIMES = np.arange(1, 141) / 10.
MODEL = None


def read_coordinates():
    with ZipFile(COORDINATES) as book:
        return {region: np.asarray([[float(c.findtext('{*}v')) for c in row] for row in
                ET.fromstring(book.read(f'xl/worksheets/sheet{i}.xml')).findall(
                    './{*}sheetData/{*}row')[1:]], dtype=float)
                for i, region in enumerate(('mucus', 'tissue'), 1)}


def context(sid, base):
    from dolfin import SpatialCoordinate, TestFunctions, as_vector, assemble, avg, dot, grad
    from scipy.spatial import cKDTree
    from calibrate_conservative import concentration_weights
    from lung_pde.mesh import INTERFACE_FLAG, RIGHT_MUCUS_FLAG
    from lung_pde.ufc_batch import BatchNewton
    from revision_runs import make_model
    from revision_sensitivity import snapshot_indices

    model = make_model(sid, 'treatment', base, 32, .1, interpolate_profile=True,
                       polynomial_velocity=True, mean_viscosity_velocity=True,
                       macrophage_diffusivity_in_flux=False)
    model.eta.assign(1.)
    model.danti.assign(1.)
    solver = BatchNewton(model, os.environ['NTM_UFC_LIBRARY'])
    index, geo = snapshot_indices(model)
    for region, xy in read_coordinates().items():
        native = geo[f'{region}_xy']
        dist, order = cKDTree(native).query(xy)
        assert dist.max() < 1e-12 and len(np.unique(order)) == len(native) == len(xy)
        for field in FIELDS:
            if field.endswith('_' + region):
                index[field] = index[field][order]

    # Preassemble linear diagnostic weights; no change to the PDE residual.
    mesh = model.mesh
    r = mesh.ds(RIGHT_MUCUS_FLAG)
    g = mesh.dS(INTERFACE_FLAG)
    y = SpatialCoordinate(mesh.mesh)[1]
    shape = as_vector(((y - .1)**2, 0.))
    b, bt, m, mt, _ = TestFunctions(model.w.function_space())
    forms = ((avg(bt) - avg(b))*g, (avg(mt) - avg(m))*g,
             dot(mesh.normal, shape*b)*r, dot(mesh.normal, shape*m)*r,
             -dot(mesh.normal, grad(b))*r, -dot(mesh.normal, grad(m))*r)
    diagnostic = np.array([assemble(form).get_local() for form in forms])
    return model, solver, index, concentration_weights(model), diagnostic


def literal_flux(model, time):
    from dolfin import assemble, avg, dot, grad, split
    from lung_pde.mesh import INTERFACE_FLAG, RIGHT_MUCUS_FLAG
    from lung_pde.velocity_fields import PolynomialVelocity

    b, bt, m, mt, _ = split(model.w)
    p, mesh = model.parameters, model.mesh
    interface, right = mesh.dS(INTERFACE_FLAG), mesh.ds(RIGHT_MUCUS_FLAG)
    baseline = PolynomialVelocity(mesh.mesh, treatment=True)
    baseline.t = time
    return np.array([assemble(form) for form in (
        p['tauB']*(avg(bt) - avg(b))*interface, p['tauM']*(avg(mt) - avg(m))*interface,
        dot(mesh.normal, baseline.form*b - p['DB1']*grad(b))*right,
        dot(mesh.normal, baseline.form*m - p['DB1']*grad(m))*right,
    )])


def integrate(ctx, params, dt, pilot=False):
    model, solver, index, weights, diagnostic = ctx
    model.assign_parameters({'DB1': params['DB1'], 'DB2': params['DB2']})
    model.dt.assign(dt)
    assert model.abm.initial_bacteria == params['B0_ABM']
    assert model.abm.initial_macrophages == params['M0_ABM']
    assert model.effective_danti == params['danti_effective']
    model.reset()
    solver.factor = None
    state = model.w.vector().get_local()
    capture = {k: [state[index[k]]] for k in FIELDS}
    trajectory = [(weights @ state)[:4]]
    flux = np.empty((140, 7))
    flux[:, 0] = TIMES
    stride = round(.1/dt)
    for k in range(1, 140 * stride + 1):
        time = k * dt
        state = solver.step(time, state, lag=True)
        if k % stride:
            continue
        i = k // stride
        e1, e2, b_adv, m_adv, b_diff, m_diff = diagnostic @ state
        adv_factor = .5 * 1.44 * (1. - cos(2. * pi * time))
        b, m = e1*params['tauB'], e2*params['tauM']
        rb, rm = adv_factor*b_adv + params['DB1']*b_diff, adv_factor*m_adv + params['DB1']*m_diff
        flux[i-1, 1:] = b, m, -b, -m, rb, rm
        if i % 10 == 0:
            for f in FIELDS:
                capture[f].append(state[index[f]])
            trajectory.append((weights @ state)[:4])
        if pilot and i in (1, 5, 140):
            vector = model.w.vector()
            vector.set_local(state)
            vector.apply('insert')
            np.testing.assert_allclose(literal_flux(model, time), (b, m, rb, rm),
                                       atol=2e-16, rtol=2e-8)
    result = {k: np.asarray(v) for k, v in capture.items()}
    result.update(flux=flux, concentrations=np.asarray(trajectory), dt_day=np.asarray(dt))
    if not all(np.isfinite(v).all() for v in result.values()):
        raise RuntimeError('Nonfinite sample output')
    return result


def chunk(job):
    global MODEL
    sid, base, rows, root, pilot = job
    if MODEL is None or MODEL[0] != sid:
        MODEL = sid, context(sid, base)
    completed = []
    for row in rows:
        branch, sample_id = row['branch'], row['sample_id']
        output = root / str(sid) / branch / f'sample_{sample_id}.npz'
        if output.exists() and not pilot:
            completed.append((sid, branch, sample_id, 'reused'))
            continue
        for dt in (.1, .05, .025, .0125):
            try:
                values = integrate(MODEL[1], row, dt, pilot)
            except RuntimeError as error:
                message = str(error)[:300]
                print(f'{sid}/{branch}/{sample_id} failed at dt={dt:g}: {message}', flush=True)
                continue
            break
        else:
            completed.append((sid, branch, sample_id, 'failed: ' + message))
            continue
        temp = output.with_suffix('.tmp')
        with temp.open('wb') as handle:
            np.savez(handle, **values)
        temp.replace(output)
        completed.append((sid, branch, sample_id, dt))
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--sets', nargs='+', type=int, choices=SETS)
    args = parser.parse_args()
    args.work.mkdir(exist_ok=True, parents=True)
    bases, records = prepare(args.work)
    root = args.work / 'raw'
    jobs = []
    sets = args.sets or SETS
    for sid in sets:
        for branch in BRANCHES:
            (root / str(sid) / branch).mkdir(exist_ok=True, parents=True)
        rows = [r for r in records if r['set_id'] == sid]
        if args.pilot:
            rows = [next(r for r in rows if r['branch'] == b) for b in BRANCHES]
        rows = [r for r in rows if args.pilot or not (root / str(sid) / r['branch'] /
                f"sample_{r['sample_id']}.npz").is_file()]
        jobs.extend((sid, bases[sid], rows[i:i+10], root, args.pilot) for i in range(0, len(rows), 10))
    total = sum(len(j[2]) for j in jobs)
    start, n, failures = monotonic(), 0, []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(chunk, job) for job in jobs]
        for future in as_completed(futures):
            batch = future.result()
            n += len(batch)
            failures.extend(x for x in batch if str(x[3]).startswith('failed'))
            if args.pilot or n % 50 <= 9 or n == total:
                print(f'{n}/{total} cases processed in {monotonic()-start:.0f}s; '
                      f'last {batch[-1]}; {len(failures)} failed', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} unresolved cases: {failures}')


if __name__ == '__main__':
    main()
