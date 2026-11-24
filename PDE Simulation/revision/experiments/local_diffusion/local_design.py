"""Design the paired-diffusivity and one-at-a-time local experiments."""

import csv
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / 'calibration'
APPLIED = ROOT / 'inputs/treatment/recalibrated/applied_parameters.csv'
SETS = (48, 83, 182, 188, 335, 586, 646, 756, 808, 843)
BRANCHES = {'all': (200, ('DB1', 'DB2')), 'DB1': (100, ('DB1',)), 'DB2': (100, ('DB2',))}
SEED = 20261110
COORDINATES = Path(os.environ.get('NTM_COORDINATES', ROOT / 'inputs/sensitivity/coordinate.xlsx'))
TEMPLATE = Path(os.environ.get('NTM_SAMPLE_TEMPLATE', ROOT / 'runs/sample_1.xlsx'))
ABM = ROOT / 'reproduce/solver/data/abm'


def load_inputs():
    with APPLIED.open(newline='') as handle:
        base = {int(row.pop('set_id')): {k: float(v) for k, v in row.items()}
                for row in csv.DictReader(handle)}
    assert set(base) == set(SETS)
    sources = {}
    for sid in SETS:
        parent = (CALIBRATION / 'multistart_182_335/selected'
                  if sid in (182, 335) else CALIBRATION / 'full')
        path = parent / f'set_{sid:03d}/metrics.json'
        report = json.loads(path.read_text())
        assert report['set'] == sid and report['mesh_backend'] == 'original_mshr'
        assert report['train_count'] == 83 and len(report['fitted_parameters']) == 15
        # Keep the five fixed background coefficients from the final treatment setup.
        base[sid].update(report['fitted_parameters'])
        assert all(np.isfinite(list(base[sid].values())))
        assert base[sid]['DB1'] > 0. and base[sid]['DB2'] > 0.
        sources[sid] = path
    return sources, base


def design_for(sid, base):
    result = []
    for branch_index, (branch, (count, varied)) in enumerate(BRANCHES.items()):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, sid, branch_index]))
        multipliers = np.exp(rng.uniform(np.log(.1), np.log(10.), (count, len(varied))))
        for i, values in enumerate(multipliers, 1):
            updates = {k: base[k] * float(v) for k, v in zip(varied, values)}
            result.append(dict(set_id=sid, branch=branch, sample_id=i, **(base | updates),
                               eta=1., zeta=1.))
    return result


def write_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(root):
    sources, bases = load_inputs()
    design = []
    for sid in SETS:
        with np.load(ABM / f'data_{sid}.npz', allow_pickle=False) as record:
            b0, m0 = float(record['B'][0]), float(record['M'][0])
        for row in design_for(sid, bases[sid]):
            design.append(row | {'B0_ABM': b0, 'M0_ABM': m0,
                                 'danti_effective': row['dB'] * m0})
    for sid in SETS:
        subset = [r for r in design if r['set_id'] == sid]
        assert len(subset) == 400
        assert len({(r['DB1'], r['DB2']) for r in subset}) == 400
    write_rows(root / 'parameter_set_samples.csv', design)
    write_rows(root / 'baseline_parameters.csv', [
        {'set_id': sid, **bases[sid], 'fitted_parameter_source': str(sources[sid]),
         'fixed_parameter_source': str(APPLIED)} for sid in SETS
    ])
    return bases, design
