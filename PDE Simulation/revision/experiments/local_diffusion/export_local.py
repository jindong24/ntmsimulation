"""Populate the verified original-format template with local sensitivity results."""

import argparse
import csv
from copy import copy, deepcopy
from pathlib import Path
from shutil import copy2
import sys
from time import monotonic, sleep
from zipfile import ZipFile

from lxml import etree as ET
import numpy as np

from local_design import BRANCHES, COORDINATES, SETS, TEMPLATE, write_rows
from run_local import FIELDS, HEADER, TIMES


NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
VARIABLE = 'xl/worksheets/sheet1.xml'
FLUX = 'xl/worksheets/sheet6.xml'
SAME_AS_TEMPLATE = {'[Content_Types].xml', 'xl/workbook.xml', 'xl/styles.xml'}
FLAT_FIELDS = dict(zip(FIELDS, ('B_MUCUS', 'B_TISSUE', 'M_MUCUS', 'M_TISSUE')))
SIZES = (838, 842, 838, 842)


class GeneratedWorkbook:
    """Use the same table layout when no original file template is supplied."""

    def matches(self, path):
        from openpyxl import load_workbook

        try:
            with path.open("rb") as stream:
                book = load_workbook(stream, read_only=True)
                try:
                    return book.sheetnames == ['variable', *FLAT_FIELDS.values(), 'flux']
                finally:
                    book.close()
        except (OSError, ValueError):
            return False

    def write(self, record, npz, dest):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'reproduce/excel'))
        from write_workbook import write_workbook

        sheets = [{'name': 'variable', 'rows': [
            ('Variable', 'Value'), *[(k, float(record[k])) for k in BRANCHES[record['branch']][1]]
        ]}]
        with np.load(npz, allow_pickle=False) as raw:
            for field, name in FLAT_FIELDS.items():
                assert raw[field].shape == (15, SIZES[FIELDS.index(field)])
                sheets.append({'name': name, 'rows': [
                    list(map(str, range(15))), *raw[field].T.tolist()
                ]})
            assert raw['flux'].shape == (140, 7) and np.array_equal(raw['flux'][:, 0], TIMES)
            assert np.array_equal(raw['flux'][:, 1:3], -raw['flux'][:, 3:5])
            sheets.append({'name': 'flux', 'rows': [HEADER, *raw['flux'].tolist()]})
        write_workbook(dest, sheets)


def export_template():
    return ExcelTemplate() if TEMPLATE.is_file() else GeneratedWorkbook()


def vals(row):
    return [c.findtext(NS + 'v') or c.findtext(f'{NS}is/{NS}t') for c in row]


def slots(rows):
    output = []
    for row in rows:
        for cell in row:
            assert cell.get('t', 'n') == 'n' and cell.find(NS + 'f') is None
            output.append(cell.find(NS + 'v'))
    assert all(node is not None for node in output)
    return output


class ExcelTemplate:
    def __init__(self):
        with ZipFile(TEMPLATE) as source:
            self.parts = [(entry, source.read(entry)) for entry in source.infolist()]
            self.crcs = {name: source.getinfo(name).CRC for name in SAME_AS_TEMPLATE}
        self.variable, self.fields = {}, {}
        for entry, data in self.parts:
            name = entry.filename
            if name == VARIABLE:
                for branch, (_, varied) in BRANCHES.items():
                    root = ET.fromstring(data)
                    rows = root.find(NS + 'sheetData')
                    assert vals(rows[0]) == ['Variable', 'Value']
                    old = {vals(row)[0]: deepcopy(row) for row in rows[1:]}
                    for row in list(rows)[1:]:
                        rows.remove(row)
                    entries = []
                    for i, key in enumerate(varied, 2):
                        row = old[key]
                        row.set('r', str(i))
                        for j, cell in enumerate(row):
                            cell.set('r', f'{chr(65+j)}{i}')
                        entries.append(row[1].find(NS + 'v'))
                        rows.append(row)
                    root.find(NS + 'dimension').set('ref', f'A1:B{1+len(varied)}')
                    self.variable[branch] = (root, entries)
            elif name == FLUX or name.startswith('xl/worksheets/sheet'):
                root = ET.fromstring(data)
                rows = root.find(NS + 'sheetData')
                if name == FLUX:
                    assert len(rows) == 141 and vals(rows[0]) == list(HEADER)
                    key = 'flux'
                else:
                    idx = int(Path(name).stem[5:]) - 2
                    assert len(rows) == SIZES[idx] + 1 and vals(rows[0]) == list(map(str, range(15)))
                    key = FIELDS[idx]
                self.fields[name] = (key, root, slots(rows[1:]))
        assert len(self.fields) == 5 and len(self.variable) == 3

    def matches(self, path):
        with ZipFile(path) as book:
            return all(book.getinfo(n).CRC == crc for n, crc in self.crcs.items())

    def write(self, record, npz, dest):
        branch = record['branch']
        var, var_slots = self.variable[branch]
        for node, name in zip(var_slots, BRANCHES[branch][1], strict=True):
            node.text = repr(float(record[name]))
        xml = {VARIABLE: ET.tostring(var, encoding='utf-8', xml_declaration=True)}
        with np.load(npz, allow_pickle=False) as raw:
            assert raw['concentrations'].shape == (15, 4)
            for part, (key, root, values) in self.fields.items():
                grid = raw[key]
                assert grid.size == len(values) and np.isfinite(grid).all(), (npz, key)
                if key == 'flux':
                    assert grid.shape == (140, 7) and np.array_equal(grid[:, 0], TIMES)
                    assert np.array_equal(grid[:, 1:3], -grid[:, 3:5])
                else:
                    grid = grid.T
                    assert grid.shape == (SIZES[FIELDS.index(key)], 15)
                for node, value in zip(values, grid.ravel().tolist(), strict=True):
                    node.text = repr(value)
                xml[part] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix('.tmp')
        with ZipFile(temp, 'w') as output:
            for entry, template in self.parts:
                output.writestr(copy(entry), xml.get(entry.filename, template), compresslevel=3)
        temp.replace(dest)


def locations(work, record):
    suffix = Path(record['set_id']) / record['branch'] / f"sample_{record['sample_id']}"
    return work / 'raw' / suffix.with_suffix('.npz'), work / 'deliver' / suffix.with_suffix('.xlsx')


def read_design(work):
    with (work / 'parameter_set_samples.csv').open(newline='') as handle:
        design = list(csv.DictReader(handle))
    assert len(design) == 4000
    return design


def stage(work, pilot, follow):
    template = export_template()
    design = read_design(work)
    if pilot:
        design = [r for r in design if r['set_id'] == '182' and r['sample_id'] == '1']
    remaining = [(r, *locations(work, r)) for r in design]
    start = monotonic()
    while remaining:
        still_missing = []
        for row, raw, xlsx in remaining:
            if xlsx.is_file() and template.matches(xlsx):
                continue
            if not raw.is_file():
                still_missing.append((row, raw, xlsx))
                continue
            template.write(row, raw, xlsx)
        previous = len(remaining)
        remaining = still_missing
        if len(remaining) != previous:
            print(f'{len(design)-len(remaining)}/{len(design)} workbooks staged', flush=True)
        if not follow or not remaining:
            break
        if monotonic() - start > 5400:
            raise TimeoutError('Incomplete source simulations after 90 minutes.')
        sleep(4.)


def final_check(work):
    from openpyxl import load_workbook

    design = read_design(work)
    template = export_template()
    for sid in SETS:
        for branch, (count, varied) in BRANCHES.items():
            folder = work / 'deliver' / str(sid) / branch
            assert {p.name for p in folder.glob('*.xlsx')} == {f'sample_{i}.xlsx' for i in range(1,count+1)}
            assert all(template.matches(p) for p in folder.glob('*.xlsx'))
            for i in (1, count):
                row = next(r for r in design if r['set_id'] == str(sid) and
                           r['branch'] == branch and r['sample_id'] == str(i))
                raw, xlsx = locations(work, row)
                book = load_workbook(xlsx, read_only=True, data_only=True)
                assert book.sheetnames == ['variable', *FLAT_FIELDS.values(), 'flux']
                header, *pairs = book['variable'].values
                assert header == ('Variable', 'Value') and [k for k, _ in pairs] == list(varied)
                np.testing.assert_allclose([v for _, v in pairs], [float(row[k]) for k in varied],
                                           rtol=5e-15, atol=0)
                with np.load(raw, allow_pickle=False) as data:
                    for field, sheet in FLAT_FIELDS.items():
                        header, *values = list(book[sheet].values)
                        assert list(header) == list(map(str, range(15)))
                        np.testing.assert_allclose(values, data[field].T, rtol=5e-15, atol=0)
                    flux_header, *flux = list(book['flux'].values)
                    assert list(flux_header) == list(HEADER)
                    np.testing.assert_allclose(flux, data['flux'], rtol=5e-15, atol=0)
                book.close()
        print(f'Set {sid}: 400 workbooks verified, including complete readback of six samples.', flush=True)
    rows, initial = [], {}
    for row in design:
        raw, _ = locations(work, row)
        with np.load(raw, allow_pickle=False) as result:
            start = result['concentrations'][0]
            np.testing.assert_allclose(start, initial.setdefault(row['set_id'], start), rtol=1e-14)
            # The archived interpolation and projection introduce a small load offset.
            np.testing.assert_allclose(start[[0, 2]],
                                      [float(row['B0_ABM']), float(row['M0_ABM'])],
                                      rtol=1e-3, atol=1e-15)
            rows.append({key: row[key] for key in ('set_id', 'branch', 'sample_id')} | {
                'dt_day': float(result['dt_day']), **dict(zip(FLAT_FIELDS.values(), result['concentrations'][-1])),
            })
    dest = work / 'deliver' / 'metadata'
    dest.mkdir(exist_ok=True)
    write_rows(dest / 'day14_summary.csv', rows)
    for name in ('parameter_set_samples.csv', 'baseline_parameters.csv'):
        copy2(work / name, dest / name)
    copy2(Path(__file__).with_name('README.txt'), dest / 'README.txt')
    copy2(COORDINATES, dest / 'coordinate.xlsx')
    print(f'All {len(rows)} simulations present; {sum(row["dt_day"] != .1 for row in rows)} used a smaller time step.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        final_check(args.work)
    else:
        stage(args.work, args.pilot, args.follow)


if __name__ == '__main__':
    main()
