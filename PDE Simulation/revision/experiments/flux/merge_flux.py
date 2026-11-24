"""Add the authored flux sheets without reserializing any existing sheet."""

import argparse
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED


MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG = 'http://schemas.openxmlformats.org/package/2006/relationships'
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'
WORKBOOK = 'xl/workbook.xml'
LINKS = 'xl/_rels/workbook.xml.rels'
TYPES = '[Content_Types].xml'
FLUX = 'xl/worksheets/flux.xml'
CHANGED = {WORKBOOK, LINKS, TYPES}
TABS = ['variable', 'B_MUCUS', 'B_TISSUE', 'M_MUCUS', 'M_TISSUE']

for prefix, uri in [('x', MAIN), ('r', REL), ('p', PKG), ('ct', CT)]:
    ET.register_namespace(prefix, uri)


def q(ns, tag):
    return f'{{{ns}}}{tag}'


def xml(book, name):
    return ET.fromstring(book.read(name))


def serialize(root):
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


def sheet_elements(book):
    return xml(book, WORKBOOK).find(q(MAIN, 'sheets'))


def flux_xml(path):
    with ZipFile(path) as book:
        assert [el.get('name') for el in sheet_elements(book)] == ['flux']
        root = xml(book, 'xl/worksheets/sheet1.xml')
        strings = []
        if 'xl/sharedStrings.xml' in book.namelist():
            strings = [''.join(node.itertext()) for node in xml(book, 'xl/sharedStrings.xml')]
    for cell in root.iter(q(MAIN, 'c')):
        cell.attrib.pop('s', None)
        if cell.get('t') == 's':
            value = cell.find(q(MAIN, 'v'))
            text = strings[int(value.text)]
            cell.remove(value)
            cell.set('t', 'inlineStr')
            ET.SubElement(ET.SubElement(cell, q(MAIN, 'is')), q(MAIN, 't')).text = text
    return serialize(root)


def append_flux(original, sheet, output):
    with ZipFile(original) as source:
        wb, links, types = (xml(source, part) for part in (WORKBOOK, LINKS, TYPES))
        sheets = wb.find(q(MAIN, 'sheets'))
        assert [el.get('name') for el in sheets] == TABS, original
        assert FLUX not in source.namelist(), original
        rel_id = 'rIdLungFlux'
        assert rel_id not in {node.get('Id') for node in links}
        sheet_id = max(int(el.get('sheetId')) for el in sheets) + 1
        ET.SubElement(sheets, q(MAIN, 'sheet'), {
            'name': 'flux', 'sheetId': str(sheet_id), q(REL, 'id'): rel_id,
        })
        ET.SubElement(links, q(PKG, 'Relationship'), {
            'Id': rel_id, 'Type': REL + '/worksheet', 'Target': '/' + FLUX,
        })
        ET.SubElement(types, q(CT, 'Override'), {
            'PartName': '/' + FLUX,
            'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml',
        })
        changed = {key: serialize(value) for key, value in
                   zip((WORKBOOK, LINKS, TYPES), (wb, links, types))}
        tmp = output.with_suffix('.tmp')
        subprocess.run(['/bin/cp', '-p', original, tmp], check=True)
        with ZipFile(tmp, 'w', compression=ZIP_DEFLATED, compresslevel=4) as out:
            for name in source.namelist():
                out.writestr(name, changed[name] if name in changed else source.read(name))
            out.writestr(FLUX, flux_xml(sheet))
    tmp.replace(output)


def verify(original, updated, raw):
    with ZipFile(original) as before, ZipFile(updated) as after:
        assert set(after.namelist()) - set(before.namelist()) == {FLUX}, updated
        assert set(before.namelist()) <= set(after.namelist()), updated
        for name in set(before.namelist()) - CHANGED:
            assert before.read(name) == after.read(name), (updated, name)
        sheets = sheet_elements(after)
        assert [el.get('name') for el in sheets] == [*TABS, 'flux'], updated
        assert all(a.attrib == b.attrib for a, b in zip(sheet_elements(before), sheets))
        flux_id = sheets[-1].get(q(REL, 'id'))
        assert any(link.get('Id') == flux_id and link.get('Target') == '/' + FLUX
                   for link in xml(after, LINKS))
        root = xml(after, FLUX)
    rows = root.findall(f'{q(MAIN, "sheetData")}/{q(MAIN, "row")}')
    with raw.open(newline='') as handle:
        expected = list(csv.reader(handle))
    assert len(rows) == len(expected) == 141, updated
    for i, (row, src) in enumerate(zip(rows, expected)):
        assert len(row) == len(src) == 7, (updated, i)
        for j, (cell, val) in enumerate(zip(row, src)):
            assert cell.get('r') == f'{chr(65 + j)}{i + 1}', (updated, i, j)
            contents = cell.findtext(q(MAIN, 'v'))
            if i == 0:
                assert (contents or cell.findtext(f'{q(MAIN, "is")}/{q(MAIN, "t")}')) == val
            else:
                assert cell.get('t', 'n') == 'n', (updated, i, j)
                assert math.isclose(float(contents), float(val), rel_tol=5e-15), (updated, i, j)


def digest(path):
    with path.open('rb') as stream:
        return sha256(stream.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'commit', 'verify'))
    parser.add_argument('samples', type=Path)
    parser.add_argument('work', type=Path)
    parser.add_argument('--ids', nargs='+', type=int)
    args = parser.parse_args()
    ids = args.ids or range(1, 401)
    staged = args.work / 'staged'
    backups = args.work / 'backup'
    manifest = args.work / 'manifest.json'
    if args.operation == 'stage':
        staged.mkdir(exist_ok=True)
        checksums = {}
        for i, sid in enumerate(ids, 1):
            name = f'sample_{sid}.xlsx'
            original, updated = args.samples / name, staged / name
            before = digest(original)
            append_flux(original, args.work / 'sheets' / name, updated)
            verify(original, updated, args.work / 'raw' / f'sample_{sid}.csv')
            assert before == digest(original)
            checksums[name] = [before, digest(updated)]
            if i % 100 == 0 or i == len(ids):
                print(f'Staged and verified {i}/{len(ids)} workbooks.', flush=True)
        manifest.write_text(json.dumps(checksums, indent=2) + '\n')
        return

    assert args.ids is None, 'Only the full batch can be committed.'
    records = json.loads(manifest.read_text())
    assert set(records) == {f'sample_{sid}.xlsx' for sid in ids}
    if args.operation == 'commit':
        backups.mkdir(exist_ok=True)
        for name, (old, new) in records.items():
            assert old == digest(args.samples / name), f'Source changed: {name}'
            assert new == digest(staged / name), f'Staged file changed: {name}'
            if not (backups / name).exists():
                try:
                    os.link(args.samples / name, backups / name)
                except OSError:
                    shutil.copy2(args.samples / name, backups / name)
            assert old == digest(backups / name), f'Backup mismatch: {name}'
        for name, (old, _) in records.items():
            original = args.samples / name
            assert old == digest(original), f'Source changed: {name}'
            (staged / name).replace(original)
    for name, (_, new) in records.items():
        assert new == digest(args.samples / name), f'Final workbook mismatch: {name}'
    print(f'Verified {len(records)} updated workbooks in {args.samples}.', flush=True)


if __name__ == '__main__':
    main()
