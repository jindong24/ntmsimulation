"""Normalize OPC manifests for spreadsheet readers requiring default namespaces."""

from copy import copy
from pathlib import Path
import sys
from zipfile import ZipFile

from lxml import etree as ET


MANIFESTS = {'[Content_Types].xml', 'xl/_rels/workbook.xml.rels'}


def normalized(name, data):
    if name not in MANIFESTS:
        return data
    root = ET.fromstring(data)
    namespace = ET.QName(root).namespace
    if root.nsmap.get(None) == namespace:
        return data
    target = ET.Element(root.tag, nsmap={None: namespace}, attrib=root.attrib)
    for item in root:
        assert len(item) == 0 and ET.QName(item).namespace == namespace
        ET.SubElement(target, item.tag, attrib=item.attrib)
    return ET.tostring(target, encoding='utf-8', xml_declaration=True)


def normalized_copy(source, target):
    with ZipFile(source) as original, ZipFile(target, 'w') as updated:
        for entry in original.infolist():
            updated.writestr(copy(entry), normalized(entry.filename, original.read(entry)), compresslevel=3)


if __name__ == '__main__':
    normalized_copy(*map(Path, sys.argv[1:]))
