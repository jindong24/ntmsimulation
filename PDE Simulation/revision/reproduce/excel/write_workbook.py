"""Write the existing workbook tables using a public Excel library."""

from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.utils import get_column_letter


SAMPLES = {"variable", "B_MUCUS", "B_TISSUE", "M_MUCUS", "M_TISSUE"}
THIN = Side(style="thin", color="000000")


def write_workbook(path, sheets):
    path.parent.mkdir(parents=True, exist_ok=True)
    book = Workbook(write_only=True)
    for spec in sheets:
        name, rows = spec["name"], iter(spec["rows"])
        tab = book.create_sheet(name)
        header = next(rows)
        for i, item in enumerate(header, 1):
            width = 18
            if name == "Basic Setting" and i in (3, 5):
                width = 5
            elif item == "provenance":
                width = 55
            elif name == "flux":
                width = 14 if i == 1 else 21
            tab.column_dimensions[get_column_letter(i)].width = width
        cells = []
        for i, value in enumerate(header):
            cell = WriteOnlyCell(tab, value=value)
            if name != "Sheet" and (name != "Basic Setting" or i == 0):
                cell.font = Font(bold=True)
            if name in SAMPLES:
                cell.alignment = Alignment(horizontal="center")
                cell.border = Border(top=THIN, bottom=THIN, left=THIN, right=THIN)
            cells.append(cell)
        tab.append(cells)
        for row in rows:
            tab.append(row)
    scratch = path.with_suffix(".tmp")
    book.save(scratch)
    scratch.replace(path)
