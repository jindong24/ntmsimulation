# Excel exports

Use `export_excel.py` after the simulation has saved its source data, as
described in the [revision README](../../README.md).

- Each sensitivity sample has `variable`, `B_MUCUS`, `B_TISSUE`,
  `M_MUCUS`, and `M_TISSUE` tabs. Columns 0 through 14 represent days.
  Rows match `coordinate.xlsx`, not the solver's internal node order.
- The summary includes the parameter workbook, the sampled parameter CSV,
  and the coordinate workbook.
- Each treatment workbook has the original eight result columns through
  `M0`. The CSV retains additional diagnostics and conversion metadata.

The exporter uses openpyxl and does not modify the source data.
