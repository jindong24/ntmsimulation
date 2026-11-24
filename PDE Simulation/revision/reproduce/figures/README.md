# Reproduce the figures

Start with the [revision workflow](../../README.md). The velocity dashboard
can be rendered directly from paired snapshots. For exact alignment with the
original manuscript panels, use `run_figures.py --only velocity` and supply
the original manuscript directory via `--velocity-reference`. This mode needs
Times New Roman, PyMuPDF, and `pdflatex`.

The treatment renderer uses the matched effective-dose coordinates in
`inputs/treatment_grid`. The `paper` option standardizes each row within
each setup. The `shared` option standardizes against the two setups together
and keeps the color ranges matched between them. Each species column has
one colorbar. Table and figure generation do not run the PDE again.
