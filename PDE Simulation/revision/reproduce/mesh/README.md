# Original mshr mesh and sample preflight

This is a preflight, not the final rerun. The original source files,
historical Excel and existing revision outputs were not overwritten.

`original_mshr.h5` was written by executing the mesh block of
`Documents/Research/KAUST/code_gpu/lung/test_sample.py` with
mshr 2019.1.0, using the source's `N=32` and region marking.
The file includes the actual mesh and cell markers, not a triangulation
reconstructed from the Excel coordinates.

- 1643 distinct vertices, 3142 triangles.
- 838 mucus and 842 tissue coordinates match the archived
  `coordinate.xlsx` to within 5.551e-17.
- Source mesh cell array SHA-256:
  `305b8ba984ff0356a520741c72b9e8eac6b803a408f208f27cc8af7e65fcb9aa`.
- The older archive only saves coordinates. Of the triangles in a separate
  Delaunay reconstruction, 1565/1567 mucus and 1575/1575 tissue triangles
  match the generated mshr mesh. The separate reconstruction was not used
  to solve the PDE.

`replay_original_sample.py` uses the original Python script and original
utility functions. The pilot replaces random sampling with sample 1's saved
inputs, loads the exported mshr mesh without remeshing, and captures arrays
in an NPZ rather than writing over the Excel files. The legacy pilot retains
the original initializations, weak form, solver settings and 0.1-day time
step. The optional conservative branch replaces only the four cell-species
exchange integrals, leaving the mucus-density interface term unchanged.

Legacy sample 1: the initial mucus bacteria and macrophage fields agree
with the archive to within 2.3e-15 relative to their peak values.
The maximum absolute nodal differences relative to each day's field peak
over days 0 to 14 are 7.809e-5 (mucus bacteria), 8.749e-4 (tissue bacteria),
1.139e-7 (mucus macrophages), and 4.188e-6 (tissue macrophages).
The stricter per-cell check in `verify_original_sample.py` flags the tissue
bacteria difference. This is not a bit-for-bit historical solver replay.

The conservative sample-1 output is a smoke test with the previous
revision's provisional permeability inputs, not a new calibration or
an updated sensitivity result. The ten calibrations and validation fits,
all sensitivity runs, velocity figures, and treatment grid are not
completed in this preflight.
