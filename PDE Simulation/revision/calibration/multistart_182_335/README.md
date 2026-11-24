# Sets 182 and 335: original-mesh multistart refit

Three starts per set are evaluated with the unchanged conservative
interface, original mshr mesh and ABM initial state, 1/6-day time step,
and 83 full-trajectory mucus observations. The 15 parameter bounds,
fixed coefficients, relative residual objective and discrete tangent
are unchanged from the original-mesh full fit.

- `near_current`: roughly 20% perturbations of the existing coefficients,
  directed inward at the parameter bounds, plus a small alpha1 shift.
- `legacy_mucus_side`: historical full-fit parameters; the seed uses
  original DB1 times tauB and original DM times tauM as effective
  conservative coefficients before clipping into the same fit bounds.
- `low_tissue_diffusion`: a different set of bacterial transport and
  reaction starts with the current macrophage coefficients as seeds.

`runs/` contains each log, metrics file, and trajectory. `all_fits.csv`
compares all starts and the previous checkpoint. Selection uses the
joint training MSRE and includes the previous fit as a candidate.
`selected_comparison.csv`, `best_parameters.csv` and `selected/` record
the selected candidates. The after-day-12 metric here is **in-sample**;
the independent training window of the previous day-12 holdout runs is
not changed, and neither are any downstream simulations or figures.

See the [revision README](../../README.md) for the FEniCS environment
and commands. Use the same study path and output directory for both
commands. The study path must contain the original-mesh full-fit trajectories
for the summary. The checked-in `selected/` entries are reference parameters;
new comparisons write to the separate output directory.
