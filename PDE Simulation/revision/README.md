# NTM lung infection PDE: manuscript revision

This directory contains the finite-element solver, archived reproduction inputs,
and analysis scripts for the revised experiments. The original workflows are
kept one level above. No solver output collections or compiled libraries are
needed to start a run.

## Environment

From this directory (`PDE Simulation/revision`), use Bash or Zsh:

```sh
conda env create -f reproduce/solver/environment.yml
conda activate lung-pde
source ./setup.sh
bash ./build_ufc.sh
python -m pytest -q tests reproduce/solver/tests/fenics experiments/local_diffusion/test_local_design.py experiments/flux
python reproduce/check_mean_viscosity.py
```

The UFC batch backend needs a C++17 compiler compatible with the local FEniCS
toolchain. Ordinary FEniCS runs do not need this library. Keep a separate output
directory for each mesh, time step, parameter design, and velocity configuration:
several batch runners resume results by file name. By default generated files
go in the ignored `runs/` directory. Run commands below from this directory
after sourcing `setup.sh`.

## Calibration, error reporting, and holdout validation

```sh
SCRIPTS=reproduce/solver/scripts
python "$SCRIPTS/run_original_calibrations.py" inputs/calibration_seeds "$NTM_RUNS" \
    --modes full partial --budget 18 --tangent --workers 4
python "$SCRIPTS/original_tau_summary.py" "$NTM_RUNS" --legacy-root inputs/legacy_xml
python calibration/multistart_182_335/run_multistart.py "$NTM_RUNS" \
    --output "$NTM_RUNS/multistart_182_335" --budget 28 --workers 4
python calibration/multistart_182_335/summarize_multistart.py "$NTM_RUNS" \
    --output "$NTM_RUNS/multistart_182_335"
python calibration/trajectory_figures/plot_calibration_trajectories.py \
    --study "$NTM_RUNS" --output "$NTM_RUNS/trajectory_figures"
```

Use `--sets 756 --workers 1` for an initial check (the trajectory figure command
expects all ten datasets). The original mshr mesh uses `N=32`; calibration
uses a 1/6-day time step and 15 fitted parameters, including separate `DB1`
and `DB2`. The full fit uses all 83 observations. The partial fit uses the
72 observations through day 12 and continues on the same trajectory for the
last 11 observations. The recorded observation times end at day 13.8333.
The per-fit `metrics.json` reports bacterial and macrophage MSREs and optimizer
termination; `trajectory.csv` contains the observed and simulated series.

The checked-in full and partial checkpoints are references from the original-mesh
study, not overwritten by a new run. The archived later refinements for 182
and 335 are in `calibration/multistart_182_335/selected/`; they are used by the
archived treatment/local diffusivity configurations, not substituted into the
earlier spatial velocity comparisons. The two `legacy_parameter_reuse*`
scripts compare the old calibrated parameters without refitting, with separate
`tauB` mappings using `DB1` and `DB2`.

## With and without mucus flow

```sh
python "$SCRIPTS/revision_runs.py" velocity "$NTM_RUNS" \
    --calibration-root . --workers 4
python reproduce/figures/ntm_revision_fig_velocity.py \
    --root "$NTM_RUNS" --out-dir "$NTM_RUNS/velocity_figures"
python experiments/velocity_scaled/run_scaled_velocity.py "$NTM_RUNS/velocity_5x" \
    --study "$NTM_RUNS" --multiplier 5
python experiments/velocity_scaled/render_scaled_velocity.py "$NTM_RUNS/velocity_5x" \
    "$NTM_RUNS/velocity_5x_figures" --study "$NTM_RUNS"
```

The cases share snapshot times 0, 1, and 2 days. The amplified-flow run is
separate from the ordinary 1x comparison and requires its saved fields.
For a calibrated concentration panel, point `--calibration-root` to the run
containing the matching full `trajectory.csv` files. To reproduce the exact
old manuscript panel proportions, `reproduce/figures/run_figures.py --only
velocity` also accepts the original Overleaf source and reference images through
`--velocity-reference`. Those files are not part of this repository.

## Global sensitivity and historical flux sheets

```sh
mkdir -p "$NTM_RUNS/sensitivity"
cp inputs/sensitivity/legacy_design.csv "$NTM_RUNS/sensitivity/"
for branch in fixed_db1 fixed_db2 range_rescaled; do
    destination="$NTM_RUNS/sensitivity/$branch"
    mkdir -p "$destination"
    cp inputs/sensitivity/"$branch"/*.csv "$destination/"
    python "$SCRIPTS/revision_sensitivity.py" "$destination" --solver ufc --workers 4
    python reproduce/excel/export_excel.py --source "$NTM_RUNS" --scope sensitivity \
        --sensitivity-branch "$destination" --output "$NTM_RUNS/xlsx_$branch"
done
```

The three branches replay the same 400 input rows. Only `tauB` changes:
`tauB_old * DB1`, `tauB_old * DB2`, or rank-preserving logarithmic range
rescaling. All store the original sampled `tauM`; the corresponding weak-form
coefficient for these replays includes `DM * tauM`. In the sensitivity sample
table, `danti` is the original multiplier (zeta); `danti_effective` is the
corresponding killing rate. Do not use the multiplier as the treatment-grid axis.

The runner exports day-14 endpoints, daily fields, cumulative diagnostics, and
PRCCs using the solver inputs. To compare the two velocity closures, repeat the
three runs with `--velocity local` in a separate tree of the same layout, then
run `python reproduce/compare_sensitivity.py MEAN_DIRECTORY LOCAL_DIRECTORY`.
This reports PRCCs using both the common original predictors and the applied
solver predictors. Use the separate [earlier interface replay](experiments/interface_paired/README.md)
for like-for-like old-interface comparisons.

The supplementary Excel `flux` tab uses the archived diagnostic and should
not be confused with the integrated PDE exchange term or the PRCC clearance
column. After the range-rescaled export:

```sh
samples="$NTM_RUNS/xlsx_range_rescaled/sensitivity/samples_xlsx"
flux="$NTM_RUNS/flux"
mkdir -p "$flux"
python experiments/flux/compute_flux.py "$samples" "$NTM_RUNS/sensitivity/range_rescaled" "$flux/raw"
python experiments/flux/write_flux_sheets.py "$flux/raw" "$flux/sheets"
python experiments/flux/merge_flux.py stage "$samples" "$flux"
```

Staging checks that the five original sheets have not changed. Use the staged
copies or explicitly run the merge script with `commit` on a working copy.
The `flux` tab has 140 times at 0.1-day intervals and six signed, space-integrated
flux columns.

## Local diffusivity and identifiability

```sh
python experiments/local_diffusion/run_local.py "$NTM_RUNS/local_diffusion" --workers 4
python experiments/local_diffusion/export_local.py "$NTM_RUNS/local_diffusion"
python experiments/local_diffusion/export_local.py "$NTM_RUNS/local_diffusion" --verify
python experiments/identifiability/run_identifiability.py . "$NTM_RUNS/identifiability"
python experiments/identifiability/general_test/run_directional_checks.py \
    . "$NTM_RUNS/identifiability" "$NTM_RUNS/directions"
python experiments/identifiability/run_profile.py . "$NTM_RUNS/profile"
python experiments/identifiability/plot_results.py "$NTM_RUNS/identifiability" \
    "$NTM_RUNS/identifiability_figures" --profile "$NTM_RUNS/profile"
```

The local design has 200 joint, 100 DB1-only, and 100 DB2-only simulations per
dataset (4,000 total). The export preserves the set/branch/sample hierarchy
and workbook tabs; set `NTM_SAMPLE_TEMPLATE` to an archived six-tab workbook
to also retain its exact Excel layout and styles. The identifiability analyses
use the relative-residual Jacobians at the reference fits, nonlinear
perturbations, local objective profiles, and independent initializations.
See the READMEs in [identifiability](experiments/identifiability/) and
[macrophage interface comparison](experiments/interface_macrophage/) for the
optional diagnostics.

## Treatment grids and normalized heatmaps

```sh
python reproduce/run_treatment.py inputs/treatment "$NTM_RUNS/treatment" --workers 4
python reproduce/excel/export_excel.py --source "$NTM_RUNS" --scope treatment \
    --output "$NTM_RUNS/treatment_xlsx"
python reproduce/figures/run_figures.py --only treatment --input-root "$NTM_RUNS" \
    --output "$NTM_RUNS/treatment_figures" --normalization both
```

The two branches use the legacy and revised calibration inputs, respectively,
with the same revised interface and mean-viscosity velocity. Each set has the
same 169 positive-dose coordinates and one zero-dose reference per branch.
The grid axes are reference viscosity `eta` and the effective killing rate
`danti`; internally `zeta = danti / (dB * M(0))`. The paired solver retries
both branches with a smaller common time step if necessary; the CSV retains
that step and the conversion metadata. Excel retains the original eight
columns through `M0`. The heatmaps exclude the zero-dose reference and use a
single colorbar per species column. The numerical comparison of local and
mean viscosity closures is in `reproduce/summarize_treatment.py`.

## Input and output boundaries

- `reproduce/solver/data/` contains the archived ABM trajectories and parameter XML.
- `reproduce/mesh/` contains the original mesh and its provenance.
- `inputs/` contains the coordinate workbook, run designs, and fit seeds.
  `inputs/treatment_grid/` is only the coordinate reference for the heatmaps.
- `calibration/` includes reference fit checkpoints and calibration utilities.
- `experiments/` holds the supplementary and historical comparisons. Keep the
  historical solver and its outputs separate from the manuscript revision.

The repository does not include the manuscript, review correspondence, large
simulation exports, or credentials. The data needed to rerun the experiments
are distinct from the generated analysis outputs.
