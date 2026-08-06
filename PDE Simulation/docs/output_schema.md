# Input and Output Schema

## Input Data

Each `data/abm/data_<set>.npz` archive contains exactly:

| Key | Shape | Meaning |
| --- | --- | --- |
| `B` | `(84,)` | mucus bacteria; element 0 is the initial condition and elements 1-83 are targets |
| `M` | `(84,)` | mucus macrophages with the same convention |
| `t` | `(83,)` | exact target grid from `1/6` through `83/6` day |

All ABM arrays must be finite, real numeric arrays. Parameter XML accepts the
supplied legacy hierarchy with a `dolfin` root and one
`parameters name="problem_parameters"` child, as well as the direct
`parameters` root written by the calibration workflow. Every child must be a
`parameter` with exactly `key`, `type="double"`, and `value`; duplicate or
non-finite values are rejected. The reader requires the 15 calibrated and five
fixed code keys. The only accepted
extra legacy keys are `dt`, `T`, `lambdaB1`, `lambdaB2`, `dm1`, `dm2`, `loss`,
and `Kbi`.

## Baseline and Velocity Results

A completed simulation directory contains `concentrations.npz`,
`metadata.json`, and `spatial.npz` when snapshots were requested.

`concentrations.npz` contains exactly six one-dimensional arrays of common
shape `(n_time,)`:

`time`, `bacteria_mucus`, `bacteria_tissue`, `macrophage_mucus`,
`macrophage_tissue`, and `mucus_density`.

Every concentration, snapshot-time, coordinate, and bacteria/macrophage state
array must contain finite real numbers. Invalid arrays are rejected before a
new result generation is committed; an existing complete generation remains
unchanged.

`time` is in days and starts at zero. The fixed step is retained; if the
requested final time is not an integer number of steps, `actual_final_time`
is the last full step before it. Concentrations are the legacy quantity
`50 * integral(state over region)`. No additional physical unit conversion is
defined by the source code.

`spatial.npz` contains exactly:

| Key | Shape |
| --- | --- |
| `time` | `(n_snapshot,)` |
| `mucus_xy` | `(n_mucus_vertices, 2)` |
| `mucus_triangles` | `(n_mucus_cells, 3)` integer vertex indices |
| `tissue_xy` | `(n_tissue_vertices, 2)` |
| `tissue_triangles` | `(n_tissue_cells, 3)` integer vertex indices |
| `bacteria_mucus` | `(n_snapshot, n_mucus_vertices)` |
| `bacteria_tissue` | `(n_snapshot, n_tissue_vertices)` |
| `macrophage_mucus` | `(n_snapshot, n_mucus_vertices)` |
| `macrophage_tissue` | `(n_snapshot, n_tissue_vertices)` |

Coordinates use the model geometry scale. State arrays are finite-element
vertex values and are not multiplied by the concentration integral factor.

`metadata.json` is an object with these fields:

`set_id`, `mode`, `parameter_source`, `parameter_source_kind`,
`parameter_source_sha256`, `dt`, `final_time`, `actual_final_time`,
`requested_snapshot_times`, `resolution`, `mesh_backend`, `solver_settings`,
`command_arguments`, `package_version`, `created_at_utc`, `initial_bacteria`,
and `initial_macrophages`.

`mesh_backend` is the resolved backend. `solver_settings` contains
`relative_tolerance`, `report`, `linear_solver`, `preconditioner`, and
`maximum_iterations`. Treatment-mode use of this general result object also
adds `eta`, `input_danti`, and `effective_danti`, although the public sweep
persists the per-case schema below.

## Calibration Results

Each run is isolated at `outputs/calibration/set_<id>/<run_id>/`.

Run-level `metadata.json` fields are:

`set_id`, `run_id`, `created_at_utc`, `command`, `pipeline`,
`pipeline_sha256`, `initial_parameters`, `initial_parameters_sha256`,
`abm_data`, `abm_data_sha256`, `calibration_dt`, `observation_count`, `stages`,
`active_parameters`, `resolution`, `requested_mesh_backend`, `mesh_backend`,
and `max_evaluations`.

`requested_mesh_backend` is the CLI/API selection (`auto`, `legacy_mshr`, or
`rectangle`). `mesh_backend` is the resolved backend (`legacy_mshr` or
`rectangle`) after successful model construction. It is `null` when model
construction fails before backend resolution; the implementation never guesses
a resolved backend from the request.

Run-level `status.json` always contains `status` and `calibration_dt`.
Running status adds `started_at_utc`. A run that reaches the stage runner adds
`completed_at_utc` and `stages_completed` at termination, plus `error` when a
stage fails. A failure while constructing the simulator instead adds
`completed_at_utc` and `error` but has no `stages_completed` field. Each
`error` is an object with `type` and `message`.

Every stage directory contains:

| File | Schema |
| --- | --- |
| `parameters_input.xml` | exact stage input checkpoint |
| `parameters.xml` | successful best checkpoint, or unchanged input on failure |
| `evaluations.jsonl` | one object per evaluation: `evaluation`, `parameters`, `loss` |
| `trajectory.npz` | when an evaluation exists: `time`, `bacteria_mucus`, `macrophage_mucus`, each `(83,)`; time is days 1/6 through 83/6 |
| `metrics.json` | optimizer and checkpoint metadata described below |
| `status.json` | running or terminal stage status |

`metrics.json` fields are `objective`, `final_loss`, `best_evaluated_loss`,
`best_evaluated_parameters`, `evaluations`, `termination`,
`optimizer_success`, `optimizer_status`, `optimizer_message`,
`optimizer_iterations`, `optimizer_reported_evaluations`,
`optimizer_settings`, `max_evaluations`, `bound_projection`,
`checkpoint_source`, and `trajectory_kind`. Failure additionally has `error`.
`optimizer_settings` has `method`, `maxiter`, `gtol`, and `ftol`.
Each `bound_projection` entry has `name`, `original`, and `projected`.

Stage `status.json` fields common to running and terminal states are `status`,
`stage`, `objective`, `active_parameters`, `calibration_dt`,
`observation_count`, `bound_projection`, and `started_at_utc`. A successful
terminal stage adds `completed_at_utc`, `evaluations`, `final_loss`, and
`termination`. A failed stage adds `completed_at_utc`, `evaluations`,
`termination`, `best_evaluated_loss`, and `error`.

## Treatment Results

Case files are named from the exact IEEE-754 bit patterns of `eta` and
`danti_input`: `cases/eta_<hex>__danti_<hex>.json`. Each JSON object contains
exactly:

`schema_version`, `case_key`, `set_id`, `eta`, `danti_input`,
`danti_effective`, `B0`, `M0`, `dB`, `B_MUCUS`, `B_TISSUE`, `M_MUCUS`,
`M_TISSUE`, `parameter_source`, `parameter_source_kind`,
`parameter_source_sha256`, `runtime_revision`, `mesh_backend`,
`resolved_mesh_backend`,
`resolution`, `dt`, `final_time`, and `status`.

`schema_version` is `2`; successful case status is `success`.
`danti_effective = danti_input * M0 * dB`, where `M0` is the ABM initial
macrophage concentration and `dB` is read from the calibrated XML. `B_MUCUS`,
`B_TISSUE`, `M_MUCUS`, and `M_TISSUE` are the four final integrated
concentrations. `final_time` is the requested horizon; the legacy fixed-step
loop stops at its last full step when the horizon is not divisible by `dt`.

`runtime_revision` is a portable SHA-256 over a fixed, repository-root-relative
source manifest. The manifest covers the treatment sweep entry point
(`scripts/sweep_danti_eta.py`) and the checked-in numerical runtime, form,
data/parameter reader, mesh, model, simulation, and solver Python sources under
`src/lung_pde`. Its computation uses only those stable relative names and file
content, with CRLF and CR line endings normalized to LF; it does not depend on
Git, absolute paths, timestamps, or dirty-tree state. Parent and worker
processes carry the same revision and workers verify it before and after
solving.

For a full grid, `treatment.csv` has the same columns in the same order except
that `schema_version` is omitted. It contains 170 rows: the zero reference
first, then 169 positive cases ordered by `eta` and `danti_input`. All rows
must share the run-identity fields `set_id`, `parameter_source_sha256`, `B0`,
`M0`, `dB`, `runtime_revision`, `mesh_backend`, `resolved_mesh_backend`,
`resolution`, `dt`, and `final_time`. Missing or mismatched revisions are never
silently skipped or aggregated; use a new output root or explicit `--overwrite`.

## Verification Report

`outputs/verification/report.json` contains:

- `schema_version`: integer `1`;
- `overall_status`: `pass` unless any check failed, otherwise `fail`;
- `checks`: ordered objects with exactly `name`, `status`, and `detail`.

Check status is one of `pass`, `fail`, or `skip`. No timestamp, machine-local
path, or test duration is stored, so identical check outcomes produce
byte-identical report JSON. A subprocess failure is recorded only as
`command failed: <normalized JSON argument list>; exit=<code>`: Python
interpreters become `python`, repository paths are relative, and other
absolute paths become `<absolute-path>`. Variable stdout/stderr remains a
console diagnostic and is never copied into the report. Other exceptions are
classified as `check failed (<ExceptionType>)` without their variable message.
The `data_inputs` check structurally loads all 10 ABM archives and all 20
parameter XML files after verifying the 31 manifest entries and hashes.
