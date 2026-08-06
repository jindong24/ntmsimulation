# Lung Bacteria PDE Reproducibility Repository

## Scientific Scope

This repository contains the simulation workflows for a coupled lung infection
PDE model. The five finite-element state variables are mucus bacteria `B1`,
tissue bacteria `B2`, mucus macrophages `M1`, tissue macrophages `M2`, and
mucus density `m`. The repository supports calibration against ABM mucus
trajectories, calibrated baseline simulation, mucus-velocity simulation, and a
two-parameter `danti`/`eta` treatment sweep. Plotting and manuscript figure
generation are intentionally outside this repository.

## Geometry and Supported Sets

The computational domain is a 0.2 by 0.2 rectangle split at `y = 0.1`:
tissue is below the interface and mucus is above it. The supported parameter
sets are `48`, `83`, `182`, `188`, `335`, `586`, `646`, `756`, `808`, and
`843`. Each set has one ABM archive, one initial XML parameter file, and one
published calibrated XML parameter file under `data/`.

## Installation

Create the complete FEniCS 2019.1 environment from the repository root:

```bash
conda env create -f environment.yml
conda activate lung-pde
python -m pip install -e .
```

An existing compatible environment can also be used:

```bash
conda activate fenics
python -m pip install -e .
```

On Apple Silicon, the `rectangle` mesh backend is the supported local path.
The optional `legacy_mshr` backend generally requires a compatible linux-64 or
Intel macOS FEniCS environment. `auto` selects `legacy_mshr` when `mshr` is
importable and otherwise selects `rectangle`. These meshes are not asserted to
be numerically interchangeable.

## Calibrated Parameters

The paper lists 14 symbols because the bacterial diffusivity symbol `D_B` is
represented by two independent regional code variables, giving 15 calibrated
variables:

| Paper symbol | Code variable | Role |
| --- | --- | --- |
| `D_B` | `DB1` | mucus bacterial diffusivity |
| `D_B` | `DB2` | tissue bacterial diffusivity |
| `lambda_B` | `lambdaB` | bacterial growth rate |
| `K_B` | `KB` | bacterial carrying capacity |
| `K_Bi` | `KBi` | interaction half-saturation constant |
| `d_Mb` | `dMb` | bacteria-macrophage interaction rate |
| `d_B` | `dB` | bacteria clearance coefficient |
| `tau_B` | `tauB` | bacteria interface transport coefficient |
| `D_M` | `DM` | macrophage diffusivity |
| `delta` | `delta` | macrophage chemotaxis coefficient |
| `d_M` | `dM` | macrophage death rate |
| `tau_M` | `tauM` | macrophage interface transport coefficient |
| `M_0` | `M0` | reference macrophage density |
| `alpha_0` | `alpha0` | basal recruitment coefficient |
| `alpha_1` | `alpha1` | bacteria-dependent recruitment coefficient |

The mucus parameters `Dm`, `dm`, `m0`, `beta0`, and `beta1` are fixed in the
publication calibration workflow. In particular, `dm` is not recalibrated.
Historical scripts that optimized `dm` are provenance records only.

## Calibration

Validate the reconstructed calibration configuration without importing
DOLFIN:

```bash
python scripts/calibrate.py \
  --set 756 \
  --pipeline configs/calibration/legacy_reconstructed.yaml \
  --validate-only
```

Run it with an explicit mesh backend:

```bash
python scripts/calibrate.py \
  --set 756 \
  --pipeline configs/calibration/legacy_reconstructed.yaml \
  --mesh-backend rectangle \
  --output-dir outputs/calibration
```

Calibration always advances exactly 83 target intervals with fixed
`dt = 1/6` day. It starts from `data/parameters/initial/`; the XML `dt` value
does not override this interval. Every stage may activate a subset of exactly
the 15 variables listed above, with independent `DB1` and `DB2`, while the five
fixed mucus parameters remain unchanged.

`configs/calibration/legacy_reconstructed.yaml` contains three reconstructed
stages based on recoverable legacy parameter subsets. This ordering is not
claimed to be the unique historical calibration sequence, nor is it claimed
to reproduce `data/parameters/calibrated/` exactly. The executable calibration
uses the later `test_final.py` interface fluxes, including `DB1*tauB`,
`DB2*tauB`, and `DM*tauM`, rather than the unscaled interface terms found in
one historical script. See [calibration provenance](docs/calibration_provenance.md).

## Simulations

All post-calibration workflows read the authoritative calibrated XML files.
Validate or run the baseline model:

```bash
python scripts/simulate.py --set 756 --validate-only
python scripts/simulate.py --set 756 --mesh-backend rectangle
```

Validate or run the mucus-velocity model:

```bash
python scripts/simulate_velocity.py --set 756 --validate-only
python scripts/simulate_velocity.py --set 756 --mesh-backend rectangle
```

Both workflows default to `dt = 0.01` day and snapshots at 0, 1, and 2 days.
Baseline defaults to the last ABM target time; velocity defaults to 9 days.

Validate the 170-case treatment grid or run one reduced case:

```bash
python scripts/sweep_danti_eta.py --set 756 --validate-only
python scripts/sweep_danti_eta.py \
  --set 756 \
  --case 1.0 1.0 \
  --mesh-backend rectangle \
  --resolution 4 \
  --final-time 0.1
```

The full grid contains 13 `eta` values by 13 positive `danti_input` values,
plus one zero-treatment reference. The protected treatment form receives
`danti_input`; the coefficient acting in the bacteria terms is recorded as
`danti_effective = danti_input * M0_ABM * dB`. The sweep uses one JSON file per
case, advisory process locks, resume validation, and an atomic aggregate CSV.
Each case also stores a deterministic `runtime_revision` SHA-256 over the
checked-in numerical runtime, form, mesh, model, and solver sources. Resume
requires the current revision to match; an older or mismatched case needs a
new output root or explicit `--overwrite`.

The lock implementation uses advisory `fcntl.flock`. Concurrent workers on
one host are supported. Cross-node Slurm use is safe only when the shared
filesystem provides coherent POSIX advisory-lock semantics to every node;
filesystems that do not honor that contract require separate output roots or
external scheduling serialization.

## Inputs and Outputs

ABM archives have keys `B`, `M`, and `t`; the first `B`/`M` values are initial
conditions and the remaining 83 values are targets. All arrays must be finite
and real, and `t` must be the complete grid from `1/6` through `83/6` day.
Initial parameters came
from the legacy `data/param` branch, while the supplied calibrated parameters
came from `data/param_final2`. Checksums are recorded in
`data/manifest.json`; repository verification checks both those hashes and the
ABM/XML structures.

Baseline and velocity runs write `concentrations.npz`, optional `spatial.npz`,
and `metadata.json`. Calibration writes run metadata/status plus an isolated
directory for every stage. Treatment writes case JSON files and, for a full
grid, `treatment.csv`. The exact keys, shapes, metadata, and time conventions
are defined in [output schema](docs/output_schema.md).

## Verification

Pure-Python tests do not require DOLFIN:

```bash
python -m pytest -m "not fenics and not slow" -v
```

Run the complete repository verifier from a FEniCS-capable environment:

```bash
python scripts/verify_repository.py --skip-slow
```

Run all FEniCS-marked tests separately when numerical verification is wanted:

```bash
python -m pytest -m fenics -v
```

The standard GitHub workflow runs only import-safe pure-Python tests; ordinary
pip CI does not install DOLFIN and therefore does not establish numerical
reproduction. Verification records use `pass` for an executed successful
check, `fail` for an executed unsuccessful check, and `skip` when an optional
dependency is unavailable. A skipped `legacy_mshr` check is not a passed
legacy-mesh reproduction. The deterministic report is written to
`outputs/verification/report.json`.

## Provenance

The protected residual fixtures and their SHA-256 digests are in
`docs/provenance/`. [Mathematical forms](docs/mathematical_forms.md) records
which legacy entry point governs each workflow. `data/manifest.json` records
the original source locations and checksums for copied scientific inputs.
These manifests describe provenance; the repository never writes to the
source locations.

## Citation and License

No citation metadata was supplied, so there is no `CITATION.cff`.
No open-source license was supplied. There is no `LICENSE`, and reuse
permission is not granted by this repository. Contact the project owners
before reuse or redistribution.
