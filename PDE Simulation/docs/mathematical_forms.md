# Mathematical Form Provenance

## Protected Residuals

Each executable model assembles six protected assignments in this order:

1. `a_B1`: mucus bacteria residual;
2. `a_B2`: tissue bacteria residual;
3. `a_M1`: mucus macrophage residual;
4. `a_M2`: tissue macrophage residual;
5. `a_m`: mucus-density residual;
6. `a`: the sum of the five component residuals.

The Jacobian is `derivative(a, w, TrialFunction(w.function_space()))`. The
checked-in source fixtures contain only these six assignments. Automated AST
tests compare each right-hand-side expression and its order against the
corresponding executable `build_form` block.

| Workflow | Executable form | Authoritative fixture |
| --- | --- | --- |
| calibration | `src/lung_pde/forms/calibration.py` | `docs/provenance/legacy_form_blocks/test_final.py` |
| calibrated baseline | `src/lung_pde/forms/baseline.py` | `docs/provenance/legacy_form_blocks/test_final.py` |
| mucus velocity | `src/lung_pde/forms/velocity.py` | `docs/provenance/legacy_form_blocks/test_v4.py` |
| `danti`/`eta` treatment | `src/lung_pde/forms/treatment.py` | `docs/provenance/legacy_form_blocks/test_anti.py` |

Fixture source paths, original line ranges, target modules, and SHA-256
digests are recorded in `docs/provenance/weak_form_manifest.json`.

## Preservation Rule

No symbolic simplification, integration-by-parts change, sign change,
rescaling, or term reordering was performed when the protected assignments
were moved into modules. The complete executable expressions, rather than a
retyped equation in this document, define the implementation. Changes to
those assignments must update neither fixtures nor checksums merely to make a
test pass; they require an explicit scientific provenance decision.

Calibration and baseline share the `test_final.py` residual. Its interface
coefficients include `DB1*tauB`, `DB2*tauB`, and `DM*tauM`. This deliberately
excludes the unscaled interface terms in historical `aaanew2.py`.

The velocity residual is the `test-v4.py` form, including its weak advection
placement and sign. The velocity field itself is implemented in
`velocity_fields.py` as the original time-dependent horizontal profile, with
zero vertical component.

The treatment residual is the `test_anti.py` form and receives the unscaled
input `danti` constant. Inside the protected bacteria terms it appears with
the ABM initial macrophage value and current calibrated `dB`; the resulting
coefficient is reported as `danti_effective`. The treatment workflow reads
the calibrated parameter branch and never the withdrawn `param_final-m`
branch.

## Geometry and Discretization Context

Both mesh backends represent the same nominal 0.2 by 0.2 domain and interface
at `y = 0.1`, but their triangulations can differ. `rectangle` creates a
structured `RectangleMesh` and requires the interface to align with a mesh
row. `legacy_mshr` calls the optional legacy generator. A successful rectangle
smoke test does not establish equality with a missing `legacy_mshr` run.

All five components use first-order continuous Galerkin elements in one mixed
space. Degrees of freedom outside each component's physical region are fixed
by the constrained nonlinear problem. Solver settings and parameter values
are recorded in output metadata rather than being folded into the protected
form source.
