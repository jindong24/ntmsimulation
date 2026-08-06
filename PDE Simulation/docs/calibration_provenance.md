# Calibration Provenance

## Publication Contract

The publication workflow calibrates only these 15 code variables:
`DB1`, `DB2`, `lambdaB`, `KB`, `KBi`, `dMb`, `dB`, `tauB`, `DM`, `delta`,
`dM`, `tauM`, `M0`, `alpha0`, and `alpha1`. `DB1` and `DB2` are independent.
`Dm`, `dm`, `m0`, `beta0`, and `beta1` remain fixed. Calibration always uses
83 ABM targets at `dt = 1/6` day and starts from the selected initial XML.

The cleaned engine reads ordered stages from YAML and applies no implicit
bacteria/macrophage grouping. `legacy_reconstructed.yaml` is a transparent
three-stage reconstruction; it is neither a claim of the unique historical
sequence nor a claim that rerunning it regenerates the supplied calibrated XML
files exactly.

## Recoverable Legacy Scripts

Bounds below are physical bounds. `log` means that the legacy optimizer used
the logarithm of the physical value; `linear` means it optimized the physical
value directly. The objective expressions name simulated mucus trajectories
`B`, `M` and ABM targets `Bt`, `Mt`.

| Script | Input parameter directory | Active variables, transforms, and bounds | Returned objective | Direct output |
| --- | --- | --- | --- | --- |
| `test_all.py` | `data/param` | `tauM` linear `[0,10]`; `tauB` linear `[0,10]`; `lambdaB` log `[lambdaB1,lambdaB2]`; `dm` linear `[dm1,dm2]` | `sum((B-Bt)^2/Bt^2) + sum((M-Mt)^2/Mt^2)` | `predict/data_<set>_<myt>.npz` (`MM`, `BB`); `plot/result_<set>.png`; no XML checkpoint |
| `test_dmb.py` | `data/param` | `tauM` linear `[0,100]`; `tauB` linear `[0,100]`; `dMb` linear `[1e-8,1]`; `lambdaB` log `[lambdaB1,lambdaB2]`; `M0` linear `[0.01,0.75]` | `mean((B-Bt)^2/Bt^2) + mean((M-Mt)^2/Mt^2)` | `predict/data_<set>_<myt>.npz` (`MM`, `BB`); `plot/result_<set>_<myt>.png`; no XML checkpoint |
| `aaa.py` | `data/param2` | `tauM` linear `[0,100]`; `tauB` linear `[0,100]`; `dMb` log `[1e-8,10]`; `lambdaB` log `[lambdaB1,lambdaB2]`; `M0` log `[1e-4,0.75]`; `alpha1` linear `[-100,100]` | relative mean squared error for both trajectories | `predict/aaa_<set>_<myt>.npz` (`MM`, `BB`); `plot/aaa_<set>_<myt>.png`; no XML checkpoint |
| `aaanew2.py` | `data/param2` | `tauM`, `tauB` linear `[0,100]`; `dMb` log `[1e-8,10]`; `lambdaB` log `[lambdaB1,lambdaB2]`; `M0` log `[1e-4,0.75]`; `alpha1` linear `[-100,100]`; `DM`, `DB1`, `DB2`, `dM`, `dB`, `delta` log `[1e-9,1]` | `norm(B-Bt)/norm(Bt) + norm(M-Mt)/norm(Mt)` | `data/calib/<set>_aaanew2.txt`; `predict/aaanew2_<set>_<myt>.npz` (`MM`, `BB`); `plot/aaanew2_<set>_<myt>.png`; no XML checkpoint |
| `aaasimple.py` | `data/param2` | `tauM`, `tauB` linear `[0,100]`; `dMb` log `[1e-12,10]`; `lambdaB` log `[lambdaB1,lambdaB2]`; `M0` log `[1e-5,1]`; `alpha0`, `alpha1` linear `[0,100]`; `DM`, `DB1` log `[1e-9,10]` | relative mean squared error for both trajectories | `data/calib/<set>_aaasimple.txt`; `predict/aaasimple_<set>_<myt>.npz` (`MM`, `BB`); `plot/aaasimple_<set>_<myt>.png`; no XML checkpoint |
| `aaasimpleB.py` | `data/param_train2` | `tauB` linear `[0,100]`; `lambdaB`, `DB1`, `DB2` log `[1e-12,1e3]` | `mean((B-Bt)^2/Bt^2)` | `data/calib/<set>_aaasimple-B.txt`; `predict/aaasimple-B_<set>_<myt>.npz` (`MM`, `BB`); `plot/aaasimple-B_<set>_<myt>.png`; no XML checkpoint |
| `test_lamB2.py` | `data/param_final` | `lambdaB` log `[0.1,10]`; `DB1`, `dMb`, `dM` log `[1e-10,1]`; `DM` log `[1e-6,1]`; `KB` log `[1e-6,1]`; `KBi` log `[1e-8,1]` | `norm(B-Bt)/norm(B) + norm(M-Mt)/norm(M)` | `data/calib/<set>_lamB.txt`; `predict/cal-lamB_<set>_<myt>.npz` (`MM`, `BB`); `data/cal-lamB/val-<set>.png`; no XML checkpoint |

`test_all.py` is historical only because it optimized `dm`. The publication
calibration excludes `dm`, regardless of its presence in that script.

## Parameter Directory History

The legacy tree contains manually propagated checkpoints in `param`, `param2`,
`paramsimple`, `param_train*`, `param_final*`, and related directories. The
scripts above prove their own input directory but generally do not write an
XML output, so the complete hand-off sequence cannot be inferred uniquely.
The cleaned mapping is therefore deliberately narrower:

| Legacy source | Repository role |
| --- | --- |
| `data/param` | immutable initial XML in `data/parameters/initial/` |
| `data/param2`, `data/paramsimple`, `data/param_train*`, `data/param_final*` | historical intermediate branches; documented but not treated as authoritative inputs |
| `data/param_final2` | supplied published calibration in `data/parameters/calibrated/` |

Calibration outputs are written beneath `outputs/calibration/set_<id>/<run>/`
and never overwrite either tracked parameter directory.

## Interface-Flux Decision

`aaanew2.py` used unscaled interface coefficients: `tauB` without `DB1` or
`DB2`, and `tauM` without `DM`. The publication calibration instead uses the
later `test_final.py` form: mucus and tissue bacteria terms use `DB1*tauB` and
`DB2*tauB`, and both macrophage interface terms use `DM*tauM`. This is an
explicit provenance choice, not an algebraic simplification of `aaanew2.py`.

The single-stage YAML files in `configs/calibration/legacy/` preserve the
recoverable `aaanew2.py`, `aaasimple.py`, and `test_lamB2.py` subsets while
still using the publication `test_final.py` residual. Their initial vectors
come from XML; hard-coded legacy `x0` values are not silently substituted.
