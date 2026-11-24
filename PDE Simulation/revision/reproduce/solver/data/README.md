# Reference Inputs

The files in this directory are immutable copies of the legacy scientific
inputs. Their source locations, byte sizes, and SHA-256 digests are recorded
in `manifest.json`.

Each ABM archive contains `B`, `M`, and `t`. Element zero of `B` and `M` is
the initial condition; elements one through 83 are the target observations.
The `t` array has 83 points, spaced at 1/6 day.

The initial XML parameter files are copied from the legacy `data/param`
directory. The calibrated XML parameter files are copied from
`data/param_final2`. XML readers retain required parameters and any calibrated
extras, including `loss`.
