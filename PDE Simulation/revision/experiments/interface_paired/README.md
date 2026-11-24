# Historical paired interface replays

This intermediate comparison used a structured rectangular mesh and the
earlier `kappaB`/`kappaM` solver. It is not the final original-mshr model.
The bundled snapshot in `solver/` uses the same compiled velocity expression
for both interface laws, as in the historical paired study.

From the revision directory, in the FEniCS environment:

```sh
(
    paired=experiments/interface_paired
    export PYTHONPATH="$paired/solver/src:$paired/solver/scripts"
    out=runs/historical_paired
    python "$paired/replay_paired_interface.py" inputs/sensitivity/legacy_design.csv "$out"
    python "$paired/replay_paired_experiments.py" velocity inputs/legacy_xml inputs/treatment_grid "$out" --resolution 32
    python "$paired/replay_paired_experiments.py" treatment inputs/legacy_xml inputs/treatment_grid "$out" --resolution 16 --axis-points 5
)
```

The comparison scripts accept explicit paths to regenerated and archived
endpoint CSVs. Use their `--help` options for input contracts. Some historical
comparisons require archived endpoints not bundled here.
