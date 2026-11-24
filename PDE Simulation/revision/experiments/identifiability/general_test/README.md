# Uniform nonlinear identifiability checks

This analysis applies the same protocol to all ten retained full-data
fits, rather than choosing a particular parameter or dataset. Using
the archived residual Jacobians, it replays the nonlinear PDE along
strong, middle, and weak right singular vectors in the common feasible
interior subspace for each fit. At both displacement sizes (0.10 and
0.25 in optimizer coordinates), both signs are evaluated. The 120
perturbations are not refits or objective profiles.

At the larger displacement the median observation-scaled mucus RMS
change for the middle directions is 0.59%, compared with a median
symmetric relative tissue macrophage change of 10.47%. In two datasets
both signs change tissue macrophages by over 10% while changing the
mucus trajectories by under 1%. The weak directions change the mucus
trajectories by at most 0.00080% in this test. These are changes in
trajectories, not relative changes in the objective or uncertainty
bounds. The original single-parameter profile remains an illustration.

- results/outcomes.csv: one row per signed perturbation.
- results/by_dataset_max_over_signs.csv: larger change over the two signs
  for each dataset, amplitude, direction, and output.
- results/summary_across_datasets.csv: minima, medians, and maxima over
  the ten datasets.
- results/set_NNN/: individual perturbation vectors and replay trajectories.
- writeup/: revised manuscript paragraph, supplement, reviewer response,
  and figures. The earlier write-up is unchanged.

## Reproduce

Activate the original-mesh FEniCS environment. With STUDY set to the
path to ntm-original-mesh-tau and ROOT set to the path to the parent
ntm-identifiability-results directory:

    PYTHONPATH="$ROOT" python run_directional_checks.py "$STUDY" "$ROOT/local" new_results
    python verify_directional_checks.py "$ROOT/local" new_results
    PYTHONPATH="$ROOT" python summarize_directions.py new_results directional_perturbations.png
