# Local identifiability of the calibration

## Results

- All ten manuscript calibration checkpoints were evaluated without refitting.
  The relative residuals reproduce the saved mucus fit errors. At a relative
  singular-value cutoff of 1e-8, the effective ranks of the 15-column
  Jacobians range from 9 to 15; nine datasets have at least one direction
  below that cutoff. No. 335 has effective rank 10 at both 1e-6 and 1e-8.
- For No. 335, the cosine between the DB2 and tauB sensitivity columns is
  0.99999977. After compensating with tauB alone, the linearized change in
  the fitted observations is about 0.068% of the change from DB2 alone.
  This is a sensitivity-column comparison, not a parameter correlation
  estimated from samples.
- The No. 335 objective profile fixed DB2 at 0.08, 0.15, 0.30, 0.60, and
  1.00. The other 14 parameters were reoptimized at each point using the
  original bounds. All five fits met the optimizer's objective tolerance
  (5 to 12 evaluations). The joint MSRE ranges from 0.047536055 to
  0.047536546, a relative spread of 0.00103%. As DB2 increases across the
  profile, the refitted tauB decreases from 0.0354764 to 0.0349958.
- The tissue trajectories also remain close along this particular profile.
  Across pairs of profile fits, the maximum symmetric relative L2
  difference is 0.0039% for bacteria and 0.159% for macrophages. This
  does not establish the robustness of other predictions or other
  parameter directions.

The flat profile shows that the mucus observations weakly constrain DB2
over the tested range in No. 335, despite nearly unchanged predictions
along this particular parameter trade-off. Neither the numerical rank
nor the profile establishes structural non-identifiability.

## Method and files

The analysis uses the original mesh, conservative calibration forms,
1/6-day step, and the 83 observed times for each set. The columns of J
differentiate the 166 relative residuals (mucus bacteria and macrophages)
in the original optimizer coordinates: log10 for positive parameters
and linear for alpha1. Parameters near a box bound are flagged, not
removed. SVD cutoffs are numerical diagnostics, not statistical
thresholds. The sensitivity-column projection follows Wang and Hao
(2025), DOI: 10.1002/advs.202504346. The projected component uses a
truncated SVD with the same absolute cutoff as the 1e-8 rank test.

The information matrix is J^T J. No observation-noise likelihood was
assumed, so we report a Gauss-Newton diagnostic and an objective profile,
not Fisher-based confidence intervals or a profile likelihood.

- local/: Jacobians and information matrices (NPZ), singular-value spectra,
  rank summaries, coordinate projections, right singular vectors, and
  finite-difference checks for DB2 and tauM in No. 335.
- profile/: fixed-DB2 objective values, refitted parameters, and trajectories.
- figures/: two supplemental figures generated from these data.

## Reproduce

After sourcing the revision's `setup.sh`, set STUDY to the absolute path to
the revision directory containing the reference fits. From this directory, run:

    python run_identifiability.py "$STUDY" new_local --sets 335 48 83 182 188 586 646 756 808 843
    python run_profile.py "$STUDY" new_profile --workers 5
    python plot_results.py new_local new_figures --profile new_profile

The saved calibration checkpoints and manuscript results are unchanged.
