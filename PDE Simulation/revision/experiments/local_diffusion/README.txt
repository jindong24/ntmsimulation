Local sensitivity of DB1 and DB2 using recalibrated PDE parameters

The folders are grouped by calibrated dataset and then by experiment:
  all: both DB1 and DB2 sampled independently, 200 workbooks.
  DB1: only DB1 sampled, 100 workbooks.
  DB2: only DB2 sampled, 100 workbooks.
Each varied coefficient was sampled log-uniformly from 0.1 to 10 times
its recalibrated value. The base random seed was 20261110, combined
with dataset ID and branch index (all=0, DB1=1, DB2=2).

The 15 fitted coefficients come from the full-data calibration,
including the selected multi-start fits for 182 and 335. The fixed
background coefficients match the final treatment simulations. No
other model parameter was sampled. In particular, tauB and tauM are
not rescaled as DB1 or DB2 changes. The recalibrated tauM is the
effective exchange coefficient and is not multiplied by DM again.

Simulations use the original N=32 mshr mesh, the dataset's ABM initial
loads and the historical initial profile, the conservative interface,
and the revised velocity u0/mean(eta(m)) in the PDE. The reference
viscosity eta0 and the antibiotic multiplier zeta both equal 1, as
in the original local study. The effective killing rate is
zeta * dB * M0_ABM. The run ends at 14 days with a nominal time step
of 0.1 day. Actual integration steps are listed in day14_summary.csv.

Workbook format: variable lists only the sampled coefficient(s).
B_MUCUS, B_TISSUE, M_MUCUS and M_TISSUE contain the nodal fields at
integer days 0 to 14, in the order of coordinate.xlsx. The flux sheet
contains signed instantaneous quantities at days 0.1, 0.2, ..., 14.0,
using the seven-column convention of the original local scripts.
The interface columns use the surface integral of tau*(tissue-mucus)
and its negative. The upper-right columns integrate the outward
normal component of u0*X - DB1*grad(X) for both species, exactly as
in the original diagnostic; they are not the PDE Robin loss. These
columns are not cumulative.

parameter_set_samples.csv records the complete applied parameter
vector for every workbook, while baseline_parameters.csv records
the nominal inputs and their provenance.
