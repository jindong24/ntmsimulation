# Macrophage interface PRCC comparison

Both branches use the same 400 range-rescaled inputs, original mesh,
initial loads, mean-viscosity velocity, 0.1-day time step, and conservative
bacterial exchange. In the hybrid branch we restore only the historical
side-specific macrophage chemotaxis terms at the interface. The nominal
shared macrophage permeability (DM times the recorded tauM) and local
viscosity mobility multiplier are identical between branches.

- `prcc/same_diagnostic.csv`: Uses the same diagnostic definitions for
  both branches. The macrophage transfer diagnostic integrates the
  positive and negative parts of the shared concentration-jump term,
  whether or not the added side-specific terms are present.
- `prcc/physical_tissue_flux.csv`: Also includes the legacy tissue-side
  chemotactic flux in the old branch's three macrophage transfer
  measures. In the old branch this flux need not equal the mucus-side
  loss. Clearance uses the modeled Robin outflow in both tables; it is
  not the alternative diagnostic in the exported sample workbooks.
- `prcc/hybrid_endpoints.csv`: The 400 matched hybrid replay outcomes
  before ranking, including both definitions of macrophage exchange.

Both PRCC tables use the same 16 archived predictor columns and
Bonferroni adjustment over 16 predictors per outcome. Significance
flags describe threshold crossings, not a statistical test of the
difference between coefficients. Archived conservative results were
checked against three conservative replays and all 224 coefficients
were recomputed.
