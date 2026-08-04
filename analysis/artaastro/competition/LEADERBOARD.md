# ArtaAstro competition — leaderboard

_Metric: plain **R²** per measure over the de-timed/shuffled holdout, averaged over the 6 measures (platform `r2`). Fit on train (2015-2019) only. **0 = no better than each measure's mean.**_

| rank | entrant | R² (mean over measures) |
|---|---|---|
| 1 | ARTAASTRO-intensity | -0.0604 |
| 2 | predict-mean | -0.0694 |
| 3 | global-phase-baseline | -0.2647 |

## Per-measure R²

| measure | ARTAASTRO-intensity | predict-mean | global-phase-baseline |
|---|---|---|---|
| material | -0.0881 | -0.0952 | +0.0153 |
| conflict | -0.0807 | -0.0971 | -0.1725 |
| verbal_conf | -0.0146 | -0.0223 | -0.7144 |
| cooperation | -0.0807 | -0.0971 | -0.1725 |
| material_coop | -0.0078 | -0.0124 | -0.2984 |
| violence | -0.0906 | -0.0925 | -0.2455 |

## Global-phase baseline — fitted phase (sign) & fit per measure

| measure | phase | sign | validated | val R² | in-sample R² | holdout R² |
|---|---|---|---|---|---|---|
| material | 166.2° | Virgo | False | -0.004 | +0.176 | +0.015 |
| conflict | 224.1° | Scorpio | True | +0.072 | +0.115 | -0.172 |
| verbal_conf | 341.5° | Pisces | True | +0.055 | +0.161 | -0.714 |
| cooperation | 224.1° | Scorpio | True | +0.072 | +0.115 | -0.172 |
| material_coop | 315.1° | Aquarius | False | +0.012 | +0.071 | -0.298 |
| violence | 257.4° | Sagittarius | False | -0.011 | +0.270 | -0.245 |

## Reading it

- **Every entrant has R² ≤ 0** — none beats predicting each measure's own holdout mean.
- The single baseline is the **global-phase** model `ŷ = Σ wᵢ·sinc(fᵢ(xᵢ−p)) + b`, fit by gradient descent from **12 sign-centre phase restarts**, time-validated, with bounded frequencies + weight decay so it can't overfit; a measure whose best sign does not beat the mean on validation falls back to the mean. Its fitted phases/frequencies are reported above and on the /astro page — but the holdout R² shows those signs do not forecast the 2020→ future.
- Consistent with the full-history backtest (`../RESULTS.md`, ρ≈0): no sky feature, and no sign, forecasts world events. Reported straight.
