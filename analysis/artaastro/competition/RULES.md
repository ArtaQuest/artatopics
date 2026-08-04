# ArtaAstro competition — rules & metric

**Does the sidereal (Lahiri) sky forecast the world's day-to-day conflict?** ArtaAstro is entered as one
competitor against a single, principled baseline.

## The data

- **Ground truth:** the **entire GDELT 1.0 event history** (1979 → present, ~500M machine-coded world
  events), aggregated to **global daily** measures and cleaned (see [DATASET.md](DATASET.md)). Every
  measure is a within-day **share**, so GDELT's ~1000× volume growth cancels.
- **6 target measures ("topics"):** `material`, `conflict`, `verbal_conf`, `cooperation`,
  `material_coop`, `violence` (cooperation measures are controls).
- **Features:** the **12 sidereal ecliptic longitudes** for each day — **Sun, Moon, Mercury, Venus,
  Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, Rahu, Ketu** (degrees; Lahiri ayanamsa).
- **Anti-cheat = de-time + shuffle.** No date column; rows shuffled; the test `id` is anonymised and
  independent of the date. The target derives from *public* GDELT, so re-deriving the date from the sky
  to look it up is disallowed (rule 3) — predict from the provided sky only.

## Windows

- **Train:** clean days **2015-2019** (the modern GDELT regime), with target.
- **Test (holdout):** **2020-01-01 → present** (the future), target hidden.

## The metric — plain **R²**

Leaderboard = **mean over the 6 measures of the plain R²** of your predictions vs the hidden holdout
(the platform's `r2` metric, `Competitions::per_target_r2s`). **0 = no better than predicting each
measure's mean** (an unpredicted row scores as the mean).

## The baseline — one global-phase model, fit by gradient descent

    ŷ(x) = Σᵢ wᵢ · sinc( fᵢ · (xᵢ − p) ) + b

One **global phase `p`** (shared by all 12 bodies), 12 per-body **frequencies `fᵢ`** (bounded, smooth),
12 per-body **weights `wᵢ`**, and a **bias `b`** — 26 parameters, fit by **gradient descent** (Adam).
`sinc(z)=sin(πz)/(πz)`; the argument uses the wrapped angular distance `(xᵢ−p)` folded to ±180° and
scaled by `/180`, so `fᵢ` is the number of sinc lobes across a half-turn.

**The learning algorithm (fair + overfit-resistant):**
1. **12 restarts of the phase `p`, one at each SIGN CENTRE** (15°, 45°, …, 345°) — an even, fair start.
2. Each restart is **early-stopped on a time-blocked validation** (the last 20 % of train), and the
   best restart is chosen by that validation R². That best phase is the measure's reported **sign**.
3. Frequencies are **bounded** (≤ 1 lobe / half-turn) and weights carry **decay**, so the model can't
   chase high-frequency noise.
4. **Validated-only:** if the best sign does not beat the mean on validation, the model **falls back to
   predicting the mean** — the sky is used only where it demonstrably helps.

Each measure's fitted **phase (→ sign)** and per-body **frequencies** are reported on the
**/astro** page and in [LEADERBOARD.md](LEADERBOARD.md). Beat the baseline: submit `trend,id,target`.

## Rules

1. Fit only on the public train data (2015-2019). The holdout targets are hidden.
2. Submit a CSV `trend,id,target` — one prediction per test row (see `sample_submission.csv`).
3. **No external leak of the hidden series** — including re-deriving the date from the sky to look the
   target up in GDELT. Predict from the provided features only.
4. Deterministic, reproducible entries only.
