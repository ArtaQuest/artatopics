# artatopics

The research campaign behind **[artaquest.com/topics](https://artaquest.com/topics/)** — a forecasting
model for 251 research fields, where the only inputs at prediction time are the positions of seven
celestial bodies. Everything here is deterministic, leak-tested, and reproducible from this
repository alone: the model of record, the evaluation harness, and every ablation the campaign ran on
the way — including the ones that failed, because a search that only publishes its wins cannot be
trusted.

## What is being predicted

For each of 251 OpenAlex subfields, its **share of the year's total citations** — the fraction of all
scholarly attention arriving that year — from 1700 to 2025, and forward to 2055. The model may read
nothing but planet angles at prediction time: the citation record is fed only during training, and
the sky is known centuries ahead, so the forecast extends to any future year by construction.

## The model of record

One independent receiver per field, **nine parameters each**, and nothing shared:

```
ŷ_j(t) = max( b_j + Σᵢ a_jᵢ · cos(θᵢ(t) − φ_j) , 0 )²
```

a rectified square-law detector: the seven planet waves (Mars, Jupiter, Saturn, Uranus, Neptune,
Pluto, lunar node — sidereal Lahiri) are projected onto the field's **one shared tuning** φ_j, summed
with **signed** arrow lengths a_jᵢ over a level b_j, and the field goes dark when the sum drops below
zero. A negative arrow is exactly a 180° flip of that body against the tuning, so the model is one
continuous phase plus seven binary ones — and both halves of that sentence were measured, not
assumed (see the findings below).

**There is no training run.** With φ fixed the model is linear on the amplitude scale, so a 1° sweep
of φ with a closed-form weighted least-squares solve at each step fits it exactly — no optimiser, no
seed, no early stopping. Refit it and you get the same bits. The **horizon anchor** (the single
biggest design decision: removing it drops the headline from +0.80 to +0.63) lives inside the same
solve as extra rows: the model's own thirty-year forecast is pulled toward the level the field held
in its last five training years, scale-free, using nothing but planet angles and training data.

### Headline results — honest walls

The model is refitted knowing nothing after a cut-off, then scored on what actually happened.
Skill is measured against carrying each field's own train-window mean forward; AUC pools it over
every horizon in the window.

| test | AUC |
|---|---|
| fit ≤ 2021 → forecast 4 years | +0.9737 |
| fit ≤ 2013 → forecast 12 years | +0.9318 |
| **fit ≤ 1995 → forecast 30 years (the headline)** | **+0.7990** |
| carry-forward persistence at the same 30-year wall | +0.7344 |
| twelve origins 1963..1996, each → its next 30 years (mean) | +0.8751 |
| persistence over the same twelve origins | +0.8511 |

Median field skill at the headline wall +0.5256, with 76.1% of fields beating their own mean thirty
years out. The carry-forward bar is reported everywhere because a model that cannot beat doing
nothing is not a model — and over the twelve origins the model is ahead at only 7 of 12, which the
live page states as plainly as this file does.

## Reproduce it

```bash
git clone https://github.com/ArtaQuest/artatopics
cd artatopics
pip install numpy pandas            # scipy + torch only for the comparison scripts
python3 analysis/arxivtopics/arxiv_fit.py
```

Two to three minutes on a laptop. The script prints the deploy fit, all three walls, the carry-forward
bar and the season occupancy, then writes `arxiv_phasor_final.npz` — and because the fit is
closed-form and seedless, your numbers should match the table above exactly, not approximately.

## Map of the repository

Everything runs from the repository root, and the tree shape is preserved from the source monorepo so
every script runs verbatim.

| path | what it is |
|---|---|
| `analysis/arxivtopics/arxiv_fit.py` | **the model of record** — fit, benchmark walls, npz export. `fit_market()` inside it is the superseded 15-parameter gradient model, kept for comparison |
| `analysis/arxivtopics/comp_harness.py` | the competition harness: one source of data, target, walls and scoring for every candidate model |
| `analysis/arxivtopics/final_decide.py` | the decision that made the deterministic model the record: both models at twelve origins |
| `analysis/arxivtopics/global_ceiling.py` → `lowrank_arrows.py` → `global_final.py` | the global-model search: the shared-structure diagnostic, the rank-3 shared basis, and the basis + field-prior combination |
| `analysis/arxivtopics/final_pooled.py` | field-level partial pooling (+0.8800) and the ridge control that proves pooling ≠ regularisation |
| `analysis/arxivtopics/staged_phases.py` · `eight_*.py` · `nonneg_nine` (in `eight_nonneg.py`) | the parameter-count and sign ablations |
| `analysis/arxivtopics/kl_balance.py` | the twelve-season balance term, measured and declined |
| `analysis/arxivtopics/hypernet.py` · `dict_spectrum.py` | the amortised map and the spectrum dictionary — one clean negative, one compression result |
| `analysis/arxivtopics/build_ephemeris.py` | regenerates the ephemeris CSVs (Skyfield, sidereal Lahiri) |
| `analysis/citations/` | the yearly citation matrices (OpenAlex, CC0) the campaign trains on |
| `analysis/arxivtopics/*.json` | the committed result of every experiment above, exactly as its script wrote it |
| `analysis/adstopics/astro_phasor2.py` | a two-constant import shim replacing a monorepo module — see its docstring |

Two scripts are deliberately site-coupled and will not run here: `arxiv_export.py` writes the fitted
atlas into the ArtaQuest web checkout, and `build_kaggle_nb.py` targets the platform's Kaggle
publishing pipeline. They are included because they are part of the campaign's record.

## The findings, briefly

- **The sign is the second phase.** Forcing every arrow positive costs −0.1858 AUC over twelve
  origins, losing at all twelve; giving every body its own free continuous phase overfits by −0.0145.
  One continuous tuning plus seven binary flips is the measured optimum.
- **The anchor is the model.** The single largest lever found in the whole campaign: the headline is
  +0.7990 with the horizon anchor and +0.6287 without it. Architecture mattered far less than
  pinning the extrapolation.
- **Persistence is the honest bar.** Carry-forward scores +0.8511 over twelve origins — better than
  several models this campaign produced along the way. Every result here is reported against it.
- **The atlas has a three-dimensional shape.** The 251 gauge-fixed arrow vectors carry 97.9% of their
  variance in three components (effective rank 2.27). A rank-3 shared basis matches the free model at
  56% of the parameters and beats it at the headline wall (+0.8290 vs +0.7990) — and rank 3 was
  predicted from the singular spectrum before any forecast was scored.
- **Pooling is not regularisation.** Shrinking a field's arrows toward its OpenAlex field's shared
  spectrum gains (+0.8800); shrinking toward zero at the same strength loses (+0.8633 vs +0.8751).
  The gain is monotone in granularity: global < domain < field.
- **What did not work, on the record:** the twelve-season KL balance term (β=0 wins at every
  temperature), deleting or mean-filling any body (a tie at best — and the one apparent +0.02 win
  was selection noise), power-law and globally-shared amplitude spectra, L1/IRLS refits, a fixed
  intercept, and the amortised feature→parameter map, which fails on unseen topics (+0.6874 vs
  persistence +0.7369). Cold-start on an unseen field is unsolved.

## Evaluation discipline

Every constant in the model was chosen by **rehearsal at historical origins** (fit ≤1935 judge
1936–65; fit ≤1965 judge 1966–95), never against the years being forecast. Every architecture search
in this repository selects on early origins or the inner wall and reports later origins untouched —
and where the campaign broke that rule it says so in the file (the one-tuning collapse was selected
on the thirty-year wall itself, so for that decision the headline is not a clean holdout; the
twelve-origin comparison, ten of whose origins had no part in any choice, is the evidence that
carries it). The twelve origins overlap heavily, so no standard error is quoted on their means.
Leakage is tested empirically, not assumed: randomising the target after the wall moves the fitted
parameters by ~1e-15.

## What this is and is not

This repository measures the predictive skill of a fixed, deterministic feature basis — planetary
longitudes — against stated baselines, under walls that forbid the model from seeing the future it is
scored on. The numbers are what they are: the model beats carrying the past forward, on this record,
by the margins in the table. **No causal claim is made or implied.** The sky here is a basis of slow,
perfectly forecastable oscillators; whether its skill reflects anything more than that is exactly the
kind of question the honest walls exist to keep open rather than answer by assertion.

## Data and provenance

The citation matrices derive from the public OpenAlex snapshot (CC0), materialised as the platform's
citations rail — the same dataset published as
[Citations per year across the scholarly record, 1000–2026](https://artaquest.com/nb/9303/citations-per-year-across-the-scholarly-record-1000-2026-9303)
(DOI 10.5281/zenodo.21537062). The ephemeris CSVs are generated by `build_ephemeris.py` (Skyfield,
DE441, sidereal Lahiri) and committed so nothing here depends on a network call.

Imported from the ArtaQuest monorepo; development history and the operating ledger of the campaign
remain there.

## Licence

Code MIT. Data files (`analysis/citations/*.csv`, the ephemeris CSVs and the committed result
artifacts) CC0, following OpenAlex.
