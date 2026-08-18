# Kaggle state — science-distribution-astro (org 5418) · 2026-08-18

## Done through the API + lab browser (verified)
- Metric picker: **ROC AUC set** (both competitions) — DataGrid radio + Select; validated by re-reading the card.
- Key dates: end 2027-02-28 23:59, private release 2027-03-07 (saved).
- Data: **v3 uploaded** — `train.csv` 134 rows × (year + 251 columns summing to 1), `test.csv` = 34 years, `ephemeris.csv`, `fields.csv`.
- Solution: **v3 wide solution attached, Ready: true, Row ID = year** (needs a `Usage` column — added, temporal).
- Public dataset `artafather/science-distribution-251` on v3.
- Two Kaggle-scored submissions under the v2/ROC pairing: **random 0.50897 · astrology entry 0.48541**.
- Custom KL metric notebook: `artafather/distribution-kl-divergence-metric` — **"Competition Metric Notebook · Validation Succeeded (eligible for use as a metric)"**, public, doctests pass, reproduces uniform 0.7361 / climatology 0.3430 exactly.

## Blocked
The competition's metric is still ROC. Selecting the KL metric in the picker is refused **client-side and silently**: the row's radio checks, the Select button is enabled, a real click lands, and **zero requests fire** — no toast, no console error, dialog stays open. The row differs from Kaggle's own ROC row in exactly one way the DOM shows: it has **no description, no category, no detail-panel toggle**, and no version of pushing (docstring styles, tags, description cell, fresh fork, slug-preserving overwrite) fills them. The picker's per-metric metadata is set at Kaggle's registration, not by notebook pushes. `Create a new metric.` in the picker is an inert `<a>` from the parked lab window (no `window.open`, no navigation, no notebook created).

## Two ways to finish, both need a human decision
1. Kaggle-native: open the metric picker in a **foregrounded** browser and click "Create a new metric." — it likely opens a popup the parked lab blocks; paste `kaggle_metric_notebook.ipynb`'s code into the created notebook and it registers with description/category. Then Select. Solution + data are already the right shape.
2. Keep ROC on Kaggle with the v2 (row-per-(field,year), `share > year median`) framing — that scores today — and publish KL/cross-entropy as the honest headline on the artatopics site, where the baselines already live.

Remaining launch-checklist items after the metric: Edit Rules · Edit Overview · Dataset Description · Dataset License · Sandbox submission (the two accepted ones may already satisfy it) · Launch.
