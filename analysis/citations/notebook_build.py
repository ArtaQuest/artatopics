#!/usr/bin/env python3
"""Builds the dataset-kind notebook nb.ipynb for the citations-per-quarter publication."""
import json, sys

MD = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
CODE = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = []

cells.append(MD("""# Citations per quarter across the scholarly record, 1500–2026

Most citation datasets tell you how many citations a paper *has*. This one tells you **when
citation activity happened**: for every research subfield and every quarter since 1500, how
many works were published, how many citations those works eventually earned, and — the rare
part — how many citations the subfield **received in that quarter**, with each of 3.0 billion
reference edges dated by the *citing* work's publication quarter.

The table is read from `data/citations-quarterly.csv`, the platform's public citations rail
(`GET /aq/v1/citations/quarterly`), derived from the full OpenAlex snapshot of 2026-06-26
(CC0). This notebook pins that snapshot, verifies it, explores what is inside, and writes the
curated dataset to `out/data.csv` with a datasheet below."""))

cells.append(CODE("""import hashlib
import matplotlib.pyplot as plt
import pandas as pd

# One mid-grey ink for every annotation so both dark and light themes read it (ArtaContrast).
GREY, GOLD, BLUE = "#808080", "#E8B923", "#1746DC"
plt.rcParams.update({"text.color": GREY, "axes.edgecolor": GREY, "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "grid.color": GREY,
                     "figure.dpi": 110, "figure.facecolor": (0, 0, 0, 0),
                     "axes.facecolor": (0, 0, 0, 0), "savefig.transparent": True,
                     "axes.grid": True, "grid.alpha": 0.4, "font.size": 11})

rail = pd.read_csv("data/citations-quarterly.csv", dtype={"subfield_id": str, "field_id": str})

# PIN the snapshot window: 1500 through 2026 Q2. Mis-dated rows OUTSIDE it drop on BOTH sides —
# 348 sparse medieval rows below, and every future-dated row above (2027-2030 metadata errors).
pinned = rail[(rail.year >= 1500)
              & ((rail.year < 2026) | ((rail.year == 2026) & (rail.quarter <= 2)))].copy()
assert list(rail.columns) == ["subfield_id", "subfield", "field_id", "field", "domain",
                              "year", "quarter", "works", "cited_by_sum", "citations_received"]
assert len(pinned) == 250_727 and pinned.subfield_id.nunique() == 253
assert int(pinned.year.max()) == 2026 and int(pinned.year.min()) == 1500
assert pinned.works.sum() == 315_096_476 and pinned.citations_received.sum() == 3_005_254_274
spot = pinned.set_index(["subfield_id", "year", "quarter"])
assert tuple(spot.loc[("1702", 2020, 1)][["works", "cited_by_sum", "citations_received"]]) == (26_213, 417_542, 499_687)
assert tuple(spot.loc[("3103", 1985, 3)][["works", "cited_by_sum", "citations_received"]]) == (2_719, 45_789, 14_215)
print(f"pinned {len(pinned):,} rows · {pinned.subfield_id.nunique()} subfields · "
      f"{pinned.works.sum():,} works · {pinned.citations_received.sum():,} citations received")"""))

cells.append(MD("""## Two measures, two questions

Each row carries both citation measures, and they answer different questions:

- **`cited_by_sum`** — the *cohort* measure: citations ever earned by works **published** in
  that quarter. Ask it "did papers published in spring age better than autumn ones?" Beware
  right-censoring: recent quarters look weak only because their papers are young.
- **`citations_received`** — the *event* measure: citations **arriving** in that quarter,
  dated by the citing work. Ask it "when does citation activity happen?" Its right edge is
  the snapshot date (2026 Q2 is partial), not a slow decay."""))

cells.append(CODE("""century = pinned.assign(era=(pinned.year // 100) * 100).groupby("era")[
    ["works", "cited_by_sum", "citations_received"]].sum()
century.index = [f"{e}s" for e in century.index]
print("coverage by century (pinned window):")
print(century.to_string(formatters={c: "{:,.0f}".format for c in century.columns}))"""))

cells.append(MD("""## Quarter 0 is not a season — it is honesty about precision

OpenAlex stores a work known only to a year as January 1st. That is a *third of all works
overall* — from 31% in the 2000s to near-total (99%) in the 1500s, as the figure below
measures — so this dataset refuses to pretend: any Jan-1-dated work lands in
**`quarter = 0`** ("year known, quarter unknown"), and quarters 1–4 hold only works with real
dates. The price is that works genuinely published on New Year's Day are swept into q0 too.
Never merge q0 into Q1 — a naive merge would fabricate a January publication spike that is
pure metadata artefact. If a gapless quarterly series is required, distribute q0 pro-rata
over that year's q1–q4 — and say so."""))

cells.append(CODE("""cent = pinned.assign(era=(pinned.year // 100) * 100).groupby(["era", "quarter"]).works.sum().unstack(fill_value=0)
q0_share = (100 * cent[0] / cent.sum(axis=1)).round(1)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(q0_share.index, q0_share.values, color=BLUE, marker="o", linewidth=2)
ax.set_xlabel("century"); ax.set_ylabel("works dated year-only (%)")
ax.set_title("The q0 share: date precision across five centuries")
plt.show()
print("year-only share of works by century (%):", q0_share.to_dict())
print(f"overall year-only share: {100 * pinned[pinned.quarter == 0].works.sum() / pinned.works.sum():.1f}%")"""))

cells.append(MD("""## When do citations arrive within the year?

Among *date-resolved* citation events of 1900–2025, the arrival share climbs steadily through
the calendar: quarters 1 to 4 carry unequal weight, with Q4 the heaviest. Read this with both
eyes open: it mixes real publishing calendars (journals clear backlogs late in the year) with
the residue of the q0 artefact (year-only *citing* works are excluded here, and they are not
a random sample of publishers). The dataset lets you take either view — or model both."""))

cells.append(CODE("""dated = pinned[(pinned.quarter > 0) & pinned.year.between(1900, 2025)]
share = 100 * dated.groupby("quarter").citations_received.sum() / dated.citations_received.sum()
fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.bar(share.index, share.values, color=[BLUE, GOLD, BLUE, BLUE], width=0.62)
ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("calendar quarter")
ax.set_ylabel("share of citations received (%)")
ax.set_title("Arrival share of date-resolved citations, 1900–2025")
plt.show()
print("received share by quarter (%):", share.round(2).to_dict())"""))

cells.append(MD("""## Fields at a glance

The event measure makes rises and falls legible at any zoom. Three large subfields over
seven decades: all three climb on a log scale, but not identically — Electrical &
Electronic Engineering holds the top line in every year but one (1974) and peaks right at
the snapshot edge, while Artificial Intelligence starts in 1950 at roughly half Astronomy's
level and passes it for good in 2002. The crossings, not the slopes, are the story."""))

cells.append(CODE("""recent = pinned[(pinned.year == 2025)].groupby(["subfield_id", "subfield"]).citations_received.sum()
print("top subfields by citations received in 2025:")
print(recent.sort_values(ascending=False).head(8).to_string())

trio = {"1702": ("Artificial Intelligence", GOLD), "3103": ("Astronomy & Astrophysics", BLUE),
        "2208": ("Electrical & Electronic Eng.", GREY)}
fig, ax = plt.subplots(figsize=(7, 3.6))
for sid, (label, colour) in trio.items():
    yearly = pinned[pinned.subfield_id == sid].groupby("year").citations_received.sum()
    yearly = yearly[yearly.index.to_series().between(1950, 2025)]
    ax.plot(yearly.index, yearly.values, color=colour, linewidth=2, label=label)
ax.set_yscale("log"); ax.legend(frameon=False, labelcolor=GREY)
ax.set_xlabel("year of the citing work"); ax.set_ylabel("citations received / year")
ax.set_title("Three subfields, three trajectories (event measure)")
plt.show()
series = {sid: pinned[pinned.subfield_id == sid].groupby("year").citations_received.sum()
          for sid in trio}
not_top = [y for y in range(1950, 2026)
           if series["2208"].get(y, 0) < max(series["3103"].get(y, 0), series["1702"].get(y, 0))]
crossed = next(y for y in range(1950, 2026)
               if all(series["1702"].get(z, 0) > series["3103"].get(z, 0) for z in range(y, 2026)))
print(f"EE not the top line only in {not_top}; AI/Astronomy in 1950: "
      f"{series['1702'].get(1950, 0) / series['3103'].get(1950, 0):.2f}x; AI passes Astronomy for good in {crossed}")"""))

cells.append(MD("""## Datasheet

### Schema
`out/data.csv`, one row per (subfield, year, quarter), sorted by all three keys:
`subfield_id` (OpenAlex subfield, string; `none` = works with no primary topic) · `year`
(1500–2026) · `quarter` (0 = year-only precision, 1–4 = calendar quarters) · `works`
(count published) · `cited_by_sum` (lifetime citations of that cohort, integer) ·
`citations_received` (event-dated arrivals that quarter, integer) · `cites_per_work`
(cohort ratio, 3 decimals; 0 where the cohort is empty).

### Provenance
Derived from the complete OpenAlex works snapshot of **2026-06-26** (510.4M records; 317.8M
canonical works kept; 3.005B reference edges joined, 95.2% resolved to a cited topic), via
the platform's public citations rail read above and pinned by row count, totals and spot
cells. Regeneration code: `analysis/citations/harvest_openalex.py` + `build_quarterly.py`
in the platform repository.

### Licence
OpenAlex metadata is **CC0**; this derived table is likewise released **CC0** — cite the
DOI so readers can find the pinned snapshot.

### Units
Counts are integers of works and of citation events; `cites_per_work` is a dimensionless
ratio. Years are calendar years (CE); quarters are calendar quarters of the Gregorian year."""))

cells.append(CODE("""out = pinned[["subfield_id", "year", "quarter", "works", "cited_by_sum",
              "citations_received"]].copy()
out["cites_per_work"] = (out.cited_by_sum / out.works.clip(lower=1)).round(3)
out = out.sort_values(["subfield_id", "year", "quarter"], kind="mergesort")
out.to_csv("out/data.csv", index=False, float_format="%.3f", lineterminator="\\n")
digest = hashlib.sha256(open("out/data.csv", "rb").read()).hexdigest()
size_mb = len(open("out/data.csv", "rb").read()) / 1e6
print(f"out/data.csv: {len(out):,} rows x {len(out.columns)} columns · {size_mb:.1f} MB")
print("sha256:", digest)"""))

cells.append(MD("""## Limits, and how to build on this

This table is built for seasonality analysis (use `citations_received`), cohort-impact
studies (use `cited_by_sum` with an age model), and field-growth accounting; it is not a
per-paper citation index and ranks no individuals. The cohort measure is right-censored
(young papers are still earning); the event measure ends at the snapshot edge (2026 Q2
partial); citation arrival is dated by the citing work's publication, which lags the citing
decision by months. A future snapshot must ship as a new rail file — this notebook's pins
guarantee its numbers never drift underneath the DOI. Natural next steps: within-year
seasonality by field, cohort quality by publication quarter, and joining the arXiv
176-category layer published in the platform repository."""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
path = sys.argv[1] if len(sys.argv) > 1 else "nb.ipynb"
json.dump(nb, open(path, "w"), indent=1)
src = json.dumps(nb)
print(f"built {path}: {len(cells)} cells, {len(src):,} bytes")
