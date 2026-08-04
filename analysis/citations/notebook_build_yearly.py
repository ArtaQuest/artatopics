#!/usr/bin/env python3
"""Builds the dataset-kind notebook for the YEARLY citations publication (max depth, 1000-2026)."""
import json, sys

MD = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
CODE = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = []

cells.append(MD("""# Citations per year across the scholarly record, 1000–2026

How far back can citation data honestly go? This notebook builds the **yearly** citation
dataset for every research subfield over the deepest axis the record supports — from the
year 1000 to the 2026-06-26 snapshot. Two base measures per (subfield, year): the citations
that year's publications eventually **earned** (dated by the publishing cohort), and the
citations **received** that year — 3.0 billion reference edges, each dated by the *citing*
work's publication year — plus five derived measures the raw record does not carry:
cohort impact, date-precision share, share-of-year, ten-year growth, and within-year
arrival balance.

Yearly resolution has a quiet superpower: the date-precision artefact that haunts finer
bins vanishes. A third of all works carry only a year (stored as Jan 1) — at quarterly
grain they must sit in a separate "quarter 0" bucket, but a yearly table absorbs every
work with a known year. The deep past becomes fully usable. The table is **built here, in
the cells**: the platform's quarterly citations rail (`data/citations-quarterly.csv`,
OpenAlex snapshot 2026-06-26, CC0) is summed over quarters — the quarter-0 bucket folded
into each year, mis-dated future rows dropped — then cross-checked cell-for-cell against
the platform's compact yearly rail (`data/citations-yearly.csv`) before the curated table
lands in `out/data.csv` with its datasheet below."""))

cells.append(CODE("""import hashlib
import matplotlib.pyplot as plt
import pandas as pd

# One mid-grey ink for annotations so dark and light themes both read it (ArtaContrast).
GREY, GOLD, BLUE = "#808080", "#E8B923", "#1746DC"
plt.rcParams.update({"text.color": GREY, "axes.edgecolor": GREY, "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "grid.color": GREY,
                     "figure.dpi": 110, "figure.facecolor": (0, 0, 0, 0),
                     "axes.facecolor": (0, 0, 0, 0), "savefig.transparent": True,
                     "axes.grid": True, "grid.alpha": 0.4, "font.size": 11})

quarterly = pd.read_csv("data/citations-quarterly.csv", dtype={"subfield_id": str, "field_id": str})

# THE BUILD. Trim mis-dated future rows (2026 Q3/Q4 and years 2027-2030 cannot exist in a
# 2026-06-26 snapshot), then sum every quarter INCLUDING the year-only bucket (quarter 0)
# into (subfield, year). The record's floor is the year 1000 — kept in full.
pinned = quarterly[(quarterly.year <= 2026) & ~((quarterly.year == 2026) & (quarterly.quarter > 2))]
yearly = pinned.groupby(["subfield_id", "subfield", "field_id", "field", "domain", "year"],
                        as_index=False)[["works", "cited_by_sum", "citations_received"]].sum()
# Quarter composition, carried alongside for the derived measures: year-only works (q0),
# and dated arrivals split so the within-year balance survives the yearly sum.
comp = pinned.assign(works_q0=pinned.works.where(pinned.quarter == 0, 0),
                     recv_dated=pinned.citations_received.where(pinned.quarter > 0, 0),
                     recv_h2=pinned.citations_received.where(pinned.quarter >= 3, 0)) \
             .groupby(["subfield_id", "year"], as_index=False)[["works_q0", "recv_dated", "recv_h2"]].sum()
yearly = yearly.merge(comp, on=["subfield_id", "year"])

# Pin the build to the snapshot: shape, bounds, totals, three spot cells across three centuries.
assert len(yearly) == 62_835 and yearly.subfield_id.nunique() == 253
assert int(yearly.year.min()) == 1000 and int(yearly.year.max()) == 2026
assert yearly.works.sum() == 315_098_982 and yearly.cited_by_sum.sum() == 2_874_825_415
assert yearly.citations_received.sum() == 3_005_254_318
spot = yearly.set_index(["subfield_id", "year"])
assert tuple(spot.loc[("1202", 1800)][["works", "cited_by_sum", "citations_received"]]) == (480, 320, 3)
assert tuple(spot.loc[("2735", 1900)][["works", "cited_by_sum", "citations_received"]]) == (228, 440, 138)
assert tuple(spot.loc[("1702", 2020)][["works", "cited_by_sum", "citations_received"]]) == (187_469, 2_219_154, 3_327_601)

# Independent cross-check: the platform's compact yearly rail must agree cell-for-cell.
rail_y = pd.read_csv("data/citations-yearly.csv", dtype={"subfield_id": str, "field_id": str})
key = ["subfield_id", "year"]
measures = ["works", "cited_by_sum", "citations_received"]
a = yearly.sort_values(key, kind="mergesort").reset_index(drop=True)
b = rail_y.sort_values(key, kind="mergesort").reset_index(drop=True)
assert a[key].equals(b[key]) and a[measures].equals(b[measures])
yearly = a
print(f"built {len(yearly):,} subfield-years from {len(pinned):,} quarterly cells · "
      f"{yearly.subfield_id.nunique()} subfields · years {yearly.year.min()}-{yearly.year.max()} · "
      f"{yearly.works.sum():,} works · {yearly.citations_received.sum():,} citations received · "
      f"cross-check vs the yearly rail: exact")"""))

cells.append(MD("""## Two measures, and how deep each one reaches

- **`cited_by_sum`** — the *cohort* measure: citations ever earned by that year's
  publications. Meaningful wherever `works` is; right-censored on the young edge (recent
  years look weak only because their papers are still earning).
- **`citations_received`** — the *event* measure: citations arriving that year, dated by
  the citing work. Its right edge is the snapshot (2026 is Q1–Q2 only).

Depth and reliability are not the same thing. Works counts stretch to the year 1000; the
received series only becomes a real signal in the nineteenth century, because a citation
event needs a *dated citing work* — and the code below shows what the record looks like
before that."""))

cells.append(CODE("""era = yearly.assign(c=(yearly.year // 100) * 100).groupby("c")[
    ["works", "cited_by_sum", "citations_received"]].sum()
print("coverage by century:")
print(era.to_string(formatters={c: "{:,.0f}".format for c in era.columns}))
w = yearly.groupby("year").works.sum()
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(w.index, w.values.clip(min=1), color=BLUE, linewidth=1.6)
ax.set_yscale("log"); ax.set_xlabel("year"); ax.set_ylabel("works published (log)")
ax.set_title("A millennium of publishing, one line")
plt.show()"""))

cells.append(MD("""## The reliability gradient — kept, and flagged

"As far back as possible" means keeping rows a cautious analyst might trim, and saying so
plainly. Before 1500 the table holds a few hundred subfield-years (printed below) — partly
genuine early works, partly mis-dated records. The received column is noise there: the year
1700 records zero citation events while a handful of "medieval" events exist only because a
few citing works are mis-dated. Digitisation batches also leave spikes — 1727 carries an
economics burst that is an archive artefact, not an eighteenth-century boom. The rows stay;
the reliability gradient is yours to respect."""))

cells.append(CODE("""pre = yearly[yearly.year < 1500]
print(f"pre-1500: {len(pre)} subfield-years · {pre.works.sum():,} works · "
      f"{pre.citations_received.sum()} received events")
print(f"received in 1700: {yearly[yearly.year == 1700].citations_received.sum()}")
b = yearly[(yearly.subfield_id == '2002') & (yearly.year == 1727)]
print(f"the 1727 batch artefact — Economics works that year: {int(b.works.iloc[0]):,}")
r = yearly.groupby("year").citations_received.sum()
r = r[(r.index >= 1800) & (r.index <= 2025)]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(r.index, r.values.clip(min=1), color=GOLD, linewidth=1.8)
ax.set_yscale("log"); ax.set_xlabel("year of the citing work")
ax.set_ylabel("citations received (log)")
ax.set_title("The event series, where it becomes real: 1800-2025")
plt.show()
print(f"received 1975: {r.loc[1975]:,} -> 2025: {r.loc[2025]:,} = "
      f"x{r.loc[2025] / r.loc[1975]:.1f} in 50 years")"""))

cells.append(MD("""## The domain mix is not standing still

Citation traffic grew 47-fold in the last half-century — but not evenly. Splitting the
received series across OpenAlex's four domains: Physical Sciences climb from about a third
of all citation traffic in 1950 to the largest share in 2025, Health Sciences fall from the
largest share to about a fifth, and Social Sciences grow from under a tenth to about a
sixth. The printed shares quantify the whole shift."""))

cells.append(CODE("""dom = yearly[yearly.domain != "unclassified"].groupby(["domain", "year"]).citations_received.sum().unstack(0).fillna(0)
dom = dom[(dom.index >= 1950) & (dom.index <= 2025)]
shares = 100 * dom.div(dom.sum(axis=1), axis=0)
colours = {"Physical Sciences": BLUE, "Health Sciences": GOLD,
           "Life Sciences": GREY, "Social Sciences": "#4A6B8A"}
fig, ax = plt.subplots(figsize=(7, 3.6))
for name, col in colours.items():
    ax.plot(shares.index, shares[name], color=col, linewidth=2, label=name)
ax.legend(frameon=False, labelcolor=GREY, fontsize=9)
ax.set_xlabel("year of the citing work"); ax.set_ylabel("share of citations received (%)")
ax.set_title("Four domains' share of the world's citation traffic")
plt.show()
for y in (1950, 2025):
    row = shares.loc[y]
    print(f"{y}: " + " · ".join(f"{k} {v:.1f}%" for k, v in row.items()))"""))

cells.append(MD("""## Datasheet

### Schema
`out/data.csv`, one row per (subfield, year), sorted by `subfield_id` then `year`. Nine
recorded columns: `subfield_id` (OpenAlex subfield, string; `none` = works with no primary
topic) · `subfield` · `field_id`, `field`, `domain` (parents) · `year` (1000–2026) ·
`works` (count published) · `cited_by_sum` (lifetime citations of that cohort, integer) ·
`citations_received` (event-dated arrivals that year, integer). Five derived columns this
dataset adds over the raw record: `cites_per_work` (= cited_by_sum ÷ max(works, 1), 3 dp;
0 where the cohort is empty) · `works_dated_share` (fraction of the cohort carrying a real
sub-year date) · `received_share_of_year` (this subfield's fraction of ALL citations
received that year, 6 dp — cross-field comparison with global growth removed) ·
`received_growth10` (annualised 10-year growth of received, blank where the decade-earlier
baseline is zero or absent) · `received_h2_share` (share of the year's *date-resolved*
arrivals landing in July–December, blank where none are date-resolved).

### Provenance
Built in the cells above from the platform's quarterly citations rail (itself derived from
the complete OpenAlex works snapshot of **2026-06-26**: 510.4M records, 317.8M canonical
works kept, 3.005B reference edges joined). The exact rule: drop mis-dated future rows
(2026 Q3/Q4 and years 2027–2030, which a 2026-06-26 snapshot cannot contain), then sum
works and both citation measures over all remaining quarters — including quarter 0, the
year-only precision bucket — per (subfield, year). The build is pinned by row count,
totals and three spot cells, and verified cell-for-cell against the platform's independent
compact yearly rail. Upstream regeneration: `analysis/citations/` in the platform repository.

### Licence
OpenAlex metadata is **CC0**; this derived table is likewise **CC0** — cite the DOI so
readers can find the pinned snapshot.

### Units
Counts are integers of works and citation events; `cites_per_work` is a dimensionless
ratio. Years are calendar years (CE, Gregorian)."""))

cells.append(CODE("""out = yearly.copy()
out["cites_per_work"] = (out.cited_by_sum / out.works.clip(lower=1)).round(3)
out["works_dated_share"] = (1 - out.works_q0 / out.works.clip(lower=1)).round(3)
year_total = out.groupby("year").citations_received.transform("sum")
out["received_share_of_year"] = (out.citations_received / year_total.clip(lower=1)).round(6)
lag = out[["subfield_id", "year", "citations_received"]].assign(year=out.year + 10) \
        .rename(columns={"citations_received": "recv_lag10"})
out = out.merge(lag, on=["subfield_id", "year"], how="left")
grow = (out.citations_received / out.recv_lag10) ** 0.1 - 1
out["received_growth10"] = grow.where((out.recv_lag10 > 0) & (out.citations_received > 0)).round(4)
out["received_h2_share"] = (out.recv_h2 / out.recv_dated.where(out.recv_dated > 0)).round(3)
out = out.drop(columns=["works_q0", "recv_dated", "recv_h2", "recv_lag10"])
out = out.sort_values(["subfield_id", "year"], kind="mergesort")
out.to_csv("out/data.csv", index=False, lineterminator="\\n")
raw = open("out/data.csv", "rb").read()
print(f"out/data.csv: {len(out):,} rows x {len(out.columns)} columns · {len(raw) / 1e6:.1f} MB")
print("sha256:", hashlib.sha256(raw).hexdigest())
recent = out[out.year == 2025]
print("largest share of 2025's citation traffic:")
print(recent.nlargest(3, "received_share_of_year")[["subfield", "received_share_of_year"]].to_string(index=False))
print("fastest 10-year received growth into 2025 (>=10k events):")
print(recent[recent.citations_received >= 10_000].nlargest(3, "received_growth10")[["subfield", "received_growth10"]].to_string(index=False))"""))

cells.append(MD("""## Uses, limits, and what to build next

Built for long-horizon seasonality and growth analysis (`received_share_of_year` and
`received_growth10` are the ready-made comparands — global growth and field size already
divided out), cohort studies (`cited_by_sum` with an age model; `works_dated_share` tells
you how much sub-year precision each cohort can support), and field-history accounting.
It is not a per-paper index and ranks no individuals. Respect the reliability gradient
(works to 1000, events from ~1800), the cohort's right-censoring, and the partial 2026.
A future snapshot must ship as a new rail file — the pins above freeze this one under its
DOI. For full within-year timing, use this dataset's quarterly sibling, which keeps the
year-only bucket explicit as quarter 0."""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
path = sys.argv[1] if len(sys.argv) > 1 else "nb.ipynb"
json.dump(nb, open(path, "w"), indent=1)
print(f"built {path}: {len(cells)} cells")
