# Citations per topic per quarter — the full scholarly record

Quarterly citation datasets over the **entire history of publications** (usable from the
1500s; dense from the early 1800s; through the 2026-06-26 snapshot), built from the free
OpenAlex snapshot (CC0) — 510,372,821 works read, 320M canonical (non-XPAC) works kept,
and the complete reference graph joined so citations are dated by the **citing** work.
Built for seasonality analysis and to extend the /topics page beyond the arXiv-only,
no-citation-graph proxy (`analysis/arxivtopics/daily_counts.csv` counts submissions;
this dataset counts real citations).

## The two measures (do not mix them)

| Measure | File suffix | Meaning of a row (topic T, year Y, quarter Q) |
|---|---|---|
| **Cohort** | `_quarter_cohort` | Works **published** in (Y,Q) with primary topic T: how many (`works`), the citations they have earned **to date** (`cited_by_sum`, lifetime as of the snapshot), and their field-normalised impact (`fwci_sum`/`fwci_n`). |
| **Received** | `_quarter_received` | Citations **received** in (Y,Q) by works of topic T: each reference edge is dated by the **citing work's publication quarter** and credited to the **cited work's topic**. This is the true citation-event time series, full history. |

Seasonality note: for "when do papers published get cited more" use cohort (but see
right-censoring below); for "when does citation activity happen" use received.

## Quarter encoding — the year-only artifact

`quarter` is `0..4`. **`0` means year-only date precision**: OpenAlex stores works known
only to a year as `YYYY-01-01`, and that is ~half the corpus at *every* era (probed:
1850 66% real months, 1900 49%, 2000 51%). All Jan-1-dated works land in `quarter=0`
(the ~0.3% genuinely published on Jan 1 are the accepted collateral — stated plainly).
`1..4` are calendar quarters from real dates. Never merge q0 into Q1; treat q0 as
"year known, quarter unknown" mass (distributable if you must, e.g. pro-rata by q1–q4).
The same encoding applies to `received` (a q0 row = citations from year-only-dated
citing works).

## Files

- `topics_taxonomy.csv` — 4,516 topics → 252 subfields → 26 fields → 4 domains (ids+names).
- `openalex_topic_quarter_cohort.csv.gz` — topic_id, topic, year, quarter, works,
  cited_by_sum, fwci_sum, fwci_n. `topic_id=none` = works with no primary topic.
- `openalex_topic_quarter_received.csv.gz` — topic_id, topic, year, quarter,
  citations_received. `none` = cited works unmapped (cited work has no topic, is outside
  the canonical set, or the citing work referenced something not in the snapshot).
- `openalex_{subfield,field,domain,all}_quarter_{cohort,received}.csv.gz` — exact rollups.
- `openalex_topic_year_received_check.csv.gz` — independent validation: OpenAlex's own
  per-work `counts_by_year` (citations received per year, last ~10 years only) summed per
  topic. Compare against the yearly sum of our edge-derived `received` series.
- `arxiv_category_quarter_cohort.csv` — per arXiv category (all 150+ incl. historical
  aliases): submissions (by first-version date — always real quarters, no q0),
  matched_openalex, cited_by_sum. A cross-listed paper counts once per listed category
  (same convention as the /topics submissions matrix).
- `arxiv_category_quarter_received.csv` — citations received per category per quarter,
  1991→2026, via the same edge join (a cited arXiv paper credits every category it lists).
- `build_stats.json` — edge totals, match rates, snapshot date.

## Caveats that matter for analysis

- **Right-censoring (cohort):** recent quarters' `cited_by_sum` is low because those
  papers are young, not because they are weak. Model citation age or cut the last ~3 years.
- **Right edge (received):** the snapshot is 2026-06-26 — 2026q2 is partial and 2026q3+
  empty. Citing works dated in the future (in-press) appear at their stated dates.
- **Citation lag:** received events are dated by the citing work's *publication* date —
  publication pipelines smear the actual "citing decision" by months.
- **XPAC excluded:** OpenAlex's 2026 extended corpus (192.6M `is_xpac` records) is
  excluded everywhere, matching the API's default counts.
- **arXiv matching:** corpus = the librarian-bots HF mirror of the Cornell snapshot
  (3.1M papers, updated 2026-07-17), matched to OpenAlex by DataCite DOI
  (`10.48550/arXiv.<id>`) and journal DOI; an unmerged preprint+published pair
  contributes both works' citations. Papers with neither DOI in OpenAlex stay
  unmatched (`matched_openalex` column tracks coverage per cell).
- Quarters are calendar quarters; they do not align with lunations. For the /topics
  lunation fit this dataset is an annual-seasonality + long-cycle source; re-run the
  harvest at finer granularity if lunation binning of citations is ever needed (the
  pipeline keeps full dates until the binning step in `harvest_openalex.py`).

## Build results (2026-07-22)

- 510,372,821 snapshot rows read (reconciled to the manifest exactly); 317,820,190
  canonical works kept (192.6M XPAC dropped, 2.7M undatable dropped from cohort bins).
- 3,005,254,375 reference edges joined; 95.2% resolved to a cited topic; 59,875,451
  citation events landed on arXiv papers.
- arXiv corpus match: 2,915,545 / 3,107,014 papers (93.8%) linked to OpenAlex works.
- Rows: topic cohort 2,738,206 · topic received 2,130,398 (plus rollups).
- **Validation:** edge-derived yearly received vs OpenAlex's own `counts_by_year`
  (29,452 topic-years ≥1k cites, 2018–2024): median deviation **0.10%**, p90 2.45%.
  Live-API spot-check (`qc_spotcheck.py`, 29 cells): median deviation **0.00%** works,
  **0.28%** citations — residuals are the 3 weeks of API drift since the snapshot.

## Provenance / rebuild

```
python3 analysis/citations/harvest_openalex.py --workers 24   # ~65 GB transfer, hours
python3 analysis/citations/build_quarterly.py                 # merge + edge join
python3 analysis/citations/qc_spotcheck.py                    # 30 live-API cell checks
```

Sources: OpenAlex snapshot 2026-06-26 (s3://openalex, CC0) · arXiv metadata snapshot
(CC0). Run dir `~/.artaquest-dev/openalex-quarterly` (intermediates, ~30 GB, deletable
after build).


## The yearly companion

`GET /aq/v1/citations/yearly` — the quarterly rail summed to YEARS (1000→2026, quarter-0
folded into each year), regenerable from the committed subfield tables by summing over
`quarter`; served from `uploads/aq-data/citations-yearly.csv` (5.2 MB — sized so the offline
relay provisions it reliably; the 21 MB quarterly rail can truncate mid-stream, which the
notebooks' asserts catch by failing closed). Its dataset notebook is built by
`notebook_build_yearly.py` (nb 9303) — which builds the yearly table IN THE CELLS from the
quarterly rail (drop 2026 Q3/Q4 + years 2027-2030, sum all quarters incl. the q0 year-only
bucket), cross-checks it against the compact yearly rail, and ships a 14-column deliverable:
the 9 recorded columns plus 5 derived measures (cites_per_work, works_dated_share,
received_share_of_year, annualised received_growth10, received_h2_share).

## Publication status

The first submission (nb 45) passed the v3 gate (balance 93.2, veto panel 3/3 on round 2 —
round 1 caught a real pin bug, future-dated rows, plus two prose overclaims) but was then
published via a forged operator approval — a process violation. The integrity sweep demoted
it and it was purged on the operator's order (2026-07-23); its minted DOI
10.5281/zenodo.21506420 is orphaned at Zenodo. The notebook (rebuilt deterministically by
`notebook_build.py`, reading the data-shelf rail `GET /aq/v1/citations/quarterly` —
`uploads/aq-data/citations-quarterly.csv` on prod; refreshes must ship under a NEW dated
filename) was resubmitted as nb 9302 after an inline
triple-check (truth fixes: the EE-top-line claim held in every year but 1974; AI/Astronomy
crossing 2002; unverifiable numerics removed) — v3 balance 93.1, veto panel 3/3 on the first
round. Publication is REQUESTED (status pending); the author's emailed single-use secret is
the only mint.
