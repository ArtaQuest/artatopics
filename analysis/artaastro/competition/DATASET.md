# ArtaAstro competition — dataset

Global **daily** conflict aggregates derived from the **entire GDELT 1.0 event history** (1979→present), aggregated by event date (SQLDATE) across all ~500M coded events. Built by `../fetch_gdelt.py` (streaming, ~52 GB) then cleaned by `clean_data.py`.

## Rows

- Raw days aggregated: **17,353** (1979-01-01 → 2026-07-05)
- Dropped as trailing-edge incomplete (last 6 days): 6
- Dropped for < 200 events/day (unstable share): **0** (by year: none)
- **Clean days kept: 17,347** (1979-01-01 → 2026-06-29)
- Internal calendar gaps remaining: 0

## Columns

| column | meaning |
|---|---|
| `date` | UTC calendar day (event occurrence date, GDELT SQLDATE) |
| `n_events` | total coded events that day (reliability weight; not a target) |
| `material` | **QuadClass 4** share — fraction of events that are *material conflict* |
| `conflict` | **QuadClass 3+4** share — fraction that are *any* conflict (verbal+material) |
| `verbal_conf` | **QuadClass 3** share — verbal conflict (demands, threats, protests) |
| `cooperation` | **QuadClass 1+2** share — any cooperation (a control: astrology should predict this no better) |
| `material_coop` | **QuadClass 2** share — material cooperation (aid, investment) (control) |
| `violence` | **CAMEO root 18/19/20** share — assault / fight / unconventional mass violence |
| `neg_tone` | −mean(AvgTone) — average adverse-ness of coverage (higher = darker) |
| `neg_gold` | −mean(GoldsteinScale) — average position on cooperation↔conflict axis |

These 8 measures are the competition's **topics** (each scored by its own phase-tuning curve).

## Why shares, not counts

GDELT's raw daily event count grows ~1000× over the record as it adds sources; that is a media-coverage trend, not a world trend. Every measure above is a within-day fraction (or per-event mean), so that growth cancels and 1979 is comparable to 2025.

## Known regime shift (documented, not corrected)

GDELT's own pipeline changed at **2013-04-01** (historical backfile → live daily feed). Mean levels differ across that boundary:

| measure | 1979–2013 | 2013–now |
|---|---|---|
| material | 0.1399 ± 0.0257 | 0.1431 ± 0.0124 |
| conflict | 0.2647 ± 0.0342 | 0.2712 ± 0.0153 |
| verbal_conf | 0.1247 ± 0.0158 | 0.1280 ± 0.0074 |
| cooperation | 0.7353 ± 0.0342 | 0.7288 ± 0.0153 |
| material_coop | 0.0914 ± 0.0153 | 0.1152 ± 0.0067 |
| violence | 0.0836 ± 0.0218 | 0.0812 ± 0.0118 |
| neg_tone | -5.1180 ± 0.4579 | 1.4067 ± 1.5621 |
| neg_gold | -0.6205 ± 0.4067 | -0.5282 ± 0.1833 |

The competition holdout is entirely in the modern (post-2013) regime, so this shift is a property of the *training* history that entrants must handle — exactly as in the real world.

_Source: GDELT Project (data.gdeltproject.org), CC-BY. Aggregation + cleaning: ArtaAstro._
