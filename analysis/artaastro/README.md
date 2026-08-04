# ArtaAstro — sidereal-Lahiri day-by-day mundane model + honest backtest

ArtaAstro computes, for **every day from 25 Feb 2020 to 9 Feb 2253** (and back to 1979 for testing),
a sidereal **Lahiri** (Vedic/Jyotish) sky reading and an **event-intensity** score, then measures —
honestly — whether that score has any skill at anticipating real world events.

## The honest framing (read this first)

Astrology has **no demonstrated skill** at predicting world events. This project does not pretend
otherwise. What makes the exercise worth doing is doing it *correctly*:

1. **The model is fixed a priori.** The rule set in [`intensity.py`](intensity.py) — which planetary
   configurations count and how much — is classical mundane astrology, written down and frozen
   **before** the ground-truth data was ever loaded. It is never tuned, fitted, or selected against
   the test data. (Tuning it to fit history would be the cardinal sin that makes every "it predicted
   X!" claim worthless.)
2. **The ground truth is objective and huge.** We test against the **entire GDELT history** — global
   daily aggregates of ~all machine-coded world events, 1979→today (~17,000 overlapping days), not a
   hand-picked list of disasters.
3. **Significance uses the right null.** Both series are autocorrelated, so we compare the observed
   correlation to a **circular-shift permutation null** (5,000 random time-shifts of the astro signal),
   which is the correct way to ask "is this better than a coincidentally-aligned cycle?"
4. **The result is reported straight** in [`RESULTS.md`](RESULTS.md), whatever it is.

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| Sidereal ephemeris + features | [`build.py`](build.py) | `out/daily_intensity.csv` (100,117 days), `out/astro-*.json` chunks, `out/astro-notable.json` |
| A-priori intensity rules | [`intensity.py`](intensity.py) | (imported by build) |
| Ground truth | [`fetch_gdelt.py`](fetch_gdelt.py) | `out/world_conflict_daily.csv` |
| The test | [`backtest.py`](backtest.py) | `out/backtest.json`, `RESULTS.md`, `out/plot_*.png` |

### Engine
**Kerykeion** (`zodiac_type="Sidereal", sidereal_mode="LAHIRI"`) for Sun…Pluto — verified to match raw
Swiss Ephemeris to the decimal and to cover the full 1979→2253 span. Rahu/Ketu (lunar nodes) and
eclipses come from the same underlying `pyswisseph`. Each day is cast for **10:00 America/New_York**
(the platform's Season-reset instant), DST-aware.

### The intensity model (mundane astrology, a priori)
Score = a saturating sum of: hard aspects (0/90/180°, orb ≤3°) between the slow bodies — Saturn–Pluto
weighted highest, then Saturn–Uranus, Jupiter–Saturn (Great Conjunction), Mars–Saturn/Pluto, node
contacts…; **eclipses** (heavier on a malefic); slow-planet **ingresses**; and outer-planet
**stations**. Full weight table and rationale in [`intensity.py`](intensity.py). Sanity: the model's
top 2020-era day is the **12 Jan 2020 Saturn–Pluto conjunction** (orb 0.0°); it does **not** flag the
24 Feb 2022 Ukraine invasion (score ~6) — exactly the kind of honesty the null test is built to expose.

### Ground truth (GDELT, entire history)
`fetch_gdelt.py` streams every GDELT 1.0 event file (yearly 1979-2005, monthly →2013, daily →today;
~52 GB), aggregating **by SQLDATE** into a global daily series: counts by CAMEO QuadClass (material
conflict = Q4), Goldstein scale, and tone. Share-based measures (Q4 / total) neutralise GDELT's ~1000×
growth in raw volume. Streaming + resumable + bounded timeouts (never stores the corpus).

## Reproduce
```bash
python3 fetch_gdelt.py     # ~1h, background; resumable; -> out/world_conflict_daily.csv
python3 build.py           # ~2min           -> out/daily_intensity.csv + chunks
python3 backtest.py        # seconds         -> RESULTS.md + backtest.json + plots
```

## Scope note
This pass is the **research core** (generator + rules + backtest + data). The `out/astro-*.json`
chunks are generated now so a later pass can wire the `/astro` SPA page + REST route with no
recomputation. Data is CC-BY GDELT-derived; the ephemeris is deterministic.
