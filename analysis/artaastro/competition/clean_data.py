#!/usr/bin/env python3
"""
clean_data.py — turn the raw GDELT global daily aggregate into a CLEAN, documented daily/global
conflict dataset for the ArtaAstro competition.

Input : ../out/world_conflict_daily.csv   (from fetch_gdelt.py — entire GDELT history, by SQLDATE)
Output: world_events_daily_clean.csv       (one row per day, derived conflict measures)
        DATASET.md                          (human-readable data dictionary + cleaning log)

Cleaning decisions (all logged to DATASET.md, all deliberately conservative and reversible):
  * SHARE-based measures only. GDELT's raw event volume grows ~1000x from 1979 (~1k events/day) to
    the 2015+ era (~100k/day) as it ingests more sources. Absolute counts are therefore dominated by
    that media-growth trend, not by the world. Every measure here is a *fraction of that day's events*
    (or a per-event average), so the media-volume trend cancels and days are comparable across eras.
  * MIN_EVENTS floor. A share computed from very few events is noise, so days with fewer than
    MIN_EVENTS coded events are dropped (this only bites the sparse earliest years).
  * Trailing-edge trim. The final few days of the feed are still filling in (events get added for a
    date over the following days), so the last TRIM_TAIL days are dropped as incomplete.
  * No smoothing, no winsorising, no outlier removal. Real spikes (e.g. 2022-02-24) stay in.

Measures (per day):
  material   = QuadClass4 / total          fraction of events that are MATERIAL CONFLICT
  conflict   = (QuadClass3+QuadClass4)/tot  fraction that are ANY conflict (verbal + material)
  violence   = EventRootCode{18,19,20}/tot  fraction that are assault / fight / mass violence
  neg_tone   = -mean(AvgTone)              average adverse-ness of coverage (higher = darker)
  neg_gold   = -mean(GoldsteinScale)       average conflict on Goldstein's cooperation<->conflict axis
"""
import os, csv, datetime as dt
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(HERE, "..", "out", "world_conflict_daily.csv")
OUT  = os.path.join(HERE, "world_events_daily_clean.csv")
DOC  = os.path.join(HERE, "DATASET.md")

MIN_EVENTS = 200     # a day needs >=200 coded events for a stable share
TRIM_TAIL  = 6       # drop the last 6 calendar days (still filling in)

def main():
    df = pd.read_csv(RAW, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    n0 = len(df)
    span0 = (df["date"].min().date(), df["date"].max().date())

    # trailing-edge trim
    last_keep = df["date"].max() - pd.Timedelta(days=TRIM_TAIL)
    df = df[df["date"] <= last_keep]
    n_trim = n0 - len(df)

    # min-events floor
    below = df["n_events"] < MIN_EVENTS
    n_below = int(below.sum())
    below_era = df.loc[below, "date"].dt.year.value_counts().sort_index()
    df = df[~below].copy()

    # derived share measures — the 8 "topics" the competition scores (QuadClass 1=verbal-coop,
    # 2=material-coop, 3=verbal-conflict, 4=material-conflict)
    df["material"]    = df["n_q4"] / df["n_events"]                    # material conflict share
    df["conflict"]    = (df["n_q3"] + df["n_q4"]) / df["n_events"]     # any conflict share
    df["verbal_conf"] = df["n_q3"] / df["n_events"]                   # verbal conflict share
    df["cooperation"] = (df["n_q1"] + df["n_q2"]) / df["n_events"]    # any cooperation share (control)
    df["material_coop"] = df["n_q2"] / df["n_events"]                 # material cooperation share (control)
    df["violence"]    = df["n_root1820"] / df["n_events"]             # assault/fight/mass-violence share
    df["neg_tone"]    = -(df["sum_tone"] / df["n_events"])            # adverse coverage
    df["neg_gold"]    = -(df["sum_goldstein"] / df["n_events"])       # Goldstein conflict axis

    # calendar-gap report (missing days inside the kept span)
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    gaps = sorted(set(full) - set(df["date"]))

    MEASURES = ["material","conflict","verbal_conf","cooperation","material_coop","violence","neg_tone","neg_gold"]
    cols = ["date","n_events"] + MEASURES
    out = df[cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    for c in MEASURES:
        out[c] = out[c].round(6)
    out.to_csv(OUT, index=False)

    # regime summary (pre/post 2013-04-01)
    pre  = df[df["date"] < "2013-04-01"]
    post = df[df["date"] >= "2013-04-01"]
    def desc(x, c): return f"{x[c].mean():.4f} ± {x[c].std():.4f}" if len(x) else "—"

    with open(DOC, "w") as f:
        w = f.write
        w("# ArtaAstro competition — dataset\n\n")
        w("Global **daily** conflict aggregates derived from the **entire GDELT 1.0 event history** "
          "(1979→present), aggregated by event date (SQLDATE) across all ~500M coded events. "
          "Built by `../fetch_gdelt.py` (streaming, ~52 GB) then cleaned by `clean_data.py`.\n\n")
        w("## Rows\n\n")
        w(f"- Raw days aggregated: **{n0:,}** ({span0[0]} → {span0[1]})\n")
        w(f"- Dropped as trailing-edge incomplete (last {TRIM_TAIL} days): {n_trim}\n")
        w(f"- Dropped for < {MIN_EVENTS} events/day (unstable share): **{n_below:,}** "
          f"(by year: {dict(below_era) if n_below else 'none'})\n")
        w(f"- **Clean days kept: {len(out):,}** ({out['date'].iloc[0]} → {out['date'].iloc[-1]})\n")
        w(f"- Internal calendar gaps remaining: {len(gaps)}"
          + (f" (e.g. {gaps[0].date()}…)" if gaps else "") + "\n\n")
        w("## Columns\n\n")
        w("| column | meaning |\n|---|---|\n")
        w("| `date` | UTC calendar day (event occurrence date, GDELT SQLDATE) |\n")
        w("| `n_events` | total coded events that day (reliability weight; not a target) |\n")
        w("| `material` | **QuadClass 4** share — fraction of events that are *material conflict* |\n")
        w("| `conflict` | **QuadClass 3+4** share — fraction that are *any* conflict (verbal+material) |\n")
        w("| `verbal_conf` | **QuadClass 3** share — verbal conflict (demands, threats, protests) |\n")
        w("| `cooperation` | **QuadClass 1+2** share — any cooperation (a control: astrology should predict this no better) |\n")
        w("| `material_coop` | **QuadClass 2** share — material cooperation (aid, investment) (control) |\n")
        w("| `violence` | **CAMEO root 18/19/20** share — assault / fight / unconventional mass violence |\n")
        w("| `neg_tone` | −mean(AvgTone) — average adverse-ness of coverage (higher = darker) |\n")
        w("| `neg_gold` | −mean(GoldsteinScale) — average position on cooperation↔conflict axis |\n\n")
        w("These 8 measures are the competition's **topics** (each scored by its own phase-tuning curve).\n\n")
        w("## Why shares, not counts\n\n")
        w("GDELT's raw daily event count grows ~1000× over the record as it adds sources; that is a "
          "media-coverage trend, not a world trend. Every measure above is a within-day fraction (or "
          "per-event mean), so that growth cancels and 1979 is comparable to 2025.\n\n")
        w("## Known regime shift (documented, not corrected)\n\n")
        w("GDELT's own pipeline changed at **2013-04-01** (historical backfile → live daily feed). "
          "Mean levels differ across that boundary:\n\n")
        w("| measure | 1979–2013 | 2013–now |\n|---|---|---|\n")
        for c in ["material","conflict","verbal_conf","cooperation","material_coop","violence","neg_tone","neg_gold"]:
            w(f"| {c} | {desc(pre,c)} | {desc(post,c)} |\n")
        w("\nThe competition holdout is entirely in the modern (post-2013) regime, so this shift is a "
          "property of the *training* history that entrants must handle — exactly as in the real world.\n\n")
        w("_Source: GDELT Project (data.gdeltproject.org), CC-BY. Aggregation + cleaning: ArtaAstro._\n")

    print(f"clean days: {len(out):,}  ({out['date'].iloc[0]} → {out['date'].iloc[-1]})")
    print(f"dropped: {n_trim} tail + {n_below} low-event")
    print(f"wrote {OUT} and {DOC}")

if __name__ == "__main__":
    main()
