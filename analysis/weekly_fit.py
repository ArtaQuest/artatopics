#!/usr/bin/env python3
"""
Weekly-resolution run for the TOP-100 topics (by representativeness).
=====================================================================
Google Trends only returns WEEKLY data for windows shorter than ~5 years, each normalised to its OWN max=100. To get
one continuous weekly series over 2004→now we fetch overlapping ~4-year weekly chunks and STITCH them, using the
cached all-time MONTHLY export as the backbone:

  1. rescale each chunk to the monthly scale (least-squares through-origin over the chunk↔monthly overlap),
  2. average overlapping chunks onto the weekly grid,
  3. MONTHLY-ANCHOR — scale every month's weeks so their mean equals the monthly backbone for that month. This makes
     the weekly→monthly resample equal the backbone (by construction) while preserving within-month weekly shape.

VALIDATION (the point of the warning): the quality signal is the PRE-anchor correlation between the combined weekly
series resampled to monthly and the monthly backbone — high corr ⇒ the chunks scaled consistently (a good stitch);
low corr ⇒ a bad chunk/seam. We store it as `stitch_corr` and print it for every topic.

Everything else (sinc design, MSE fit, variance shares, representativeness, pages) is the verified trends_fit.py
logic, reused at weekly resolution by monkeypatching its globals (GRID/DROP_LAST/EPHEM/DATA_DIR).

  python3 analysis/weekly_fit.py            # stitch + fit the top-100 (resumes via cached chunks)
  python3 analysis/weekly_fit.py --n 3      # just the first 3 (a quick stitch-quality check)
  python3 analysis/weekly_fit.py --stitch-only --n 3   # stitch + validate only, no fit
"""
import importlib.util as _u, os, sys
import numpy as np, pandas as pd
_spec = _u.spec_from_file_location("tf", os.path.join(os.path.dirname(__file__) or ".", "trends_fit.py"))
tf = _u.module_from_spec(_spec); _spec.loader.exec_module(tf)

# ── weekly config: patch tf globals so ALL of tf's logic runs at weekly resolution ──
GRID_W = pd.date_range("2004-01-04", "2026-06-24", freq="W-SUN")   # Sunday-anchored, matches Trends' weekly dates
tf.GRID = GRID_W
import json as _json                                               # drop the most recent ~year of weeks (recency bias);
tf.DROP_LAST = (_json.load(open("analysis/_period_scale.json")).get("drop", 52)   # tuned jointly with PERIOD_SCALE
                if os.path.exists("analysis/_period_scale.json") else 52)         # (tune_period.py), else default 52
tf.EPHEM = "analysis/_ephemeris_weekly.csv"
tf.DATA_DIR = "analysis/data_weekly"                              # tf.load_y / tf.fetch_daily read stitched weekly here
MONTHLY_DIR = "analysis/data_monthly"
START, END = tf.START, tf.END
os.makedirs(tf.DATA_DIR, exist_ok=True)

def windows(span=1490, step=720):
    """Overlapping date windows covering START→END; ~4-yr span (⇒ weekly) with ~770-day (≈50%) overlap — denser than
       before so every seam is over-determined and the stitch correlation is high (more sampling ⇒ a cleaner stitch)."""
    out = []; a = pd.Timestamp(START)
    while True:
        b = min(a + pd.Timedelta(days=span), END)
        out.append((a, b))
        if b >= END: break
        a = a + pd.Timedelta(days=step)
    return out

def stitch(query, s):
    """Stitch overlapping weekly chunks onto GRID_W, monthly-anchored to the backbone. Returns (df, stitch_corr) or
       (None, reason). stitch_corr = pre-anchor corr(weekly→monthly, monthly backbone) ∈ [-1,1] (the quality check)."""
    mp = f"{MONTHLY_DIR}/{tf.slug(query)}.csv"
    if not os.path.exists(mp): return None, "no-monthly-backbone"
    M = pd.read_csv(mp); M["Time"] = pd.to_datetime(M["Time"]); M = M.drop_duplicates("Time").set_index("Time")["v"].astype(float)
    chunks = []
    for a, b in windows():
        c = tf.fetch_single(query, s, f"{a.date()} {b.date()}")   # SINGLE-field weekly chunks → max-precision trend shape
        if c is None or c.empty: continue
        c = c.copy(); c["Time"] = pd.to_datetime(c["Time"]); c = c.drop_duplicates("Time").set_index("Time")["v"].astype(float)
        cm = c.resample("MS").mean()
        ov = [d for d in cm.index.intersection(M.index) if cm.get(d, 0) > 0 and np.isfinite(M.get(d, np.nan))]
        if len(ov) < 6: continue                                  # need enough overlap to scale this chunk
        cv = cm.loc[ov].to_numpy(); mv = M.loc[ov].to_numpy()
        denom = float(cv @ cv)
        if denom <= 0: continue
        scale = float((cv @ mv) / denom)                          # least-squares through origin: chunk·scale ≈ monthly
        if scale > 0: chunks.append(c * scale)
    if not chunks: return None, "no-usable-chunks"
    W = pd.concat(chunks, axis=1).mean(axis=1)                     # average overlapping chunks (now on ~monthly scale)
    W = W.reindex(GRID_W).interpolate(limit_direction="both")
    Wm = W.resample("MS").mean()                                   # ── VALIDATE: pre-anchor weekly→monthly vs backbone
    j = pd.concat([Wm.rename("w"), M.rename("m")], axis=1).dropna()
    corr = float(j["w"].corr(j["m"])) if len(j) > 3 else 0.0
    factor = (M / Wm).replace([np.inf, -np.inf], np.nan)          # ── MONTHLY-ANCHOR each month's weeks to the backbone
    f = factor.reindex(pd.DatetimeIndex(W.index.to_period("M").to_timestamp())).to_numpy()
    f = np.where(np.isfinite(f), f, 1.0)
    return pd.DataFrame({"Time": GRID_W, "v": W.to_numpy() * f}), corr

# NOTE: this module now exists ONLY for `stitch()` (used by collect_weekly.py). The old standalone run loop (the
# monthly-funnel `load_reg`/`render_topic`/`rebuild` flow) was retired with the topic-centric pipeline.
if __name__ == "__main__":
    print("weekly_fit.py is a library now (stitch()). Run the pipeline with analysis/collect_weekly.py.")
