#!/usr/bin/env python3
"""Unify the 6 ArtaAstro WORLD-EVENT measures into the TOPICS atlas (operator directive 2026-07-07).

The /astro page is purged; its 6 GLOBAL-DAILY GDELT measures (material conflict, any conflict, verbal
conflict, cooperation, material cooperation, violence) become entries in the SAME monthly Topics atlas as
the search topics, so /skills shows them under their fitted sign and renders them EXACTLY like a search
topic (phase -> sign hero, per-body frequencies, fit). We:

  * aggregate world_events_daily_clean.csv (daily GDELT shares, 1979->present) to MONTHLY means aligned to
    the atlas GRID (2004-01 .. 2026-06), scaled share -> PERCENT (x100) so they read 0-100 like interest
    (a linear scale is R^2/phase/frequency invariant — the fit is identical);
  * write each measure's monthly series to analysis/data_monthly/<slug>.csv (so reanalyze + the competition
    builder both find it via the label slug, exactly like a topic);
  * seed a registry record per measure into analysis/_topics_weekly.json with res="weekly", axis="topic",
    world_event=True (flagged for the label), then let reanalyze_topic500_winner.py refit them with the ONE
    constrained sinc-GD model (non-negative weights + frequencies, circular phase) and export_research.py
    carry them into research.json.

Run:  python3 analysis/build_world_measures.py   (then reanalyze_topic500_winner.py + export_research.py)
"""
import importlib.util as _u, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
os.chdir(REPO)
tf = _u.module_from_spec(_u.spec_from_file_location("tf", os.path.join(HERE, "trends_fit.py")))
_u.spec_from_file_location("tf", os.path.join(HERE, "trends_fit.py")).loader.exec_module(tf)

CLEAN = "analysis/artaastro/competition/world_events_daily_clean.csv"
POOL = "analysis/_topics_weekly.json"
# (registry key = label slug, human label, GDELT column). These 6 are the competition's world-event measures.
MEASURES = [
    ("Material conflict", "material"),
    ("Any conflict", "conflict"),
    ("Verbal conflict", "verbal_conf"),
    ("Cooperation", "cooperation"),
    ("Material cooperation", "material_coop"),
    ("Violence", "violence"),
]


def main():
    grid = pd.DatetimeIndex(tf.GRID)                                  # 2004-01 .. 2026-06 (270 months)
    daily = pd.read_csv(CLEAN, parse_dates=["date"]).sort_values("date")
    monthly = daily.set_index("date").resample("MS").mean(numeric_only=True)   # daily shares -> monthly means
    reg = json.load(open(POOL))
    lon = tf.ephemeris()
    added = []
    for label, col in MEASURES:
        key = tf.slug(label)
        v = monthly[col].reindex(grid)
        v = (pd.to_numeric(v, errors="coerce") * 100.0).interpolate(limit_direction="both").ffill().bfill()  # share -> percent
        y_full = v.to_numpy(float)
        assert np.isfinite(y_full).all() and len(y_full) == len(grid), f"{label}: bad series"
        # the monthly CSV (Time, v) — reanalyze + make_*_comp load it by the label slug, exactly like a topic
        pd.DataFrame({"Time": grid, "v": np.round(y_full, 4)}).to_csv(f"{tf.DATA_DIR}/{key}.csv", index=False)
        # seed the base record with the ORIGINAL model (gives it tuning/sign_scores/period like a topic);
        # reanalyze_topic500_winner.py then overwrites the model fields with the constrained sinc-GD fit.
        y = y_full[:-tf.DROP_LAST]                                    # the fit window (drop the recency year)
        rec = tf.fit_topic(y, lon)
        rec.update(key=key, label=label, res="weekly", axis="topic", world_event=True, system="world-events",
                   pos="noun", topics=[], freq=0, popularity=round(float(np.mean(y_full)), 1))
        rec["shares"] = tf.body_shares(rec, lon, y)
        reg[key] = rec
        added.append((label, key, rec["sign"], round(rec["r2"] * 100)))
    json.dump(reg, open(POOL, "w"), indent=1)
    print(f"seeded {len(added)} world-event measures into {POOL} + {tf.DATA_DIR}/ (base fit; reanalyze refits with sinc-GD):")
    for label, key, sign, r2 in added:
        print(f"  {label:22s} {key:22s} base {sign:11s} R2 {r2}%")


if __name__ == "__main__":
    main()
