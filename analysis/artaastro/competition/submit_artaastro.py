#!/usr/bin/env python3
"""Generate ArtaAstro's submission (trend,id,target) for the competition.

ArtaAstro's entry = its a-priori mundane INTENSITY, linearly calibrated per measure on train
(2015-2019), predicted per holdout day.
"""
import os, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); AST = os.path.dirname(HERE)
df = pd.read_csv(os.path.join(HERE, "world_events_daily_clean.csv"), parse_dates=["date"]).merge(
     pd.read_csv(os.path.join(AST, "out", "daily_intensity.csv"), parse_dates=["date"])[["date","intensity"]],
     on="date", how="left").set_index("date").sort_index()
hold = pd.read_csv(os.path.join(AST, "out", "artaastro_holdout.csv"), parse_dates=["date"])  # topic,id,date,target
TOPICS = ["material","conflict","verbal_conf","cooperation","material_coop","violence"]
tr = df[(df.index >= "2015-01-01") & (df.index < "2020-01-01")]
cal = {}
for t in TOPICS:
    x = tr["intensity"].to_numpy(float); y = tr[t].to_numpy(float); ok = np.isfinite(x) & np.isfinite(y)
    cal[t] = np.polyfit(x[ok], y[ok], 1)
inten = df["intensity"]
b1 = hold["topic"].map(lambda t: cal[t][0]); b0 = hold["topic"].map(lambda t: cal[t][1])
hold["pred"] = (b0 + b1 * hold["date"].map(lambda d: float(inten.get(pd.Timestamp(d), np.nan)))).round(6)
out = os.path.join(AST, "out", "artaastro_intensity_submission.csv")
hold[["topic","id","pred"]].rename(columns={"topic":"trend","pred":"target"}).to_csv(out, index=False)
print(f"wrote {out} ({len(hold):,} rows)")
