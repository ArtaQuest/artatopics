#!/usr/bin/env python3
"""Offline re-fit of a registry from its cached MONTHLY series (no network) — used after a model change
(frequency floor / Moon drop). Reads AQ_REG (default _fields_weekly.json), AQ_MDIR (default data_monthly).
Preserves popularity/label/axis/system/isco; recomputes the fit (sign, freq, rep, weights). Checkpoints each field."""
import json, os, importlib.util as u
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
s = u.spec_from_file_location("tf", "analysis/trends_fit.py"); tf = u.module_from_spec(s); s.loader.exec_module(tf)
lon = tf.ephemeris()
REG = os.environ.get("AQ_REG", "analysis/_fields_weekly.json")
MDIR = os.environ.get("AQ_MDIR", "analysis/data_monthly")

def series(key, label):
    for nm in (key, tf.slug(label or key)):
        p = f"{MDIR}/{nm}.csv"
        if os.path.exists(p):
            M = pd.read_csv(p); M["Time"] = pd.to_datetime(M["Time"])
            M = M.drop_duplicates("Time").set_index("Time")["v"].reindex(tf.GRID)
            return tf.clean_y(pd.DataFrame({"v": M.values}))
    return None

d = json.load(open(REG)); n = 0
for k, r in d.items():
    if r.get("res") != "weekly": continue
    y = series(k, r.get("label"))
    if y is None: print(f"  ! no series for {k}", flush=True); continue
    yv = y[:-tf.DROP_LAST]
    rec = tf.fit_topic(yv, lon)
    rec["shares"] = tf.body_shares(rec, lon, yv)            # body variance decomposition (the analysis-page chart)
    for keep in ("popularity", "label", "axis", "system", "isco", "key", "res", "pos", "freq", "topics", "final", "anchor"):
        if keep in r: rec[keep] = r[keep]
    rec["rep"] = tf.rep_score(rec)
    d[k] = rec; n += 1
    if n % 10 == 0:
        json.dump(d, open(REG, "w"), indent=0); print(f"  refit {n}…", flush=True)
json.dump(d, open(REG, "w"), indent=0)
print(f"[refit_monthly] {os.path.basename(REG)}: refit {n} fields", flush=True)
