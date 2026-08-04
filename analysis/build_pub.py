#!/usr/bin/env python3
"""Assemble the public reproducibility bundle (analysis/_pub/) for the Journal of Seasonality:
   full-resolution MONTHLY series, the monthly sidereal ephemeris, the fitted results, the recency
   results, and two self-contained Colab notebooks. Uploaded to wp-content/uploads/research/ on prod.
     python3 analysis/build_pub.py
"""
import importlib.util as _u, os, json, shutil
import numpy as np, pandas as pd
_spec = _u.spec_from_file_location("tf", os.path.join(os.path.dirname(__file__) or ".", "trends_fit.py"))
tf = _u.module_from_spec(_spec); _spec.loader.exec_module(tf)
# MONTHLY model — tf's own defaults (GRID=MS, DROP_LAST=12, DATA_DIR=data_monthly) are correct.
REG = os.environ.get("AQ_REG", "analysis/_fields_weekly.json")
OUT = "analysis/_pub"; os.makedirs(OUT, exist_ok=True)

def clean(df):
    y = pd.to_numeric(df["v"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return y.interpolate(limit_direction="both").ffill().bfill().to_numpy(float)

def main():
    reg = json.load(open(REG)) if os.path.exists(REG) else tf.load_reg()
    pub = sorted([r for r in reg.values() if r.get("res") == "weekly"], key=lambda r: -r.get("rep", 0.0))
    # 1) full-resolution MONTHLY series (one row per field) + the month grid
    series = {}
    for r in pub:
        p = f"{tf.DATA_DIR}/{tf.slug(r['label'])}.csv"
        if os.path.exists(p): series[r["key"]] = [round(float(v), 2) for v in clean(pd.read_csv(p))]
    json.dump({"months": [str(d.date()) for d in tf.GRID], "series": series,
               "fields": {r["key"]: r["label"] for r in pub}},
              open(f"{OUT}/series.json", "w"), separators=(",", ":"))
    # 2) monthly ephemeris (sidereal longitudes) — so notebooks need no kerykeion
    if os.path.exists(tf.EPHEM): shutil.copy(tf.EPHEM, f"{OUT}/ephemeris_monthly.csv")
    # 3) fitted results + recency results
    shutil.copy("artaquest-web/src/data/research.json", f"{OUT}/atlas.json")
    if os.path.exists("analysis/_recency.json"): shutil.copy("analysis/_recency.json", f"{OUT}/recency.json")
    print(f"  bundle: {len(series)} series, ephemeris, atlas.json, recency.json → {OUT}/")
    for f in os.listdir(OUT):
        if os.path.isfile(OUT + "/" + f): print(f"    {f}: {os.path.getsize(OUT+'/'+f)//1024} KB")

if __name__ == "__main__":
    main()
