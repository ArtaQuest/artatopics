#!/usr/bin/env python3
"""Astro four-property — assemble the four study datasets from the fetched series.

Per property (images/news/shop/youtube): one long-format CSV over the clean window
2008-01 .. 2025-06 (non-web Trends data begins 2008; the provisional most-recent 12 months are
dropped per the platform's recency guard):

    topic, class, n_lunations, avg_score, month, value, sun..node (11 sidereal body phases, deg)

`class` is the FROZEN pre-registered argmax-season label (score-weighted, from the lunation charts —
committed before any series was fetched). `value` = the term's own-max=100 monthly interest on ITS
OWN property. Topics whose fetched series is unusable (too sparse) are listed in the manifest.

  python3 analysis/astro1200/build_datasets.py
→ analysis/astro1200/<Prop>Topics.csv + datasets_manifest.json
"""
import importlib.util as u, json, os
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

tf = _load("analysis/trends_fit.py", "tf")
HERE = "analysis/astro1200"
PROPS = {"images": "ImageTopics", "news": "NewsTopics", "shop": "ShopTopics", "youtube": "YouTubeTopics"}
START = pd.Timestamp("2008-01-01")

def main():
    lon = tf.ephemeris()                                   # on tf.GRID (2004-01..2026-06)
    grid = pd.DatetimeIndex(tf.GRID)
    i0 = int(np.where(grid == START)[0][0])                # 2008-01 offset into the platform grid
    n_clean_end = len(grid) - tf.DROP_LAST                 # exclusive end: 2025-06
    months = [d.strftime("%Y-%m") for d in grid[i0:n_clean_end]]
    X = np.column_stack([np.asarray(lon[b], float)[i0:n_clean_end] for b in tf.BODIES])
    n = len(months)

    manifest = {"window": [months[0], months[-1]], "months": n, "bodies": list(tf.BODIES),
                "sidereal_mode": tf.SIDEREAL_MODE, "datasets": {}}
    for prop, name in PROPS.items():
        terms = json.load(open(f"{HERE}/{prop}_chart_terms.json"))
        sel = sorted(((t, r) for t, r in terms.items() if r.get("selected")),
                     key=lambda kv: -kv[1]["avg_score"])
        rows, skipped = [], []
        for t, r in sel:
            p = f"{HERE}/series_{prop}/{tf.slug(t)}.csv"
            if not os.path.exists(p):
                skipped.append([t, "missing"]); continue
            df = pd.read_csv(p); df["Time"] = pd.to_datetime(df["Time"])
            ser = df.drop_duplicates("Time").set_index("Time")["v"].reindex(grid[i0:n_clean_end])
            v = pd.to_numeric(ser, errors="coerce")
            if v.notna().sum() < n * 0.5:
                skipped.append([t, "sparse"]); continue
            y = v.interpolate(limit_direction="both").to_numpy(float)
            if not np.isfinite(y).all() or float(np.nanmax(y)) <= 0:
                skipped.append([t, "flat"]); continue
            for i in range(n):
                rows.append((t, r["class"], r["n_lunations"], r["avg_score"], months[i],
                             round(float(y[i]), 2), *[int(round(X[i, j])) % 360 for j in range(len(tf.BODIES))]))
        out = pd.DataFrame(rows, columns=["topic", "class", "n_lunations", "avg_score", "month", "value"]
                           + list(tf.BODIES))
        path = f"{HERE}/{name}.csv"
        out.to_csv(path, index=False)
        manifest["datasets"][name] = {"topics": int(out["topic"].nunique()), "rows": len(out),
                                      "skipped": skipped}
        print(f"{name}: {out['topic'].nunique()} topics · {len(out)} rows · skipped {len(skipped)}")
    json.dump(manifest, open(f"{HERE}/datasets_manifest.json", "w"), indent=1)

if __name__ == "__main__":
    main()
