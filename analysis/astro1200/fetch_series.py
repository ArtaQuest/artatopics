#!/usr/bin/env python3
"""Astro five-property — fetch each SELECTED term's monthly interest series ON ITS OWN PROPERTY.

For property <prop>, walks the selected terms of <prop>_chart_terms.json and fetches the single-frame
all-time monthly Trends series with the MATCHING gprop (web terms get web interest, youtube terms get
YouTube search interest, ...). Own-max=100 (full shape precision), cached one CSV per term under
analysis/astro1200/series_<prop>/. Rides 429s; transient failures retry on the next pass. Resumable.

  python3 analysis/astro1200/fetch_series.py <web|images|news|shop|youtube> [--limit N]
"""
import importlib.util as u, json, os, sys, time
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

tf = _load("analysis/trends_fit.py", "tf")
HERE = "analysis/astro1200"
PROPS = {"web": "", "images": "images", "news": "news", "shop": "froogle", "youtube": "youtube"}
PROP = sys.argv[1]
API = PROPS[PROP]
OUT = f"{HERE}/series_{PROP}"
os.makedirs(OUT, exist_ok=True)
RANGE = tf.ALL_TIME                                    # 2004-01..2026-06 (non-web data begins 2008)


def fetch_one(query, sess):
    cache = f"{OUT}/{tf.slug(query)}.csv"
    if os.path.exists(cache):
        return "cached"
    td = tf._fetch_timeline({"comparisonItem": [{"keyword": query, "geo": "", "time": RANGE}],
                             "category": 0, "property": API}, sess)
    if td is None:
        return "retry-later"
    v = np.array([float(d["value"][0]) for d in td])
    df = pd.DataFrame({"Time": pd.to_datetime([int(d["time"]) for d in td], unit="s"), "v": v})
    df.to_csv(cache, index=False)
    return f"ok ({len(df)} pts)"


def main():
    terms = json.load(open(f"{HERE}/{PROP}_chart_terms.json"))
    sel = sorted([t for t, r in terms.items() if r.get("selected")],
                 key=lambda t: -terms[t]["n_lunations"])
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 10 ** 9
    sess = tf.make_session()
    todo = [t for t in sel if not os.path.exists(f"{OUT}/{tf.slug(t)}.csv")][:limit]
    print(f"[series {PROP}] selected {len(sel)} · to fetch {len(todo)}", flush=True)
    for t in todo:
        print(f"  {t:30.30s} {fetch_one(t, sess)}", flush=True)


if __name__ == "__main__":
    main()
