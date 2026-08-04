#!/usr/bin/env python3
"""adstopics — fetch each taxonomy topic's worldwide monthly YOUTUBE-search Trends series.
Single-frame (own max=100), cached one CSV per topic under analysis/adstopics/series/. Rides 429s;
transient failures retry next pass. Resumable.
  python3 analysis/adstopics/fetch_series.py [--limit N]
"""
import importlib.util as u, json, os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
tf = _load("analysis/trends_fit.py", "tf")
OUT = "analysis/adstopics/series"
os.makedirs(OUT, exist_ok=True)
RANGE = tf.ALL_TIME

def fetch_one(query, sess):
    cache = f"{OUT}/{tf.slug(query)}.csv"
    if os.path.exists(cache): return "cached"
    td = tf._fetch_timeline({"comparisonItem": [{"keyword": query, "geo": "", "time": RANGE}],
                             "category": 0, "property": "youtube"}, sess)
    if td is None: return "retry-later"
    v = np.array([float(d["value"][0]) for d in td])
    pd.DataFrame({"Time": pd.to_datetime([int(d["time"]) for d in td], unit="s"), "v": v}).to_csv(cache, index=False)
    return f"ok ({len(td)} pts)"

def main():
    vocab = json.load(open("analysis/adstopics/vocabulary.json"))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 10 ** 9
    sess = tf.make_session()
    import random
    keys = sorted(vocab)
    random.Random(42).shuffle(keys)                       # deterministic representative order
    todo = [t for t in keys if not os.path.exists(f"{OUT}/{tf.slug(t)}.csv")][:limit]
    print(f"[adstopics] vocab {len(vocab)} · to fetch {len(todo)}", flush=True)
    for t in todo:
        print(f"  {t:36.36s} {fetch_one(t, sess)}", flush=True)

if __name__ == "__main__":
    main()
