#!/usr/bin/env python3
"""Astro-1200 — measure the anchored POPULARITY of every candidate term (selection stage).

Walks the merged candidate pool (analysis/astro1200/candidates.json) and measures each term's mean
worldwide monthly interest as % of the platform's 'quest' anchor, THREE candidates per Trends request
([q1,q2,q3,arta,quest] — 5 comparison slots), writing the standard fetch_range caches so nothing is
ever fetched twice. Resumable; results checkpointed to analysis/astro1200/popularity.json.

The TOP 1200 by this measured popularity become the study set — selection is by measured volume, not
by the enumerators' ranking. No time-series is examined at this stage (only the scalar mean).

  python3 analysis/astro1200/measure_pop.py --limit 90
"""
import importlib.util as u, json, os, sys
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

tf = _load("analysis/trends_fit.py", "tf")
HERE = "analysis/astro1200"
CAND = f"{HERE}/candidates.json"
OUT = f"{HERE}/popularity.json"
RANGE = tf.ALL_TIME

def load(p): return json.load(open(p)) if os.path.exists(p) else {}
def save(p, o):
    tmp = p + ".tmp"; json.dump(o, open(tmp, "w"), indent=0); os.replace(tmp, p)

def pop_batch(queries, s):
    need = [q for q in queries
            if not os.path.exists(f"{tf.CHUNK_DIR}/{tf.slug(q)}__vs-aq__{RANGE.replace(' ', '_')}.csv")]
    if need:
        req = {"comparisonItem": [{"keyword": q, "geo": "", "time": RANGE} for q in need]
               + [{"keyword": tf.ANCHOR_A, "geo": "", "time": RANGE},
                  {"keyword": tf.ANCHOR_B, "geo": "", "time": RANGE}], "category": 0, "property": ""}
        td = tf._fetch_timeline(req, s)
        if td is None:
            return None                                        # transient — retry next pass
        times = pd.to_datetime([int(d["time"]) for d in td], unit="s")
        vals = np.array([[float(v) for v in d["value"]] for d in td], float)
        am = float(vals[:, len(need)].mean()); bm = float(vals[:, len(need) + 1].mean())
        scale = bm if bm >= 1.0 else (am * tf.QUEST_PER_ARTA if am >= 0.5 else 1.0)
        saturated = bm < 1.0 and am < 0.5
        for j, q in enumerate(need):
            v = vals[:, j] / scale * 100.0
            tf.LAST_ANCHOR[q] = {"arta": round(am, 3), "quest": round(bm, 3),
                                 "scale": round(scale, 3), "saturated": bool(saturated)}
            pd.DataFrame({"Time": times, "v": v}).to_csv(
                f"{tf.CHUNK_DIR}/{tf.slug(q)}__vs-aq__{RANGE.replace(' ', '_')}.csv", index=False)
    out = {}
    for q in queries:
        df = tf.fetch_range(q, None, RANGE)                    # cache read only
        pv = pd.to_numeric(df["v"], errors="coerce").dropna() if df is not None else None
        if pv is not None and len(pv):
            out[q] = {"popularity": round(float(pv.mean()), 2),            # full 2004.. mean (context)
                      "pop_12y": round(float(pv.tail(144).mean()), 2),     # THE SELECTION METRIC: last 12 years
                      "pop_recent": round(float(pv.tail(36).mean()), 2),   # last ~3 years — the relevance gate
                      "anchor": dict(tf.LAST_ANCHOR.get(q, {}))}
        else:
            out[q] = {"popularity": 0.0, "pop_12y": 0.0, "pop_recent": 0.0, "anchor": dict(tf.LAST_ANCHOR.get(q, {}))}
    return out

def main():
    cand = json.load(open(CAND))                               # [term, ...] merged + deduped
    done = load(OUT)
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 10 ** 9
    todo = [t for t in cand if t not in done][:limit]
    s = tf.make_session()
    print(f"[measure_pop] {len(cand)} candidates · {len(done)} measured · this run {len(todo)}", flush=True)
    n = 0
    for lo in range(0, len(todo), 3):
        grp = todo[lo:lo + 3]
        res = pop_batch(grp, s)
        if res is None:
            print(f"  ~ batch failed ({grp[0]}…) — will retry next pass", flush=True); continue
        done.update(res); n += len(grp)
        for q in grp:
            print(f"  + {q:32.32s} pop {done[q]['popularity']}", flush=True)
        save(OUT, done)
    print(f"[measure_pop] measured {len(done)}/{len(cand)}", flush=True)

if __name__ == "__main__":
    main()
