#!/usr/bin/env python3
"""WEEKLY collection + fit for the topic-selected fields — the resolution the analysis now runs at (daily purged).

Per keyword (walking analysis/_fields.json most-common-first): fetch the monthly backbone (1 request) + stitch
the overlapping weekly chunks onto it (weekly_fit.stitch, ~5 requests), then fit the RAW sidereal model at WEEKLY
resolution. Far cheaper than daily (~6 requests/field vs ~50) and plenty for the cycles we fit. The fit's SIGN is
the field's house (house = sign). Result → analysis/_fields_weekly.json, checkpointed per keyword.

  python3 analysis/collect_weekly.py            # whole work-list, most-common first
  python3 analysis/collect_weekly.py --limit 20 # next 20 only
"""
import importlib.util as u, os, sys, json
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

def _load(p):
    s = u.spec_from_file_location(p.split("/")[-1][:-3], p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

tf = _load("analysis/trends_fit.py")          # MONTHLY model (11 bodies, Moon dropped) — no weekly stitch
sw = _load("analysis/_stopwords.py")

# Two parallel TRACKS share this collector, selected by env (defaults = the Professions track):
#   Professions: AQ_POOL=_fields.json  AQ_REG=_fields_weekly.json   (the ISCO occupations → 12 houses)
#   Topics:      AQ_POOL=_topics.json  AQ_REG=_topics_weekly.json   (the learning topics → 12 houses)
FIELDS = os.environ.get("AQ_POOL", "analysis/_fields.json")
REG = os.environ.get("AQ_REG", "analysis/_fields_weekly.json")
MDIR = os.environ.get("AQ_MDIR", "analysis/data_monthly")
os.makedirs(MDIR, exist_ok=True)
RANGE = "2004-01-01 2026-06-24"
POP_MIN = 0.0               # NO selection — every occupation/topic is analysed (popularity is recorded, not gated)

LON = tf.ephemeris()       # MONTHLY ephemeris; fit_topic sweeps the per-field FREQUENCY itself (14-param model) → takes LON


def load(p): return json.load(open(p)) if os.path.exists(p) else {}
def save(p, o): json.dump(o, open(p, "w"), indent=0)


def collect(word, s):
    sl = tf.slug(word)
    mp = f"{MDIR}/{sl}.csv"
    if not os.path.exists(mp):
        m = tf.fetch_single(word, s, RANGE)            # SINGLE-field MONTHLY series (own max=100) — the fit data, full precision
        if m is None or len(m) < 24: return None, "no-monthly"
        m.to_csv(mp, index=False)
    # POPULARITY (separate, monthly): one anchored [query, arta, quest] comparison → the field's mean level as % of 'quest'.
    pop_df = tf.fetch_range(word, s, RANGE)
    pv = pd.to_numeric(pop_df["v"], errors="coerce").dropna() if pop_df is not None else None
    popularity = float(pv.mean()) if pv is not None and len(pv) else 0.0
    if popularity < POP_MIN:
        return None, f"low-pop {popularity:.1f}%"
    anchor = dict(tf.LAST_ANCHOR.get(word, {}))        # {arta, quest, scale, saturated} from the popularity comparison
    M = pd.read_csv(mp); M["Time"] = pd.to_datetime(M["Time"])  # MONTHLY fit — align the backbone to the monthly GRID
    M = M.drop_duplicates("Time").set_index("Time")["v"].reindex(tf.GRID)
    y = tf.clean_y(pd.DataFrame({"v": M.values}))
    if y is None: return None, "unclean"
    yv = y[:-tf.DROP_LAST]                              # fit on the history minus the recent 12 months (recency bias)
    rec = tf.fit_topic(yv, LON)
    rec["popularity"] = round(popularity, 1)           # GLOBAL: mean monthly interest as % of 'quest'. Set BEFORE rep.
    rec["shares"] = tf.body_shares(rec, LON, yv)
    rec["rep"] = tf.rep_score(rec)
    rec["anchor"] = anchor
    return rec, "ok"


def main():
    work = load(FIELDS)
    reg = load(REG)
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 10 ** 9
    s = tf.make_session()
    if not os.path.exists("analysis/_anchor_calib.json"):
        tf.calibrate(s); print(f"[collect_weekly] calibrated quest/arta = {tf.QUEST_PER_ARTA}", flush=True)
    todo = [(w, meta) for w, meta in work.items() if w not in reg]   # NO selection — every ISCO-08 occupation is analysed
    print(f"[collect_weekly] {len(work)} fields · {len(reg)} done · {len(todo)} to collect (most-common first) · {len(s.cookies)} cookies", flush=True)
    done = 0
    for w, meta in todo:
        if done >= limit: break
        query = meta.get("query", w)                       # Trends search term (Topics: a clean phrase; key stays a slug)
        axis = meta.get("axis", "house")                   # 'house' = Professions track · 'topic' = Topics track
        try:
            rec, why = collect(query, s)
        except Exception as e:
            print(f"  ! {w:20s} ERROR {e}", flush=True); continue
        label = meta.get("label", w)                       # ISCO occupation / topic title; falls back to the key
        if rec is None:
            reg[w] = {"key": w, "label": label, "skip": why, "isco": meta.get("isco", ""),
                      "axis": axis, "system": meta.get("system", ""), "res": "skip"}
            print(f"  - {w:30s} {why}", flush=True)
        else:
            rec.update(key=w, label=label, res="weekly", pos="noun", freq=0, topics=[],
                       isco=meta.get("isco", ""), axis=axis, system=meta.get("system", ""))
            reg[w] = rec
            print(f"  + {w:30.30s} raw R² {rec['r2']*100:5.1f}  {rec['sign']:11s}  pop {rec['popularity']}%", flush=True)
        save(REG, reg); done += 1
    ok = sum(1 for v in reg.values() if v.get("res") == "weekly")
    print(f"[collect_weekly] {done} processed this run · {ok} fields fitted total → {REG}", flush=True)


if __name__ == "__main__":
    main()
