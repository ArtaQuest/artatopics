#!/usr/bin/env python3
"""One-shot sidereal fit for a SINGLE candidate topic (the NO-POOL, fill-missing-houses-one-by-one workflow).

  python3 analysis/analyze_one.py "creative writing"            # fit + report sign/rep (writes nothing permanent)
  python3 analysis/analyze_one.py "creative writing" --finalize # also persist into the Topics registry + pool

Reports the fitted house (sign), rep (= runner-up ratio), R² and popularity so the operator can decide whether the
candidate decisively fills a missing house (rep > 1.2). --finalize keeps it; otherwise nothing is added.
"""
import os, sys, json
os.environ.setdefault("AQ_POOL", "analysis/_topics.json")
os.environ.setdefault("AQ_REG", "analysis/_topics_weekly.json")
os.environ.setdefault("AQ_MDIR", "analysis/data_monthly_topics")
os.environ.setdefault("AQ_WDIR", "analysis/data_weekly_topics")
import importlib.util as u, re
def L(p):
    s = u.spec_from_file_location(p.split("/")[-1][:-3], p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
cw = L("analysis/collect_weekly.py")

q = sys.argv[1]
final = "--finalize" in sys.argv
slug = re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")
s = cw.tf.make_session()
if not os.path.exists("analysis/_anchor_calib.json"):
    cw.tf.calibrate(s)
rec, why = cw.collect(q, s)
if rec is None:
    print(json.dumps({"ok": False, "query": q, "why": why})); sys.exit(0)
out = {"ok": True, "query": q, "slug": slug, "sign": rec["sign"], "rep": round(cw.tf.rep_score(rec), 3),
       "house_ratio": round(float(rec.get("house_ratio", 0)), 3), "r2": round(float(rec["r2"]), 3),
       "pop": rec.get("popularity")}
print(json.dumps(out))
if final:
    reg = json.load(open(cw.REG)) if os.path.exists(cw.REG) else {}
    rec.update(key=slug, label=q, res="weekly", axis="topic", system="", pos="noun", final=1)
    reg[slug] = rec
    json.dump(reg, open(cw.REG, "w"), indent=0)
    pool = json.load(open(cw.FIELDS))
    pool[slug] = {"axis": "topic", "label": q, "query": q, "final": 1}
    json.dump(pool, open(cw.FIELDS, "w"), indent=0)
    print(f"[finalized] {slug} → {rec['sign']} (rep {out['rep']})")
