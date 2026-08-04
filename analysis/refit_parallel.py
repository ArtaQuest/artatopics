#!/usr/bin/env python3
"""Parallel offline re-fit of a registry from its cached MONTHLY series (no network), across all cores.
Same per-field work as refit_monthly.py (recompute sign/freq/rep/weights/shares; preserve metadata) but
fanned out over a process pool — the huber least-squares fit is the bottleneck, ~10 s/field single-threaded.
Reads AQ_REG (default _fields_weekly.json), AQ_MDIR (default data_monthly). Writes AQ_OUT (default = AQ_REG)."""
import json, os, sys, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis")
import pandas as pd
import trends_fit as tf

REG = os.environ.get("AQ_REG", "analysis/_fields_weekly.json")
OUT = os.environ.get("AQ_OUT", REG)
MDIR = os.environ.get("AQ_MDIR", "analysis/data_monthly")
KEEP = ("popularity", "label", "axis", "system", "isco", "key", "res", "pos", "freq", "topics", "final", "anchor")
_lon = None


def _init():
    global _lon
    _lon = tf.ephemeris()


def _series(key, label):
    for nm in (key, tf.slug(label or key)):
        p = f"{MDIR}/{nm}.csv"
        if os.path.exists(p):
            M = pd.read_csv(p); M["Time"] = pd.to_datetime(M["Time"])
            M = M.drop_duplicates("Time").set_index("Time")["v"].reindex(tf.GRID)
            return tf.clean_y(pd.DataFrame({"v": M.values}))
    return None


def _work(item):
    k, r = item
    y = _series(k, r.get("label"))
    if y is None:
        return k, None
    yv = y[:-tf.DROP_LAST]
    rec = tf.fit_topic(yv, _lon)
    rec["shares"] = tf.body_shares(rec, _lon, yv)
    for keep in KEEP:
        if keep in r:
            rec[keep] = r[keep]
    rec["rep"] = tf.rep_score(rec)
    return k, rec


if __name__ == "__main__":
    d = json.load(open(REG))
    items = [(k, r) for k, r in d.items() if r.get("res") == "weekly"]
    nproc = max(1, (os.cpu_count() or 4) - 2)
    print(f"[refit_parallel] {len(items)} fields on {nproc} workers…", flush=True)
    done = 0
    with Pool(nproc, initializer=_init) as pool:
        for k, rec in pool.imap_unordered(_work, items, chunksize=3):
            if rec is not None:
                d[k] = rec
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(items)}", flush=True)
    json.dump(d, open(OUT, "w"), indent=0)
    print(f"[refit_parallel] {len(items)} fields done -> {os.path.basename(OUT)}", flush=True)
