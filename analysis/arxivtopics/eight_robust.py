#!/usr/bin/env python3
"""IS THE 8-PARAMETER WIN REAL, OR DID I PICK IT OFF THE TEST SET?

eight_param.py found that deleting PLUTO — the slowest body, the one the staged ablation unlocked
first — lifts the model from +0.7990 (9 params) to +0.8193 (8 params) at the 1996 wall. That wall is
the headline metric. Choosing the variant that scores best on it is selection on the test set, and the
number means nothing until the choice is made WITHOUT looking at it.

So: refit every leave-one-out variant at FIVE origins spread across the record, and select on the four
EARLIER walls only. If the same body is chosen there, the 1996 score is an out-of-sample confirmation
rather than a self-fulfilling one. If a different body wins early, the 1996 result was luck and must be
reported as such.

  python3 analysis/arxivtopics/eight_robust.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import eight_param as EP

WALLS = [n - 60, n - 52, n - 45, n - 37, n - 30]      # 1966 1974 1981 1989 1995(headline)

def at(wall, variant):
    """Re-point the module's wall-dependent state, then reuse its solver verbatim."""
    EP.WALL = wall; EP.HZ = min(wall + HORIZON, EP.ne)
    tv = train_mask(wall).astype(float); wy = np.clip(N[:wall], 0, None) ** 0.75
    w = tv * wy[None]; w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
    EP.w, EP.ysq = w, np.sqrt(Y[:, :wall])
    Wa = np.zeros_like(tv); Wa[:, wall-EP.AK:] = (tv*wy[None])[:, wall-EP.AK:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv*wy[None])[bad]; Wa /= np.maximum(Wa.sum(1,keepdims=True),1e-9)
    EP.MJ = np.maximum((EP.ysq * Wa).sum(1), 1e-3)
    return EP.run(variant)

if __name__ == "__main__":
    VAR = ["all7"] + [f"drop_{b}" for b in EP.BODS]
    print(f"═══ LEAVE-ONE-BODY-OUT AT FIVE ORIGINS · selection uses the four EARLY walls only ═══", flush=True)
    print(f"    {'variant':16s} " + " ".join(f"{YEARS[w]:>8d}" for w in WALLS) + "   early-mean   all-mean", flush=True)
    tab = {}
    for v in VAR:
        row = [at(w, v)[0] for w in WALLS]
        tab[v] = row
        print(f"    {v:16s} " + " ".join(f"{a:+8.4f}" for a in row) +
              f"   {np.mean(row[:-1]):+9.4f}  {np.mean(row):+9.4f}", flush=True)
    early = {v: float(np.mean(tab[v][:-1])) for v in VAR}
    pick = max(early, key=early.get)
    print(f"\n  CHOSEN ON THE EARLY WALLS ALONE: {pick}  (early-mean {early[pick]:+.4f})", flush=True)
    print(f"  its 1996 score, now genuinely out of sample: {tab[pick][-1]:+.4f}"
          f"   vs 9-param {tab['all7'][-1]:+.4f}  → {tab[pick][-1]-tab['all7'][-1]:+.4f}", flush=True)
    print(f"  {'CONFIRMED — the early walls pick the same body' if pick=='drop_pluto' else 'NOT CONFIRMED — the 1996 win was selection noise'}", flush=True)
    json.dump({"walls": [int(YEARS[w]) for w in WALLS], "auc": tab, "early_pick": pick},
              open("analysis/arxivtopics/eight_robust.json", "w"), indent=1)
    print("ROBUSTDONE", flush=True)
