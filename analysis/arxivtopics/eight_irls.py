#!/usr/bin/env python3
"""FIXING THE ESTIMATOR, NOT THE PARAMETER COUNT (operator 2026-07-28: improve AUC at ≤8 params).

Neither deleting a body nor constraining the spectrum improves anything: at 8 parameters the best
honest result is a TIE with 9 (+0.0001). That says the parameter count was never what was costing us
accuracy — the SOLVER was. The closed-form fit carries two mis-specifications the deployed gradient
model does not have:

  L1 vs L2         the deployed loss is |√ŷ − √y| (absolute). The closed form minimises the SQUARE.
                   These are different models, not different implementations of one: squared error lets
                   a handful of spike years in a field's history drag its tuning away from the shape
                   the other fifty years agree on. Citation series are spiky by nature.
  RECTIFICATION    the receiver is max(C,0)² — below zero the diode simply does not conduct. The closed
                   form fits C to √y as if it were linear everywhere, so years the model correctly
                   predicts as silent still pull on the fit as though they were errors.

Both are repaired by the same device: iteratively reweighted least squares. Reweighting by 1/|r|
minimises absolute error, and zeroing the weight where C < 0 tells the fit what the diode already
knows. The phase grid is re-searched at every iteration, so the tuning is re-chosen under the corrected
weights rather than inherited from the wrong ones. It stays deterministic — no seeds, no restarts.

The Huber floor δ is each field's own median absolute residual at the first iteration: data-selected,
per field, not a knob.

  python3 analysis/arxivtopics/eight_irls.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import eight_param as EP
from eight_struct import prep, auc_of, WALLS

BODS, NB, COS, GRID, NG, ne = EP.BODS, EP.NB, EP.COS, EP.GRID, EP.NG, EP.ne
ITERS = 5


def fit(wall, drop=None, l1=True, rect=True, iters=ITERS):
    w, ysq, MJ, hz = prep(wall)
    keep = [i for i in range(NB) if drop is None or i != BODS.index(drop)]
    K = 1 + len(keep)
    X = np.empty((NG, ne, K)); X[:, :, 0] = 1.0
    for q, i in enumerate(keep): X[:, :, 1+q] = COS[i]
    Xt, Xa = X[:, :wall], X[:, wall:hz]
    XaS = np.einsum('gtp,gtq->gpq', Xa, Xa); XaB = np.einsum('gtp->gp', Xa)
    I = 1e-8*np.eye(K)[None]
    P = np.zeros((Tn, ne))
    for j in range(Tn):
        om = w[j].copy(); aw = EP.LAM/(MJ[j]**2)/max(hz-wall, 1); delta = None
        for it in range(iters):
            A = np.einsum('gtp,t,gtq->gpq', Xt, om, Xt) + aw*XaS + I
            b = np.einsum('gtp,t->gp', Xt, om*ysq[j]) + aw*XaB*MJ[j]
            c = np.linalg.solve(A, b[..., None])[..., 0]
            fitv = np.einsum('gtp,gp->gt', Xt, c)
            res = (np.maximum(fitv, 0) if rect else fitv) - ysq[j][None]
            r = (res**2) @ om + aw*((np.einsum('gtp,gp->gt', Xa, c)-MJ[j])**2).sum(1)
            g = int(np.argmin(r))
            if it == iters-1: break
            rr = np.abs(res[g])
            if delta is None:                      # data-selected scale: this field's own median |r|
                delta = max(float(np.median(rr[om > 0])) if (om > 0).any() else 1e-3, 1e-6)
            om = w[j].copy()
            if l1:   om = om / np.maximum(rr, delta)          # 1/|r| ⇒ absolute error
            if rect: om = om * (fitv[g] > 0)                  # the diode does not conduct below zero
            s = om.sum()
            if not np.isfinite(s) or s <= 0: om = w[j].copy(); break
            om *= w[j].sum()/s                                # keep the anchor's relative strength fixed
        P[j] = np.maximum(X[g] @ c[g], 0)**2
    return auc_of(P, wall)


if __name__ == "__main__":
    print("═══ IRLS: ABSOLUTE ERROR + RECTIFICATION-AWARE · selected on EARLY walls ═══", flush=True)
    print(f"    {'estimator':22s}{'params':>7s} " + " ".join(f"{YEARS[w]:>8d}" for w in WALLS) + "   early-mean", flush=True)
    ent = [("L2 plain (9)",        9, dict(drop=None,      l1=False, rect=False)),
           ("L2 plain (8)",        8, dict(drop="jupiter", l1=False, rect=False)),
           ("rect only (8)",       8, dict(drop="jupiter", l1=False, rect=True)),
           ("L1 only (8)",         8, dict(drop="jupiter", l1=True,  rect=False)),
           ("L1 + rect (8)",       8, dict(drop="jupiter", l1=True,  rect=True)),
           ("L1 + rect (9)",       9, dict(drop=None,      l1=True,  rect=True))]
    tab, npar = {}, {}
    for name, K, kw in ent:
        row = [fit(w, **kw) for w in WALLS]
        tab[name], npar[name] = row, K
        print(f"    {name:22s}{K:>7d} " + " ".join(f"{a:+8.4f}" for a in row) + f"   {np.mean(row[:-1]):+9.4f}", flush=True)
    early8 = {v: float(np.mean(tab[v][:-1])) for v in tab if npar[v] <= 8}
    pick = max(early8, key=early8.get)
    print(f"\n  CHOSEN ON EARLY WALLS (≤8 params): {pick}  early-mean {early8[pick]:+.4f}", flush=True)
    print(f"  its 1996 score, out of sample: {tab[pick][-1]:+.4f}", flush=True)
    print(f"  vs 9-param L2 closed form  +0.7990 → {tab[pick][-1]-0.7990:+.4f}", flush=True)
    print(f"  vs deployed 15-param torch +0.8193 → {tab[pick][-1]-0.8193:+.4f}", flush=True)
    print(f"  vs carry-forward persistence +0.7344 → {tab[pick][-1]-0.7344:+.4f}", flush=True)
    json.dump({"walls": [int(YEARS[w]) for w in WALLS], "auc": tab, "params": npar, "early_pick": pick},
              open("analysis/arxivtopics/eight_irls.json", "w"), indent=1)
    print("IRLSDONE", flush=True)
