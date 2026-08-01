#!/usr/bin/env python3
"""EIGHT PARAMETERS OR FEWER (operator 2026-07-28).

The 9-param model is b + seven amplitudes + one grid-searched phase, at +0.7990. Every result in this
project says CONSTRAINT HELPS out of sample (seven free phases cost −0.0145; free-phase least squares
explodes to −27). So the question is not "what can we afford to lose" but "which parameter is costing
us accuracy". Ways to spend one fewer, each tested at the same wall with the same deterministic
grid-search + closed-form anchored solve:

  DROP A BODY (8)      — leave-one-out over all seven. Mars contributed exactly +0.0000 in the staged
                          phase ablation; it may be pure variance.
  FIX THE LEVEL (8)    — stop fitting b and set it to the anchor level m the model is already told.
                          The intercept is the one parameter with a known correct value.
  TIE AMPLITUDES (8)   — make the two slowest bodies (or the three fastest) share one amplitude: they
                          are nearly collinear over a 300-year window anyway.
  GLOBAL SHAPE (3)     — a_i = s_j·g_i with g GLOBAL (shared by all topics, fitted once): a topic then
                          has only a level, a scale and a phase. The logical end of the constraint
                          argument, included to find where the curve actually turns.
  PHYSICS PRIOR (3)    — the same, but g fixed a priori by 1/r̄ instead of fitted.
  2nd HARMONIC (8)     — drop mars, spend the slot on cos(2(θ_pluto − φ)) instead: more structure on
                          the body that matters rather than a body that does not.

  python3 analysis/arxivtopics/eight_param.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *

PERIOD = {"pluto":248.,"neptune":164.8,"uranus":84.,"saturn":29.46,"node":18.6,"jupiter":11.86,"mars":1.88}
BODS = sorted(CHAMPION_BODIES, key=lambda b: -PERIOD[b])
BI = [BODIES_ALL.index(b) for b in BODS]; NB = len(BI)
TH = TH_ALL[:, BI]; ne = TH.shape[0]
RBAR = R_ALL[:, BI].mean(0)
GRID = np.deg2rad(np.arange(0, 360, 1.0)); NG = len(GRID)
LAM, AK, WALL = 0.03, 5, WALL_OUTER
HZ = min(WALL + HORIZON, ne)
COS = np.stack([np.cos(TH[:, i][None, :] - GRID[:, None]) for i in range(NB)])       # (NB,NG,ne)
COS2 = np.cos(2 * (TH[:, 0][None, :] - GRID[:, None]))                                # pluto, 2nd harmonic

tv = train_mask(WALL).astype(float); wy = np.clip(N[:WALL], 0, None) ** 0.75
w = tv * wy[None]; w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
ysq = np.sqrt(Y[:, :WALL])
Wa = np.zeros_like(tv); Wa[:, WALL-AK:] = (tv*wy[None])[:, WALL-AK:]
bad = Wa.sum(1) <= 0; Wa[bad] = (tv*wy[None])[bad]; Wa /= np.maximum(Wa.sum(1,keepdims=True),1e-9)
MJ = np.maximum((ysq * Wa).sum(1), 1e-3)

def cols_for(variant, gvec=None):
    """Return a list of (NG, ne) regressor blocks, and whether an intercept column is fitted."""
    if variant == "all7":      return [COS[i] for i in range(NB)], True
    if variant.startswith("drop_"):
        k = BODS.index(variant[5:]); return [COS[i] for i in range(NB) if i != k], True
    if variant == "fixed_b":   return [COS[i] for i in range(NB)], False
    if variant == "tie_slow":  return [COS[0] + COS[1]] + [COS[i] for i in range(2, NB)], True
    if variant == "tie_fast":  return [COS[i] for i in range(NB-3)] + [COS[NB-3]+COS[NB-2]+COS[NB-1]], True
    if variant == "harm2":     return [COS[i] for i in range(NB-1)] + [COS2], True
    if variant == "global_shape": return [sum(gvec[i]*COS[i] for i in range(NB))], True
    if variant == "physics":   return [sum((1.0/RBAR[i])*COS[i] for i in range(NB))], True
    raise ValueError(variant)

def run(variant, gvec=None):
    blocks, fit_b = cols_for(variant, gvec)
    K = len(blocks) + (1 if fit_b else 0)
    P = np.zeros((Tn, ne))
    for j in range(Tn):
        X = np.empty((NG, ne, K)); o = 0
        if fit_b: X[:, :, 0] = 1.0; o = 1
        for q, blk in enumerate(blocks): X[:, :, o+q] = blk
        tgt = ysq[j] if fit_b else ysq[j] - MJ[j]
        atg = MJ[j] if fit_b else 0.0
        Xt, Xa = X[:, :WALL], X[:, WALL:HZ]
        aw = LAM/(MJ[j]**2)/max(HZ-WALL, 1)
        A = np.einsum('gtp,t,gtq->gpq', Xt, w[j], Xt) + aw*np.einsum('gtp,gtq->gpq', Xa, Xa) + 1e-8*np.eye(K)[None]
        b = np.einsum('gtp,t->gp', Xt, w[j]*tgt) + aw*np.einsum('gtp->gp', Xa)*atg
        c = np.linalg.solve(A, b[..., None])[..., 0]
        r = (np.einsum('gtp,gp->gt', Xt, c) - tgt[None])**2 @ w[j] + aw*((np.einsum('gtp,gp->gt', Xa, c)-atg)**2).sum(1)
        g = int(np.argmin(r))
        base = 0.0 if fit_b else MJ[j]
        P[j] = np.maximum(X[g] @ c[g] + base, 0)**2
    tvw = TV[:, :WALL].astype(float)
    mu = (Y[:, :WALL]*tvw).sum(1)/np.maximum(tvw.sum(1), 1.0)
    hi = min(WALL+HORIZON, n); yt, yp = Y[:, WALL:hi], P[:, WALL:hi]
    auc = float(np.mean([1-((yt[:,h]-yp[:,h])**2).sum()/max(((yt[:,h]-mu)**2).sum(),1e-9) for h in range(hi-WALL)]))
    sk = 1-((yt-yp)**2).sum(1)/np.maximum(((yt-mu[:,None])**2).sum(1),1e-9)
    return auc, float(np.median(sk)), float((sk>0).mean()*100), K+1     # +1 for the phase

if __name__ == "__main__":
    print(f"═══ FEWER PARAMETERS · wall {YEARS[WALL]} · deterministic grid + closed-form anchored solve ═══", flush=True)
    res = {}
    for v in ["all7"] + [f"drop_{b}" for b in BODS] + ["fixed_b", "tie_slow", "tie_fast", "harm2", "physics"]:
        a, s, p, K = run(v)
        res[v] = {"auc": a, "skill": s, "pct": p, "params": K}
        print(f"  {v:16s} {K:2d} params · AUC {a:+.4f} · skill {s:+.4f} · {p:.1f}%>0", flush=True)
    # global shape: fit g once on the training fields by averaging each topic's normalised amplitudes
    a9 = res["all7"]["auc"]
    best8 = max([v for v in res if res[v]["params"] <= 8], key=lambda v: res[v]["auc"])
    print(f"\n  9-param reference   {a9:+.4f}", flush=True)
    print(f"  BEST ≤8 params      {res[best8]['auc']:+.4f}  ({best8}, {res[best8]['params']} params) "
          f"→ {res[best8]['auc']-a9:+.4f} vs the 9-param model", flush=True)
    print(f"  deployed multi-phase +0.8193 · persistence +0.7344", flush=True)
    json.dump(res, open("analysis/arxivtopics/eight_param.json", "w"), indent=1)
    print("EIGHTDONE", flush=True)
