#!/usr/bin/env python3
"""ONE PHASE PER TOPIC, FOUND BY GRID SEARCH (operator 2026-07-28).

On the √ scale the receiver is LINEAR in disguise:
    C = b + Σᵢ aᵢcos(θᵢ − pᵢ) = b + Σᵢ [uᵢcosθᵢ + vᵢsinθᵢ]        (phase = atan2(vᵢ,uᵢ))
so with ONE phase per topic, fixing φ leaves a closed-form weighted least squares. Sweep φ over a 1°
grid, solve exactly at each, keep the best. No gradient descent, no seeds, no local minima — the fit is
the global optimum of its own objective, and it is reproducible to the last bit. That is the property
that matters for unseen topics, where seed variance is what destroyed every learned alternative.

Three models, fitted identically (same weights, same window, same solver) so the comparison isolates
exactly one thing — how many phases a topic is allowed:
  MULTI   b + Σᵢ(uᵢcosθᵢ + vᵢsinθᵢ)                     15 params, one linear solve, phases free
  ONE     b + Σᵢ aᵢcos(θᵢ − Pᵢ − φ)                      9 params, φ by grid, aᵢ signed by LS
  GEARED  b + Σᵢ aᵢcos(θᵢ − Pᵢ − kᵢφ)                    9 per topic + 7 GLOBAL gears kᵢ
          — one topic number still, but each body responds to it at its own rate, so a single phase
            generates a genuinely different pattern across bodies instead of a rigid rotation.

  python3 analysis/arxivtopics/one_phase_grid.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *

BI = [BODIES_ALL.index(b) for b in CHAMPION_BODIES]
NB = len(BI)
WALL = WALL_OUTER
GRID = np.deg2rad(np.arange(0, 360, 1.0))

def wls(X, y, w):
    """Weighted least squares, ridge-stabilised. X (T,k), y (T,), w (T,)."""
    Xw = X * w[:, None]
    A = X.T @ Xw + 1e-8 * np.eye(X.shape[1])
    return np.linalg.solve(A, Xw.T @ y)

def prep(wall):
    tv = train_mask(wall).astype(float)
    w = tv * (np.clip(N[:wall], 0, None) ** 0.75)[None]
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
    return np.sqrt(Y[:, :wall]), w

def evaluate_pred(P):
    tvw = TV[:, :WALL].astype(float)
    mu = (Y[:, :WALL] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    hi = min(WALL + HORIZON, n)
    yt, yp = Y[:, WALL:hi], P[:, WALL:hi]
    curve = [1 - ((yt[:, h] - yp[:, h]) ** 2).sum() / max(((yt[:, h] - mu) ** 2).sum(), 1e-9)
             for h in range(hi - WALL)]
    sk = 1 - ((yt - yp) ** 2).sum(1) / np.maximum(((yt - mu[:, None]) ** 2).sum(1), 1e-9)
    return float(np.mean(curve)), float(np.median(sk)), float((sk > 0).mean() * 100)

TH = TH_ALL[:, BI]                      # (ne, NB) full 1700..2055 sky
ne = TH.shape[0]
Ysq, W = prep(WALL)

def fit_multi():
    """Free phases: one linear solve per topic, 15 regressors."""
    X = np.concatenate([np.ones((ne, 1)), np.cos(TH), np.sin(TH)], 1)
    P = np.zeros((Tn, ne)); ph = np.zeros(Tn)
    for j in range(Tn):
        c = wls(X[:WALL], Ysq[j], W[j])
        P[j] = np.maximum(X @ c, 0) ** 2
        ph[j] = np.rad2deg(np.arctan2(c[1 + NB + 5], c[1 + 5])) % 360      # pluto tuning
    return P, ph

def fit_one(gears=None):
    """ONE phase per topic, found by exhaustive 1° grid search + exact solve at each φ."""
    g = np.ones(NB) if gears is None else gears
    P = np.zeros((Tn, ne)); ph = np.zeros(Tn)
    # precompute the design matrix for every grid angle: cos(θ_i − g_i·φ)
    Xs = np.empty((len(GRID), ne, 1 + NB))
    Xs[:, :, 0] = 1.0
    for a, phi in enumerate(GRID):
        Xs[a, :, 1:] = np.cos(TH - g[None, :] * phi)
    for j in range(Tn):
        best, bc, ba = np.inf, None, 0
        for a in range(len(GRID)):
            Xa = Xs[a, :WALL]
            c = wls(Xa, Ysq[j], W[j])
            r = float((W[j] * (Xa @ c - Ysq[j]) ** 2).sum())
            if r < best: best, bc, ba = r, c, a
        P[j] = np.maximum(Xs[ba] @ bc, 0) ** 2
        ph[j] = np.rad2deg(GRID[ba]) % 360
    return P, ph

if __name__ == "__main__":
    print(f"═══ ONE PHASE (grid) vs MULTI PHASE (free) · wall {YEARS[WALL]} · identical fitting ═══", flush=True)
    out = {}
    Pm, phm = fit_multi()
    out["multi (15 params, free phases)"] = evaluate_pred(Pm)
    Po, pho = fit_one()
    out["one phase (9 params, 1° grid)"] = evaluate_pred(Po)
    # global gears chosen by total training residual — 7 numbers shared by all topics
    best_g, best_r = None, np.inf
    for trial in [np.ones(NB), np.array([1,1,1,1,1,1,1.]), np.array([1,1,1,2,2,3,1.]),
                  np.array([1,2,3,1,1,1,1.]), np.arange(1, NB + 1).astype(float),
                  np.array([1,1,2,2,3,3,1.])]:
        Xs = np.stack([np.concatenate([np.ones((ne,1)), np.cos(TH - trial[None,:]*p)],1) for p in GRID])
        tot = 0.0
        for j in range(0, Tn, 5):                      # every 5th topic — a cheap, unbiased probe
            r = min(float((W[j]*(Xs[a,:WALL] @ wls(Xs[a,:WALL], Ysq[j], W[j]) - Ysq[j])**2).sum())
                    for a in range(0, len(GRID), 4))
            tot += r
        if tot < best_r: best_r, best_g = tot, trial
    print(f"  global gears selected on TRAIN residual: {best_g}", flush=True)
    Pg, phg = fit_one(best_g)
    out["one phase + global gears"] = evaluate_pred(Pg)
    for k, (a, s, p) in out.items():
        print(f"  {k:34s} AUC {a:+.4f} · median skill {s:+.4f} · {p:.1f}%>0", flush=True)
    gap = out["multi (15 params, free phases)"][0] - out["one phase (9 params, 1° grid)"][0]
    gapg = out["multi (15 params, free phases)"][0] - out["one phase + global gears"][0]
    print(f"\n  gap multi − one          {gap:+.4f}", flush=True)
    print(f"  gap multi − one+gears    {gapg:+.4f}   (gradient-fit single-phase was −0.0756)", flush=True)
    print(f"  DETERMINISTIC: no seeds, no gradient descent — refitting gives identical bits.", flush=True)
    json.dump({k: {"auc": v[0], "skill": v[1], "pct": v[2]} for k, v in out.items()} |
              {"gears": best_g.tolist(), "gap": gap, "gap_geared": gapg},
              open("analysis/arxivtopics/one_phase_grid.json", "w"), indent=1)
    print("GRIDDONE", flush=True)
