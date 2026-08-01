#!/usr/bin/env python3
"""STAGED PHASE FITTING, SLOWEST BODY FIRST (operator 2026-07-28).

    stage 0  one shared phase φ for all seven bodies      (grid search)
    stage 1  LOCK the others, re-tune PLUTO   (248 y)     (grid search)   ← slowest
    stage 2  lock, re-tune NEPTUNE (164.8 y)
    stage 3  URANUS (84 y) · 4 SATURN (29.5 y) · 5 NODE (18.6 y) · 6 JUPITER (11.9 y) · 7 MARS (1.9 y)

Greedy coordinate descent over the phases, ordered by orbital period. Every stage is a 1° grid search
on ONE angle with all other phases held fixed, and at each candidate angle the amplitudes + anchor are
solved in CLOSED FORM (the receiver is linear in them on the √ scale, and the horizon anchor is a
linear constraint folded in as Tikhonov rows). So the whole procedure is deterministic — no gradient
descent, no seeds, identical bits on every refit — and it produces a REGULARISATION PATH: the ablation
below is the honest gain from each additional phase, which answers how many phases the data supports.

  python3 analysis/arxivtopics/staged_phases.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *

PERIOD = {"pluto": 248.0, "neptune": 164.8, "uranus": 84.0, "saturn": 29.46,
          "node": 18.6, "jupiter": 11.86, "mars": 1.88}
BODS = sorted(CHAMPION_BODIES, key=lambda b: -PERIOD[b])          # slowest → fastest
BI = [BODIES_ALL.index(b) for b in BODS]
NB = len(BI)
TH = TH_ALL[:, BI]; ne = TH.shape[0]
GRID = np.deg2rad(np.arange(0, 360, 1.0)); NG = len(GRID)
LAM, ANCHOR_K = 0.03, 5
WALL = WALL_OUTER
print(f"stage order (slowest first): {' → '.join(BODS)}", flush=True)

tv = train_mask(WALL).astype(float)
w = tv * (np.clip(N[:WALL], 0, None) ** 0.75)[None]
w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
ysq = np.sqrt(Y[:, :WALL])
Wa = np.zeros_like(tv); Wa[:, WALL - ANCHOR_K:] = (tv * (np.clip(N[:WALL], 0, None) ** 0.75)[None])[:, WALL - ANCHOR_K:]
bad = Wa.sum(1) <= 0; Wa[bad] = (tv * (np.clip(N[:WALL], 0, None) ** 0.75)[None])[bad]
Wa /= np.maximum(Wa.sum(1, keepdims=True), 1e-9)
MJ = np.maximum((np.sqrt(Y[:, :WALL]) * Wa).sum(1), 1e-3)
HZ = min(WALL + HORIZON, ne)
COS = np.stack([np.cos(TH[:, i][None, :] - GRID[:, None]) for i in range(NB)])   # (NB, NG, ne)

def solve_grid(j, cols, k):
    """Grid-search body k's phase with the others fixed; exact solve (fit + anchor) at each angle."""
    X = np.empty((NG, ne, NB + 1)); X[:, :, 0] = 1.0
    for i in range(NB):
        X[:, :, 1 + i] = COS[i][cols[i]][None, :] if i != k else COS[i]
    Xt, Xa = X[:, :WALL], X[:, WALL:HZ]
    aw = LAM / (MJ[j] ** 2) / max(HZ - WALL, 1)
    A = np.einsum('gtp,t,gtq->gpq', Xt, w[j], Xt) + aw * np.einsum('gtp,gtq->gpq', Xa, Xa)
    A += 1e-8 * np.eye(NB + 1)[None]
    b = np.einsum('gtp,t->gp', Xt, w[j] * ysq[j]) + aw * np.einsum('gtp->gp', Xa) * MJ[j]
    c = np.linalg.solve(A, b[..., None])[..., 0]
    r = (np.einsum('gtp,gp->gt', Xt, c) - ysq[j][None]) ** 2 @ w[j] \
        + aw * ((np.einsum('gtp,gp->gt', Xa, c) - MJ[j]) ** 2).sum(1)
    g = int(np.argmin(r))
    return g, c[g], X[g]

def sc(P):
    tvw = TV[:, :WALL].astype(float)
    mu = (Y[:, :WALL] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    hi = min(WALL + HORIZON, n); yt, yp = Y[:, WALL:hi], P[:, WALL:hi]
    curve = [1 - ((yt[:, h] - yp[:, h]) ** 2).sum() / max(((yt[:, h] - mu) ** 2).sum(), 1e-9) for h in range(hi - WALL)]
    sk = 1 - ((yt - yp) ** 2).sum(1) / np.maximum(((yt - mu[:, None]) ** 2).sum(1), 1e-9)
    return float(np.mean(curve)), float(np.median(sk)), float((sk > 0).mean() * 100)

if __name__ == "__main__":
    cols = np.zeros((Tn, NB), int)                 # each topic's per-body grid index
    P = np.zeros((Tn, ne))
    # ── stage 0: ONE shared phase ────────────────────────────────────────────────────────────
    for j in range(Tn):
        X = np.empty((NG, ne, NB + 1)); X[:, :, 0] = 1.0
        for i in range(NB): X[:, :, 1 + i] = COS[i]
        Xt, Xa = X[:, :WALL], X[:, WALL:HZ]
        aw = LAM / (MJ[j] ** 2) / max(HZ - WALL, 1)
        A = np.einsum('gtp,t,gtq->gpq', Xt, w[j], Xt) + aw * np.einsum('gtp,gtq->gpq', Xa, Xa) + 1e-8 * np.eye(NB + 1)[None]
        b = np.einsum('gtp,t->gp', Xt, w[j] * ysq[j]) + aw * np.einsum('gtp->gp', Xa) * MJ[j]
        c = np.linalg.solve(A, b[..., None])[..., 0]
        r = (np.einsum('gtp,gp->gt', Xt, c) - ysq[j][None]) ** 2 @ w[j] + aw * ((np.einsum('gtp,gp->gt', Xa, c) - MJ[j]) ** 2).sum(1)
        g = int(np.argmin(r)); cols[j, :] = g
        P[j] = np.maximum(X[g] @ c[g], 0) ** 2
    rows = [("0 · one shared phase", *sc(P))]
    print(f"\n  {rows[0][0]:34s} AUC {rows[0][1]:+.4f} · skill {rows[0][2]:+.4f} · {rows[0][3]:.1f}%>0", flush=True)
    # ── stages 1..7: unlock one body at a time, slowest first ────────────────────────────────
    for k, body in enumerate(BODS):
        for j in range(Tn):
            g, c, Xg = solve_grid(j, cols[j], k)
            cols[j, k] = g
            P[j] = np.maximum(Xg @ c, 0) ** 2
        a, s, p = sc(P)
        prev = rows[-1][1]
        rows.append((f"{k+1} · +{body} ({PERIOD[body]:g} y)", a, s, p))
        print(f"  {rows[-1][0]:34s} AUC {a:+.4f} · skill {s:+.4f} · {p:.1f}%>0   gain {a-prev:+.4f}", flush=True)
    print(f"\n  deployed multi-phase (gradient) +0.8193 · persistence +0.7344", flush=True)
    json.dump({"order": BODS, "stages": [{"stage": r[0], "auc": r[1], "skill": r[2], "pct": r[3]} for r in rows]},
              open("analysis/arxivtopics/staged_phases.json", "w"), indent=1)
    print("STAGEDONE", flush=True)
