#!/usr/bin/env python3
"""ONE PHASE + HORIZON ANCHOR, STILL CLOSED-FORM (operator 2026-07-28).

The deployed model's biggest single ingredient is the horizon anchor (+0.12): it penalises the model's
own forecast for drifting away from the level the field held in its last few TRAINING years. Because
the receiver is linear in its coefficients on the √ scale, that penalty is a LINEAR constraint — it
folds into the same least squares as extra Tikhonov rows:

    minimise   Σ_{t<wall} w_t (X_t·c − √y_t)²   +   λ Σ_{t=wall}^{wall+30} ( X_t·c − m )² / m²
                └─ fit the history ─┘                └─ do not wander in the forecast ─┘

with m = the field's N^¾-weighted mean √share over its last ANCHOR_K training years. Both halves are
quadratic in c, so the whole thing is still one exact solve per grid angle. The model therefore keeps
every property that made the grid attractive: ONE interpretable number per topic, found by exhaustive
search, no gradient descent, no seeds, identical bits on every refit.

λ is chosen on the INNER rehearsal wall (fit ≤1965, judged 1966-95) and applied unchanged to the outer.

  python3 analysis/arxivtopics/one_phase_anchored.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *

BI = [BODIES_ALL.index(b) for b in CHAMPION_BODIES]
NB = len(BI)
TH = TH_ALL[:, BI]; ne = TH.shape[0]
GRID = np.deg2rad(np.arange(0, 360, 1.0))
GEARS = np.arange(1, NB + 1).astype(float)      # selected on train residual in one_phase_grid.py
ANCHOR_K = 5

# design matrices per grid angle, shared by every topic and wall
XS = np.stack([np.concatenate([np.ones((ne, 1)), np.cos(TH - GEARS[None, :] * p)], 1) for p in GRID])

def anchor_level(wall):
    tv = train_mask(wall).astype(float)
    wy = np.clip(N[:wall], 0, None) ** 0.75
    Wa = np.zeros_like(tv); Wa[:, wall - ANCHOR_K:] = (tv * wy[None])[:, wall - ANCHOR_K:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv * wy[None])[bad]
    Wa /= np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    return (np.sqrt(Y[:, :wall]) * Wa).sum(1)

def fit(wall, lam):
    """One exact solve per (topic, grid angle); returns predictions and the chosen phase."""
    tv = train_mask(wall).astype(float)
    w = tv * (np.clip(N[:wall], 0, None) ** 0.75)[None]
    w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
    ysq = np.sqrt(Y[:, :wall])
    m = anchor_level(wall)
    hz = min(wall + HORIZON, ne)
    P = np.zeros((Tn, ne)); ph = np.zeros(Tn)
    for j in range(Tn):
        mj = max(float(m[j]), 1e-3)
        aw = lam / (mj * mj) / max(hz - wall, 1)          # spread the anchor over the window
        best, bc, ba = np.inf, None, 0
        for a in range(len(GRID)):
            Xt = XS[a, :wall]; Xa = XS[a, wall:hz]
            A = Xt.T @ (Xt * w[j][:, None]) + aw * (Xa.T @ Xa) + 1e-8 * np.eye(1 + NB)
            b = Xt.T @ (w[j] * ysq[j]) + aw * (Xa.T @ np.full(hz - wall, mj))
            c = np.linalg.solve(A, b)
            r = float((w[j] * (Xt @ c - ysq[j]) ** 2).sum() + aw * ((Xa @ c - mj) ** 2).sum())
            if r < best: best, bc, ba = r, c, a
        P[j] = np.maximum(XS[ba] @ bc, 0) ** 2
        ph[j] = np.rad2deg(GRID[ba]) % 360
    return P, ph

def sc(P, wall):
    tvw = TV[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    hi = min(wall + HORIZON, n)
    yt, yp = Y[:, wall:hi], P[:, wall:hi]
    curve = [1 - ((yt[:, h] - yp[:, h]) ** 2).sum() / max(((yt[:, h] - mu) ** 2).sum(), 1e-9) for h in range(hi - wall)]
    sk = 1 - ((yt - yp) ** 2).sum(1) / np.maximum(((yt - mu[:, None]) ** 2).sum(1), 1e-9)
    return float(np.mean(curve)), float(np.median(sk)), float((sk > 0).mean() * 100)

if __name__ == "__main__":
    print("═══ INNER WALL — choose λ (1996+ never consulted) ═══", flush=True)
    best_lam, best_a = None, -9
    for lam in (0.0, 0.03, 0.1, 0.3, 1.0, 3.0):
        a, s, p = sc(fit(WALL_INNER, lam)[0], WALL_INNER)
        star = ""
        if a > best_a: best_lam, best_a, star = lam, a, "  ←"
        print(f"  λ={lam:<5g} inner AUC {a:+.4f} · skill {s:+.4f}{star}", flush=True)
    print(f"  chosen λ = {best_lam}", flush=True)
    print("\n═══ OUTER WALL — fitted once with that λ ═══", flush=True)
    P, ph = fit(WALL_OUTER, best_lam)
    a, s, p = sc(P, WALL_OUTER)
    print(f"  ONE PHASE + anchor (grid)   AUC {a:+.4f} · median skill {s:+.4f} · {p:.1f}%>0", flush=True)
    print(f"  one phase, no anchor        +0.6545", flush=True)
    print(f"  deployed multi-phase        +0.8193  (gradient descent + anchor + early stop + softplus)", flush=True)
    print(f"  persistence                 +0.7344", flush=True)
    import collections
    print(f"  sign spread: {dict(sorted(collections.Counter((ph//30).astype(int)).items()))}", flush=True)
    json.dump({"lam": best_lam, "auc": a, "skill": s, "pct": p,
               "phase_deg": [round(float(v), 1) for v in ph]},
              open("analysis/arxivtopics/one_phase_anchored.json", "w"), indent=1)
    print("ANCHDONE", flush=True)
