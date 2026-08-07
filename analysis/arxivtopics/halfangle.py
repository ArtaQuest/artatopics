#!/usr/bin/env python3
"""HALF-ANGLE VARIANT (operator 2026-08-07): y_j = (max(b_j + SUM_i a_ji cos((theta_i + phi_j)/2), 0))^2

Two notes on the definition, stated up front. With phi_j fitted, +phi vs -phi is a pure
reparameterisation, so the substantive change is the /2: every body's wave now completes one cycle
per TWO orbits (pluto's 248-year rhythm becomes a 496-year wave). And halving an angle that lives
mod 360 has a branch: cos((theta+phi)/2) changes sign where theta wraps past zero, so that
discontinuity is part of the model being tested, exactly as written.

Same everything else as the record: sqrt-scale weighted least squares, N^0.75 year weights, the
horizon anchor as extra rows, a 1-degree sweep -- widened to 0..359 because doubling the period
halves the gauge (phi -> phi+360 is the sign flip the signed amplitudes absorb). Twelve origins,
headline wall, reference = the record model at the same walls.

  python3 analysis/arxivtopics/halfangle.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import arxiv_fit as af

names, Y, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
n = Y.shape[1]; ne = TH.shape[0]; nb = TH.shape[1]
WALLS = list(range(n - 63, n - 29, 3))


def fit_halfangle(wall):
    GRID = np.deg2rad(np.arange(0.0, 360.0, 1.0)); NG = len(GRID)
    d = 1 + nb
    tv = af.META["topic_valid"][:, :wall].astype(float)
    wy = np.clip(af.META["evidence"][:wall], 0, None) ** 0.75
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    Wa = np.zeros_like(W); Wa[:, wall - af.ANCHOR_K:] = (tv * wy[None])[:, wall - af.ANCHOR_K:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    Ysq = np.sqrt(Y); MJ = np.maximum((Ysq[:, :wall] * Wa).sum(1), 1e-3)
    hz = min(wall + af.HORIZON, ne)
    X = np.empty((NG, ne, d)); X[:, :, 0] = 1.0
    for i in range(nb):
        X[:, :, 1 + i] = np.cos((TH[:, i][None, :] + GRID[:, None]) / 2.0)
    Xt, Xa = X[:, :wall], X[:, wall:hz]
    XaS = np.einsum('gtp,gtq->gpq', Xa, Xa); XaB = np.einsum('gtp->gp', Xa)
    I = 1e-8 * np.eye(d)[None]
    P = np.zeros((Y.shape[0], ne))
    for j in range(Y.shape[0]):
        aw = af.LAM_HORIZON / (MJ[j] ** 2) / max(hz - wall, 1)
        A = np.einsum('gtp,t,gtq->gpq', Xt, W[j], Xt) + aw * XaS + I
        b = np.einsum('gtp,t->gp', Xt, W[j] * Ysq[j, :wall]) + aw * XaB * MJ[j]
        c = np.linalg.solve(A, b[..., None])[..., 0]
        r = (((np.einsum('gtp,gp->gt', Xt, c) - Ysq[j, :wall][None]) ** 2) @ W[j]
             + aw * ((np.einsum('gtp,gp->gt', Xa, c) - MJ[j]) ** 2).sum(1))
        g = int(np.argmin(r))
        P[j] = np.maximum(X[g] @ c[g], 0.0) ** 2
    return P


def auc_at(Yh, wall):
    tv = af.META["topic_valid"]; tvw = tv[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    return float(np.mean([1.0 - ((Y[:, wall + h] - Yh[:, wall + h]) ** 2).sum() /
                max(((Y[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(min(30, n - wall))]))


def main():
    print("═══ HALF-ANGLE vs THE RECORD · twelve origins ═══", flush=True)
    t0 = time.time()
    ha = np.array([auc_at(fit_halfangle(w), w) for w in WALLS])
    rec = np.array([auc_at(af.fit_final(Y, TH, w)[0], w) for w in WALLS])
    print(f"  [{time.time()-t0:.0f}s]", flush=True)
    print(f"  half-angle  mean {ha.mean():+.4f} · 1996 {ha[-1]:+.4f}   per-origin " +
          " ".join(f"{v:+.3f}" for v in ha), flush=True)
    print(f"  record      mean {rec.mean():+.4f} · 1996 {rec[-1]:+.4f}   per-origin " +
          " ".join(f"{v:+.3f}" for v in rec), flush=True)
    d = ha - rec
    print(f"  half-angle − record: {d.mean():+.4f} · wins {int((d>0).sum())}/12", flush=True)
    json.dump({"walls": [labels[w] for w in WALLS],
               "half_angle": [round(float(v), 4) for v in ha],
               "record": [round(float(v), 4) for v in rec],
               "delta_mean": round(float(d.mean()), 4), "wins": int((d > 0).sum())},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "halfangle.json"), "w"), indent=1)
    print("HADONE", flush=True)


if __name__ == "__main__":
    main()
