#!/usr/bin/env python3
"""STRUCTURED SPECTRA — spending the budget instead of deleting from it (operator 2026-07-28: "≤8 params").

Leave-one-out is dead: chosen honestly on early walls it returns +0.0000 (eight_robust.py). Deleting a
body throws away signal to save a parameter. The alternative is to keep all seven bodies and stop
letting their amplitudes be seven INDEPENDENT numbers — tie them to a shape with far fewer knobs.

  powerlaw   aᵢ = s·(Pᵢ/P̄)^κ                     4 params (b, s, κ, φ)
             A receiver's response should fall off smoothly with frequency, not jump body to body.
             Two numbers — an overall gain and a spectral slope — replace seven free amplitudes, and
             κ is fitted per topic, so a field can still choose slow-dominated or fast-dominated.
  shape      aᵢ = s·gᵢ, g GLOBAL (fitted once, shared by all 251)   3 params/topic + 7 shared
             The shape of the response is a property of the SKY, not of a research field; only the
             gain and the tuning are personal. Fitted by alternation, never seeing past the wall.
  shape+1    the same, but one body keeps a free amplitude                          4 params
  meanfill_X aₓ frozen at its cross-topic mean rather than deleted                  8 params
             The honest version of leave-one-out: a body we cannot afford to fit individually still
             contributes its typical amount instead of vanishing.

Every family is selected on the FOUR EARLY WALLS and only then reported at 1996, so the headline
number is never the one that chose the model.

  python3 analysis/arxivtopics/eight_struct.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import eight_param as EP

BODS, NB, TH = EP.BODS, EP.NB, EP.TH
PER = np.array([EP.PERIOD[b] for b in BODS])
GLOG = np.exp(np.log(PER).mean())
GRID, NG, ne = EP.GRID, EP.NG, EP.ne
COS = EP.COS                                   # (NB, NG, ne)
KAP = np.linspace(-1.2, 1.2, 13)               # spectral slopes: fast-dominated … slow-dominated
WALLS = [n - 60, n - 52, n - 45, n - 37, n - 30]


def prep(wall):
    tv = train_mask(wall).astype(float); wy = np.clip(N[:wall], 0, None) ** 0.75
    w = tv * wy[None]; w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
    ysq = np.sqrt(Y[:, :wall])
    Wa = np.zeros_like(tv); Wa[:, wall-EP.AK:] = (tv*wy[None])[:, wall-EP.AK:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv*wy[None])[bad]; Wa /= np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    return w, ysq, np.maximum((ysq * Wa).sum(1), 1e-3), min(wall + HORIZON, ne)


def solve(X, wall, hz, w_j, y_j, m_j, off=None):
    """Anchored weighted LS over a grid of designs; returns (best index, coefs, full-series curve)."""
    K = X.shape[2]
    tgt = y_j if off is None else y_j - off[:wall]
    Xt, Xa = X[:, :wall], X[:, wall:hz]
    ao = np.zeros(hz-wall) if off is None else off[wall:hz]
    aw = EP.LAM / (m_j ** 2) / max(hz - wall, 1)
    A = np.einsum('gtp,t,gtq->gpq', Xt, w_j, Xt) + aw*np.einsum('gtp,gtq->gpq', Xa, Xa) + 1e-8*np.eye(K)[None]
    b = np.einsum('gtp,t->gp', Xt, w_j*tgt) + aw*np.einsum('gtp,gt->gp', Xa, np.broadcast_to(m_j-ao, Xa.shape[:2]))
    c = np.linalg.solve(A, b[..., None])[..., 0]
    r = ((np.einsum('gtp,gp->gt', Xt, c) - tgt[None])**2 @ w_j
         + aw*((np.einsum('gtp,gp->gt', Xa, c) + ao[None] - m_j)**2).sum(1))
    g = int(np.argmin(r))
    base = 0.0 if off is None else off
    return g, c[g], np.maximum(X[g] @ c[g] + base, 0)**2


def auc_of(P, wall):
    tvw = TV[:, :wall].astype(float)
    mu = (Y[:, :wall]*tvw).sum(1)/np.maximum(tvw.sum(1), 1.0)
    hi = min(wall+HORIZON, n); yt, yp = Y[:, wall:hi], P[:, wall:hi]
    return float(np.mean([1-((yt[:,h]-yp[:,h])**2).sum()/max(((yt[:,h]-mu)**2).sum(),1e-9) for h in range(hi-wall)]))


def run_powerlaw(wall):
    w, ysq, MJ, hz = prep(wall)
    SH = np.stack([sum(((PER[i]/GLOG)**k)*COS[i] for i in range(NB)) for k in KAP])   # (NK,NG,ne)
    X = np.empty((len(KAP)*NG, ne, 2)); X[:, :, 0] = 1.0
    X[:, :, 1] = SH.reshape(-1, ne)
    P = np.stack([solve(X, wall, hz, w[j], ysq[j], MJ[j])[2] for j in range(Tn)])
    return auc_of(P, wall)


def run_shape(wall, free=None, iters=4):
    """a_i = s_j·g_i with g GLOBAL. Alternates per-topic (b,s,φ) with a pooled solve for g."""
    w, ysq, MJ, hz = prep(wall)
    g = np.ones(NB)
    fi = BODS.index(free) if free else None
    for it in range(iters):
        SH = sum(g[i]*COS[i] for i in range(NB) if i != fi)                            # (NG,ne)
        K = 2 + (1 if fi is not None else 0)
        X = np.empty((NG, ne, K)); X[:, :, 0] = 1.0; X[:, :, 1] = SH
        if fi is not None: X[:, :, 2] = COS[fi]
        sol = [solve(X, wall, hz, w[j], ysq[j], MJ[j]) for j in range(Tn)]
        if it == iters-1: break
        # pooled update of g: columns are s_j·cos(θ_i − φ_j), target is √y − b_j (− free body)
        A = np.zeros((NB, NB)); bb = np.zeros(NB)
        for j, (gi, c, _) in enumerate(sol):
            D = np.stack([c[1]*COS[i][gi, :wall] for i in range(NB)], 1)               # (wall,NB)
            tg = ysq[j] - c[0] - (c[2]*COS[fi][gi, :wall] if fi is not None else 0.0)
            A += D.T @ (D * w[j][:, None]); bb += D.T @ (w[j]*tg)
        if fi is not None: A[fi, :] = 0; A[:, fi] = 0; A[fi, fi] = 1; bb[fi] = g[fi]
        gn = np.linalg.solve(A + 1e-6*np.eye(NB), bb)
        nz = np.linalg.norm(gn)
        if np.isfinite(nz) and nz > 1e-9: g = gn/nz*np.sqrt(NB)                        # gauge: |g| fixed
    return auc_of(np.stack([s[2] for s in sol]), wall), g


def run_meanfill(wall, body):
    """Freeze one body's amplitude at the cross-topic mean from the unconstrained fit."""
    w, ysq, MJ, hz = prep(wall)
    k = BODS.index(body)
    X0 = np.empty((NG, ne, 1+NB)); X0[:, :, 0] = 1.0
    for i in range(NB): X0[:, :, 1+i] = COS[i]
    sol0 = [solve(X0, wall, hz, w[j], ysq[j], MJ[j]) for j in range(Tn)]
    abar = float(np.median([c[1+k] for _, c, _ in sol0]))
    keep = [i for i in range(NB) if i != k]
    X = np.empty((NG, ne, NB)); X[:, :, 0] = 1.0
    for q, i in enumerate(keep): X[:, :, 1+q] = COS[i]
    P = np.stack([solve(X, wall, hz, w[j], ysq[j], MJ[j], off=abar*COS[k][sol0[j][0]])[2] for j in range(Tn)])
    return auc_of(P, wall)


if __name__ == "__main__":
    print("═══ STRUCTURED SPECTRA · selected on the four EARLY walls, reported at 1996 ═══", flush=True)
    print(f"    {'family':18s}{'params':>7s} " + " ".join(f"{YEARS[w]:>8d}" for w in WALLS) + "   early-mean", flush=True)
    tab, npar = {}, {}
    ent = [("powerlaw", 4, lambda w: run_powerlaw(w)),
           ("shape", 3, lambda w: run_shape(w)[0]),
           ("shape+saturn", 4, lambda w: run_shape(w, free="saturn")[0]),
           ("shape+neptune", 4, lambda w: run_shape(w, free="neptune")[0])] + \
          [(f"meanfill_{b}", 8, (lambda bb: (lambda w: run_meanfill(w, bb)))(b)) for b in BODS]
    for name, K, fn in ent:
        row = [fn(w) for w in WALLS]
        tab[name], npar[name] = row, K
        print(f"    {name:18s}{K:>7d} " + " ".join(f"{a:+8.4f}" for a in row) + f"   {np.mean(row[:-1]):+9.4f}", flush=True)
    ref = [+0.8781, +0.8984, +0.8991, +0.8558, +0.7990]     # all7, 9 params (eight_robust.py)
    print(f"    {'all7 (9 params)':18s}{9:>7d} " + " ".join(f"{a:+8.4f}" for a in ref) + f"   {np.mean(ref[:-1]):+9.4f}", flush=True)
    early = {v: float(np.mean(tab[v][:-1])) for v in tab}
    pick = max(early, key=early.get)
    print(f"\n  CHOSEN ON EARLY WALLS: {pick} ({npar[pick]} params, early-mean {early[pick]:+.4f})", flush=True)
    print(f"  its 1996 score, out of sample: {tab[pick][-1]:+.4f}  vs 9-param +0.7990 "
          f"→ {tab[pick][-1]-0.7990:+.4f}", flush=True)
    _, gfit = run_shape(WALLS[-1])
    print(f"  fitted global shape g (slow→fast): " + " ".join(f"{b[:3]} {v:+.2f}" for b, v in zip(BODS, gfit)), flush=True)
    json.dump({"walls": [int(YEARS[w]) for w in WALLS], "auc": tab, "params": npar,
               "early_pick": pick, "g": gfit.tolist()},
              open("analysis/arxivtopics/eight_struct.json", "w"), indent=1)
    print("STRUCTDONE", flush=True)
