#!/usr/bin/env python3
"""NON-NEGATIVE ARROWS — the last untested difference from the deployed model (operator 2026-07-28).

Everything tried so far spends or saves a PARAMETER. This spends a SIGN. The deployed model puts its
amplitudes through a softplus, so every arrow length is positive by construction; the closed-form
solver lets them go negative. That is not a cosmetic difference — with one shared tuning φ, a negative
amplitude is a 180° phase flip smuggled in for free, so seven "free signs" restore much of the
per-body phase freedom the staged ablation already showed to be harmful (−0.0145).

Forcing aᵢ ≥ 0 therefore removes a hidden degree of freedom without touching the parameter count: still
b + six amplitudes + φ = 8 numbers, but now every body must either agree with the shared tuning or fall
silent, and it cannot quietly oppose it.

Two-stage so it stays cheap and deterministic: search the grid unconstrained, keep the twenty best
tunings, then solve each under the bound and take the winner. The bound is on the amplitudes only —
the DC level b keeps its sign, since a field's baseline is not a physical arrow.

  python3 analysis/arxivtopics/eight_nonneg.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
from scipy.optimize import lsq_linear
import eight_param as EP
from eight_struct import prep, auc_of
from eight_final import WALLS, persistence

BODS, NB, COS, NG, ne = EP.BODS, EP.NB, EP.COS, EP.NG, EP.ne
TOPK = 20


def fit(wall, drop="jupiter", nonneg=True):
    w, ysq, MJ, hz = prep(wall)
    keep = [i for i in range(NB) if drop is None or i != BODS.index(drop)]
    K = 1 + len(keep)
    X = np.empty((NG, ne, K)); X[:, :, 0] = 1.0
    for q, i in enumerate(keep): X[:, :, 1+q] = COS[i]
    Xt, Xa = X[:, :wall], X[:, wall:hz]
    XaS = np.einsum('gtp,gtq->gpq', Xa, Xa); XaB = np.einsum('gtp->gp', Xa)
    I = 1e-8*np.eye(K)[None]
    lo = np.full(K, -np.inf); hi_b = np.full(K, np.inf)
    if nonneg: lo[1:] = 0.0                                  # arrows have length, not sign
    P = np.zeros((Tn, ne))
    for j in range(Tn):
        aw = EP.LAM/(MJ[j]**2)/max(hz-wall, 1)
        A = np.einsum('gtp,t,gtq->gpq', Xt, w[j], Xt) + aw*XaS + I
        b = np.einsum('gtp,t->gp', Xt, w[j]*ysq[j]) + aw*XaB*MJ[j]
        c = np.linalg.solve(A, b[..., None])[..., 0]
        r = ((np.einsum('gtp,gp->gt', Xt, c) - ysq[j][None])**2) @ w[j] \
            + aw*((np.einsum('gtp,gp->gt', Xa, c)-MJ[j])**2).sum(1)
        if not nonneg:
            g = int(np.argmin(r)); P[j] = np.maximum(X[g] @ c[g], 0)**2; continue
        sw = np.sqrt(w[j]); sa = np.sqrt(aw)
        best = (np.inf, None, None)
        for g in np.argsort(r)[:TOPK]:                        # re-solve the best tunings under the bound
            Ad = np.vstack([Xt[g]*sw[:, None], Xa[g]*sa])
            bd = np.concatenate([ysq[j]*sw, np.full(hz-wall, MJ[j]*sa)])
            s = lsq_linear(Ad, bd, bounds=(lo, hi_b), method="bvls")
            if s.cost < best[0]: best = (s.cost, int(g), s.x)
        _, g, cc = best
        P[j] = np.maximum(X[g] @ cc, 0)**2
    return auc_of(P, wall)


def nine_param():
    """THE SAME ABLATION ON THE MODEL THAT SHIPS. fit() defaults to drop="jupiter", so the −0.1778 above
    is a property of the 8-parameter, six-body variant — it was quoted for the deployed model for a
    while, and it should not have been. This arm passes drop=None: all seven arrows, nine parameters,
    exactly arxiv_fit.fit_final. Writes nonneg_nine.json, which is the file the model-of-record
    docstring and the /topics page cite."""
    free = np.array([fit(w, drop=None, nonneg=False) for w in WALLS])
    nn = np.array([fit(w, drop=None, nonneg=True) for w in WALLS])
    d = nn - free; se = float(d.std(ddof=1) / np.sqrt(len(d)))
    print(f"\n═══ THE DEPLOYED ROSTER · 9 params · all seven arrows · {len(WALLS)} origins ═══", flush=True)
    print("  free sign  " + " ".join(f"{a:+.3f}" for a in free) + f"   mean {free.mean():+.4f}", flush=True)
    print("  aᵢ ≥ 0     " + " ".join(f"{a:+.3f}" for a in nn) + f"   mean {nn.mean():+.4f}", flush=True)
    print(f"  Δ {d.mean():+.4f} ± {se:.4f}  ·  wins {int((d>0).sum())}/{len(d)}", flush=True)
    json.dump({"walls": [int(YEARS[w]) for w in WALLS], "free": [round(float(v), 4) for v in free],
               "nonneg": [round(float(v), 4) for v in nn],
               "delta_mean": round(float(d.mean()), 4), "se": round(se, 4),
               "wins": int((d > 0).sum()), "params": 9, "bodies": len(BODS)},
              open("analysis/arxivtopics/nonneg_nine.json", "w"), indent=1)


if __name__ == "__main__":
    print(f"═══ NON-NEGATIVE AMPLITUDES · 8 params · {len(WALLS)} origins ═══", flush=True)
    tab = {}
    for name, kw in (("8-param free sign", dict(nonneg=False)), ("8-param aᵢ ≥ 0", dict(nonneg=True))):
        tab[name] = [fit(w, **kw) for w in WALLS]
        print(f"  {name:20s} " + " ".join(f"{a:+.3f}" for a in tab[name]), flush=True)
    pe = np.array([persistence(w) for w in WALLS])
    a_free, a_nn = np.array(tab["8-param free sign"]), np.array(tab["8-param aᵢ ≥ 0"])
    d = a_nn - a_free; se = float(d.std(ddof=1)/np.sqrt(len(d)))
    print(f"\n  free sign  mean {a_free.mean():+.4f}   ·  aᵢ ≥ 0  mean {a_nn.mean():+.4f}", flush=True)
    print(f"  Δ {d.mean():+.4f} ± {se:.4f}  (t={d.mean()/max(se,1e-9):.2f}, wins {int((d>0).sum())}/{len(d)})", flush=True)
    print(f"  persistence {pe.mean():+.4f} · deployed 15-param (1996 only) +0.8193 · "
          f"aᵢ≥0 at 1996 {a_nn[-1]:+.4f}", flush=True)
    print(f"  → {'REAL IMPROVEMENT at 8 params' if d.mean() > 2*se else 'inside the noise — no improvement'}", flush=True)
    json.dump({"walls": [int(YEARS[w]) for w in WALLS], "auc": {k: [round(float(v),4) for v in tab[k]] for k in tab},
               "delta": {"mean": round(float(d.mean()),4), "se": round(se,4), "wins": int((d>0).sum())},
               "NOTE": "8 PARAMS, jupiter dropped — do NOT quote this for the 9-param model of record; "
                       "see nonneg_nine.json for that roster"},
              open("analysis/arxivtopics/eight_nonneg.json", "w"), indent=1)
    nine_param()                      # the number the docstring and the page actually cite
    print("NNDONE", flush=True)
