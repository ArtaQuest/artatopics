#!/usr/bin/env python3
"""HOW MUCH OF THE ATLAS IS SHARED? — the diagnostic that bounds any global model (2026-07-29).

Before asking whether a global model can match 251 independent fits, measure how much shared structure
there is to find. The deployed model gives every topic seven SIGNED arrows a_ji, and each vector is
already expressed in that topic's own frame (the design column is cos(θᵢ − φⱼ)), so the 251×7 matrix
is directly comparable across topics — up to ONE gauge choice per topic, because (φ, a) and
(φ+180°, −a) are the same model. Fix that gauge, then just look:

  SPECTRUM     the singular values of the gauge-fixed, scale-normalised arrow matrix. If one component
               explains most of the variance, a rank-1 global model MUST be able to work and the
               earlier shared-shape failure was about signs, not about sharing. If the spectrum is
               flat, no global model in this family can succeed and the search should stop.
  SIGN PATTERNS how many of the 64 distinguishable patterns are actually used, and how concentrated.
               A handful of patterns means an archetype dictionary is the right shape; 60-odd means
               the sign pattern is genuinely per-topic and must stay free.
  SUBSTITUTION replace every topic's arrow MAGNITUDES with the pooled mean while keeping its own sign
               pattern, tuning and level, and re-score. This is NOT an upper bound — a FITTED shared
               shape can beat the pooled mean, and the earlier shared-shape variant (eight_struct.py,
               g fitted by alternation) did: +0.7719. What the substitution measures is whether the
               magnitudes are INTERCHANGEABLE across topics, which is the question a global model
               actually turns on.

  python3 analysis/arxivtopics/global_ceiling.py
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import arxiv_fit as af


def fitted_arrows(Y, TH, wall):
    """Re-run the deployed sweep but keep the SIGNED coefficients and the chosen grid index."""
    Tn, ne, nb = Y.shape[0], TH.shape[0], TH.shape[1]
    tv = af.META["topic_valid"][:, :wall].astype(float)
    wy = np.clip(af.META["evidence"][:wall], 0, None) ** 0.75
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    Wa = np.zeros_like(W); Wa[:, wall - af.ANCHOR_K:] = (tv * wy[None])[:, wall - af.ANCHOR_K:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    Ysq = np.sqrt(Y); MJ = np.maximum((Ysq[:, :wall] * Wa).sum(1), 1e-3)
    hz = min(wall + af.HORIZON, ne); NG = len(af.GRID)
    X = np.empty((NG, ne, 1 + nb)); X[:, :, 0] = 1.0
    for i in range(nb): X[:, :, 1 + i] = np.cos(TH[:, i][None, :] - af.GRID[:, None])
    Xt, Xa = X[:, :wall], X[:, wall:hz]
    XaS = np.einsum('gtp,gtq->gpq', Xa, Xa); XaB = np.einsum('gtp->gp', Xa)
    I = 1e-8 * np.eye(1 + nb)[None]
    A = np.zeros((Tn, nb)); B = np.zeros(Tn); G = np.zeros(Tn, int); P = np.zeros((Tn, ne))
    for j in range(Tn):
        aw = af.LAM_HORIZON / (MJ[j] ** 2) / max(hz - wall, 1)
        M = np.einsum('gtp,t,gtq->gpq', Xt, W[j], Xt) + aw * XaS + I
        b = np.einsum('gtp,t->gp', Xt, W[j] * Ysq[j, :wall]) + aw * XaB * MJ[j]
        c = np.linalg.solve(M, b[..., None])[..., 0]
        r = (((np.einsum('gtp,gp->gt', Xt, c) - Ysq[j, :wall][None]) ** 2) @ W[j]
             + aw * ((np.einsum('gtp,gp->gt', Xa, c) - MJ[j]) ** 2).sum(1))
        g = int(np.argmin(r)); G[j] = g; A[j] = c[g][1:]; B[j] = c[g][0]
        P[j] = np.maximum(X[g] @ c[g], 0.0) ** 2
    return A, B, G, P, X, W, MJ, Xt, Xa, aw


def auc_at(Y, Yh, wall):
    tv = af.META["topic_valid"]; tvw = tv[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    return float(np.mean([1.0 - ((Y[:, wall + h] - Yh[:, wall + h]) ** 2).sum() /
                          max(((Y[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(af.HORIZON)]))


def main():
    names, Y, labels, future = af.load_lunar()
    TH, _ = af.sky_lunar(labels + future)
    n = Y.shape[1]; wall = n - 30
    A, B, G, P, X, W, MJ, Xt, Xa, _ = fitted_arrows(Y, TH, wall)
    print(f"═══ HOW MUCH SHARED STRUCTURE IS THERE? · wall {labels[wall]} · {A.shape[0]} topics ═══", flush=True)
    print(f"  baseline AUC at this wall: {auc_at(Y, P, wall):+.4f}", flush=True)

    # GAUGE FIX: (φ, a) ≡ (φ+180°, −a). Choose the sign that puts each topic's LARGEST arrow positive,
    # so the 251 vectors live in one half-space and averaging them means something.
    lead = np.abs(A).argmax(1)
    gauge = np.sign(A[np.arange(len(A)), lead]); gauge[gauge == 0] = 1.0
    Ag = A * gauge[:, None]
    nrm = np.linalg.norm(Ag, axis=1, keepdims=True)
    An = Ag / np.maximum(nrm, 1e-12)                      # scale removed: shape only

    U, S, Vt = np.linalg.svd(An - 0*An.mean(0), full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    print(f"\n  SPECTRUM of the 251×7 arrow-shape matrix (gauge-fixed, unit-norm rows):", flush=True)
    print("    component   " + "  ".join(f"{i+1:>6d}" for i in range(7)), flush=True)
    print("    share       " + "  ".join(f"{v:>6.3f}" for v in ev), flush=True)
    print("    cumulative  " + "  ".join(f"{v:>6.3f}" for v in np.cumsum(ev)), flush=True)
    print(f"    participation ratio (effective rank): {1.0/ (ev**2).sum():.2f} of 7", flush=True)
    print(f"    leading direction: " + " ".join(f"{b[:3]} {v:+.2f}" for b, v in zip(af.BODIES, Vt[0])), flush=True)

    sgn = (Ag > 0).astype(int)
    pats = collections.Counter(tuple(r) for r in sgn)
    print(f"\n  SIGN PATTERNS: {len(pats)} distinct of {2**len(af.BODIES)} possible "
          f"({len(pats)/2**len(af.BODIES)*100:.0f}% of the space)", flush=True)
    top = pats.most_common(6)
    print(f"    most common: " + " · ".join(f"{''.join('+' if b else '-' for b in p)}×{c}" for p, c in top), flush=True)
    print(f"    top-6 cover {sum(c for _, c in top)}/{len(sgn)} topics "
          f"({sum(c for _,c in top)/len(sgn)*100:.0f}%)", flush=True)
    print(f"    singletons: {sum(1 for _, c in pats.items() if c == 1)} patterns used by exactly one topic", flush=True)

    # THE CEILING: keep each topic's own sign pattern, tuning and level; impose the POOLED magnitudes.
    mag = np.abs(Ag)
    magn = mag / np.maximum(mag.sum(1, keepdims=True), 1e-12)
    gbar = magn.mean(0)                                   # the pooled magnitude shape
    print(f"\n  POOLED MAGNITUDE SHAPE: " + " ".join(f"{b[:3]} {v:.3f}" for b, v in zip(af.BODIES, gbar)), flush=True)
    res = {}
    for lab, Anew in (("free magnitudes (baseline)", A),
                      ("POOLED magnitudes, own signs", np.sign(Ag) * gauge[:, None] * (gbar[None, :] * mag.sum(1, keepdims=True))),
                      ("POOLED magnitudes AND pooled signs", (gauge[:, None] * gbar[None, :] * mag.sum(1, keepdims=True)))):
        Pn = np.stack([np.maximum(X[G[j], :, 0] * B[j] + X[G[j], :, 1:] @ Anew[j], 0.0) ** 2
                       for j in range(len(A))])
        res[lab] = auc_at(Y, Pn, wall)
        print(f"    {lab:38s} AUC {res[lab]:+.4f}", flush=True)
    print(f"\n  READ: this is a SUBSTITUTION test, not a bound — a fitted shared shape can beat the pooled", flush=True)
    print(f"  mean (eight_struct.py's fitted g reached +0.7719). What it shows is whether magnitudes are", flush=True)
    print(f"  interchangeable between topics while tuning, level and signs stay free.", flush=True)
    json.dump({"wall": labels[wall], "baseline_auc": auc_at(Y, P, wall),
               "spectrum": [round(float(v), 4) for v in ev],
               "effective_rank": round(float(1.0/(ev**2).sum()), 2),
               "leading_direction": {b: round(float(v), 3) for b, v in zip(af.BODIES, Vt[0])},
               "distinct_sign_patterns": len(pats),
               "top_patterns": [["".join('+' if b else '-' for b in p), c] for p, c in top],
               "pooled_shape": {b: round(float(v), 4) for b, v in zip(af.BODIES, gbar)},
               "ceilings": {k: round(v, 4) for k, v in res.items()}},
              open("analysis/arxivtopics/global_ceiling.json", "w"), indent=1)
    print("CEILDONE", flush=True)


if __name__ == "__main__":
    main()
