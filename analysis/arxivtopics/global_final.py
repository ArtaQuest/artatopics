#!/usr/bin/env python3
"""THE GLOBAL MODEL — a rank-3 shared basis, with a field-level prior on top (operator 2026-07-29).

Two independent wins came out of the global search, and they work by DIFFERENT mechanisms, so the
question is whether they add:

  RANK-3 BASIS   the 251x7 arrow matrix is truncated to the top-3 right singular vectors of the free
                 fit's OWN arrows at that wall. Every topic then owns a level, a tuning and three
                 loadings — 5 numbers — inside a basis shared by the whole atlas. 1,276 parameters
                 against the free model's 2,259, and it wins on all four measures. This rank was
                 PREDICTED IN ADVANCE by global_ceiling.py: the shapes carry 97.9% of their variance
                 in three components with an effective rank of 2.27, measured without looking at a
                 single AUC.
  FIELD PRIOR    each topic's arrows are shrunk toward the shared signed spectrum of its OpenAlex
                 FIELD (26 of them), the prior placed inside the closed-form solve so the tuning is
                 re-chosen under it. Shrinking toward ZERO at the same strength is WORSE than not
                 shrinking at all (+0.8633 vs +0.8751) — so this is not regularisation, it is the
                 claim that fields within a discipline share a receiver.

Restricting to a basis and shrinking toward a group mean are not the same operation, so they may
compose. Here the prior is applied to the LOADINGS inside the rank-3 space rather than to the raw
arrows, which is the natural way to combine them and also the cheaper one.

GAUGE, again: (phi, u) and (phi+180, -u) are the same model, so a field's mean loading is meaningless
until the flip is resolved. Resolved per field by the leading eigenvector of sum(u u'), exactly as the
pooling result does for raw arrows.

tau is chosen on the FIRST NINE origins only and applied unchanged to 1990/1993/1996.

  python3 analysis/arxivtopics/global_final.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import arxiv_fit as af
from global_ceiling import fitted_arrows

names, Y, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
n, ne, nb = Y.shape[1], TH.shape[0], TH.shape[1]
WALLS = list(range(n - 63, n - 29, 3))
SEL = 9
GRP = np.unique([af.META["field"][nm] for nm in names], return_inverse=True)[1]
NF = int(GRP.max()) + 1
COS = np.stack([np.cos(TH[:, i][None, :] - af.GRID[:, None]) for i in range(nb)])
TAUS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
RANK = 3


def auc_at(Yh, wall):
    tv = af.META["topic_valid"]; tvw = tv[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    return float(np.mean([1.0 - ((Y[:, wall + h] - Yh[:, wall + h]) ** 2).sum() /
                max(((Y[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(30)]))


def field_prior(U):
    """Gauge-resolved, scale-preserving field mean of the rank-r loadings."""
    T = np.zeros_like(U)
    for f in range(NF):
        idx = np.where(GRP == f)[0]
        if len(idx) == 0: continue
        Uf = U[idx]
        nrm = np.linalg.norm(Uf, axis=1, keepdims=True)
        e = Uf / np.maximum(nrm, 1e-12)
        M = e.T @ e                                   # leading eigenvector resolves the 180° flip
        v = np.linalg.eigh(M)[1][:, -1]
        s = np.sign(e @ v); s[s == 0] = 1.0
        g = (e * s[:, None]).mean(0)
        g = g / max(np.linalg.norm(g), 1e-12)
        T[idx] = (nrm * s[:, None]) * g[None, :]      # keep each topic's own scale and gauge
    return T


def fit(wall, r=RANK, tau=0.0, rounds=2):
    A, B, G, P0, X0, W0, MJ0, _, _, _ = fitted_arrows(Y, TH, wall)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt[:r].T                                                   # (nb, r) GLOBAL basis, train-only
    tv = af.META["topic_valid"][:, :wall].astype(float)
    wy = np.clip(af.META["evidence"][:wall], 0, None) ** 0.75
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    Wa = np.zeros_like(W); Wa[:, wall - af.ANCHOR_K:] = (tv * wy[None])[:, wall - af.ANCHOR_K:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    Ysq = np.sqrt(Y); MJ = np.maximum((Ysq[:, :wall] * Wa).sum(1), 1e-3)
    hz = min(wall + af.HORIZON, ne); NG = len(af.GRID); K = 1 + r
    Xr = np.empty((NG, ne, K)); Xr[:, :, 0] = 1.0
    for k in range(r):
        Xr[:, :, 1 + k] = np.einsum('i,ige->ge', V[:, k], COS)
    Xt, Xa = Xr[:, :wall], Xr[:, wall:hz]
    XaS = np.einsum('gtp,gtq->gpq', Xa, Xa); XaB = np.einsum('gtp->gp', Xa)
    I = 1e-8 * np.eye(K)[None]
    D = np.eye(K); D[0, 0] = 0.0                                   # never shrink the level

    def solve(target):
        Pn = np.zeros((Y.shape[0], ne)); U = np.zeros((Y.shape[0], r))
        for j in range(Y.shape[0]):
            aw = af.LAM_HORIZON / (MJ[j] ** 2) / max(hz - wall, 1)
            M = np.einsum('gtp,t,gtq->gpq', Xt, W[j], Xt) + aw * XaS + I + tau * D[None]
            b = np.einsum('gtp,t->gp', Xt, W[j] * Ysq[j, :wall]) + aw * XaB * MJ[j]
            if target is not None:
                b = b + tau * np.concatenate([[0.0], target[j]])[None, :]
            c = np.linalg.solve(M, b[..., None])[..., 0]
            rr = ((((np.einsum('gtp,gp->gt', Xt, c) - Ysq[j, :wall][None]) ** 2) @ W[j])
                  + aw * ((np.einsum('gtp,gp->gt', Xa, c) - MJ[j]) ** 2).sum(1))
            g = int(np.argmin(rr))
            Pn[j] = np.maximum(Xr[g] @ c[g], 0.0) ** 2; U[j] = c[g][1:]
        return Pn, U

    Pn, U = solve(None)
    if tau > 0:
        for _ in range(rounds):
            Pn, U = solve(field_prior(U))
    return Pn


def main():
    print(f"═══ GLOBAL MODEL: rank-{RANK} shared basis + field prior · {len(WALLS)} origins ═══", flush=True)
    print(f"    tau chosen on the first {SEL} origins only · {NF} OpenAlex fields", flush=True)
    res = {}
    for tau in TAUS:
        t0 = time.time()
        a = np.array([auc_at(fit(w, RANK, tau), w) for w in WALLS])
        res[tau] = a
        print(f"  tau={tau:<6g} select9 {a[:SEL].mean():+.4f}  held3 {a[SEL:].mean():+.4f}  "
              f"all12 {a.mean():+.4f}  1996 {a[-1]:+.4f}   [{time.time()-t0:.0f}s]", flush=True)
    pick = max(TAUS, key=lambda t: res[t][:SEL].mean())
    a = res[pick]
    par_topic = 2 + RANK                                   # level + tuning + r loadings
    par_total = Y.shape[0] * par_topic + nb * RANK + (NF * RANK if pick > 0 else 0)
    print(f"\n  CHOSEN ON THE FIRST {SEL} ORIGINS: tau = {pick}", flush=True)
    print(f"    {'model':34s}{'select9':>10s}{'held3':>9s}{'all12':>9s}{'1996':>9s}{'params':>9s}", flush=True)
    for lab, v, p in (("free per-topic (model of record)", [0.8922, 0.8237, 0.8751, 0.7990], 2259),
                      ("field pooling (rank 7)", [0.8956, 0.8330, 0.8800, 0.8146], 2441),
                      (f"rank-{RANK} basis, no prior",
                       [res[0.0][:SEL].mean(), res[0.0][SEL:].mean(), res[0.0].mean(), res[0.0][-1]],
                       Y.shape[0] * par_topic + nb * RANK),
                      (f"rank-{RANK} basis + field prior",
                       [a[:SEL].mean(), a[SEL:].mean(), a.mean(), a[-1]], par_total),
                      ("carry-forward persistence", [0.8780, 0.7704, 0.8511, 0.7344], 0)):
        print(f"    {lab:34s}{v[0]:>+10.4f}{v[1]:>+9.4f}{v[2]:>+9.4f}{v[3]:>+9.4f}{p:>9d}", flush=True)
    json.dump({"rank": RANK, "taus": TAUS, "chosen_tau": pick, "n_fields": NF,
               "origins": [labels[w] for w in WALLS],
               "auc": {str(t): [round(float(v), 4) for v in res[t]] for t in TAUS},
               "params_total": par_total, "params_per_topic": par_topic},
              open("analysis/arxivtopics/global_final.json", "w"), indent=1)
    print("GLOBALDONE", flush=True)


if __name__ == "__main__":
    main()
