#!/usr/bin/env python3
"""DOES A TWELVE-SIGN KL BALANCE TERM IMPROVE THE AUC? (operator 2026-07-29)

The superseded gradient model carried a twelve-sign KL term: push each field's phase to fall
DECISIVELY inside one season while pushing the atlas to use ALL TWELVE. Those two together are exactly
the mutual information between field and season. The deterministic model dropped it — there is no
optimiser to attach it to — and season occupancy went from a balanced 19-23 to a lumpy 9..41.

But the term does not need gradients. It only touches WHICH TUNING IS CHOSEN, and the tuning is chosen
by an exhaustive 1° sweep. So it belongs in the SELECTION, not in the solve: amplitudes and level stay
closed-form, and the sweep simply scores each candidate tuning by its fit AND by what that choice does
to the atlas. The coupling across topics is handled by coordinate descent — sweep the topics in a
fixed order, recomputing the atlas marginal each time — which is deterministic, seedless, and
converges to a fixed point we can check.

    L = Σⱼ r̃ⱼ(φⱼ)  +  β·[ Σⱼ H(qⱼ) − T·H(q̄) ]          q_j = softmax(cos(cls_j − centres)/τ)

  r̃ⱼ = the anchored weighted residual, divided by the field's own level² so β means the same thing to
       a big field and a small one — without that, the term would only ever move the quiet fields.
  cls_j = THE CLASSIFYING ANGLE THE PAGE ACTUALLY SHOWS: pluto's canonical tuning, i.e. φ if pluto's
       arrow is positive and φ+180° if it is negative. Penalising φ itself would balance a quantity no
       reader ever sees.
  Σ H(qⱼ) ↓ decisive placement · T·H(q̄) ↑ all twelve seasons used · together = mutual information.

HONESTY ON β: it is a new constant, so it is chosen on the NINE EARLIEST origins only (1963..1987) and
then applied unchanged to 1990, 1993 and 1996 — which include the headline wall. The last audit caught
this project selecting a design choice on the wall it then published; not repeating that here.

  python3 analysis/arxivtopics/kl_balance.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import arxiv_fit as af

TAU = 0.15                                   # the superseded model's temperature, carried over unchanged
CENTRES = np.deg2rad(np.arange(12) * 30.0 + 15.0)
SWEEPS = 6
BETAS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
WALLS_ALL = None                             # filled in main()
SELECT_N = 9                                 # the first nine origins choose β; the rest are held out


def per_topic_grid(Y, TH, fit_end):
    """Everything the sweep needs, computed ONCE: for every topic and every candidate tuning, the
    closed-form coefficients, the anchored residual, and the resulting prediction path."""
    Tn, ne, nb = Y.shape[0], TH.shape[0], TH.shape[1]
    tv = af.META["topic_valid"][:, :fit_end].astype(float)
    wy = np.clip(af.META["evidence"][:fit_end], 0, None) ** 0.75
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    Wa = np.zeros_like(W)
    Wa[:, fit_end - af.ANCHOR_K:] = (tv * wy[None])[:, fit_end - af.ANCHOR_K:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    Ysq = np.sqrt(Y)
    MJ = np.maximum((Ysq[:, :fit_end] * Wa).sum(1), 1e-3)
    hz = min(fit_end + af.HORIZON, ne)
    NG = len(af.GRID)
    X = np.empty((NG, ne, 1 + nb)); X[:, :, 0] = 1.0
    for i in range(nb):
        X[:, :, 1 + i] = np.cos(TH[:, i][None, :] - af.GRID[:, None])
    Xt, Xa = X[:, :fit_end], X[:, fit_end:hz]
    XaS = np.einsum('gtp,gtq->gpq', Xa, Xa); XaB = np.einsum('gtp->gp', Xa)
    I = 1e-8 * np.eye(1 + nb)[None]
    R = np.zeros((Tn, NG)); C = np.zeros((Tn, NG, 1 + nb))
    for j in range(Tn):
        aw = af.LAM_HORIZON / (MJ[j] ** 2) / max(hz - fit_end, 1)
        A = np.einsum('gtp,t,gtq->gpq', Xt, W[j], Xt) + aw * XaS + I
        b = np.einsum('gtp,t->gp', Xt, W[j] * Ysq[j, :fit_end]) + aw * XaB * MJ[j]
        c = np.linalg.solve(A, b[..., None])[..., 0]
        C[j] = c
        R[j] = ((((np.einsum('gtp,gp->gt', Xt, c) - Ysq[j, :fit_end][None]) ** 2) @ W[j]
                 + aw * ((np.einsum('gtp,gp->gt', Xa, c) - MJ[j]) ** 2).sum(1))
                / (MJ[j] ** 2))               # SCALE-FREE: β must mean the same to every field
    return X, C, R


def q_of(C, PL):
    """Soft twelve-season assignment for every topic × candidate tuning, on the CLASSIFYING angle."""
    cls = af.GRID[None, :] + np.pi * (C[:, :, 1 + PL] < 0)          # (Tn, NG) pluto's canonical tuning
    z = np.cos(cls[:, :, None] - CENTRES[None, None, :]) / TAU
    z = z - z.max(2, keepdims=True)
    e = np.exp(z)
    return e / e.sum(2, keepdims=True)                               # (Tn, NG, 12)


def select(R, Q, beta, sweeps=SWEEPS):
    """Coordinate descent over the tunings. β=0 reduces EXACTLY to the deployed argmin."""
    Tn, NG, _ = Q.shape
    g = R.argmin(1)
    if beta <= 0:
        return g, 0
    H = -(Q * np.log(np.clip(Q, 1e-12, None))).sum(2)                # (Tn, NG) per-field entropy
    S = Q[np.arange(Tn), g].sum(0)                                   # (12,) unnormalised atlas marginal
    for it in range(sweeps):
        moved = 0
        for j in range(Tn):
            S_wo = S - Q[j, g[j]]
            qb = (S_wo[None, :] + Q[j]) / Tn                         # (NG,12) atlas if j chose each g
            Hb = -(qb * np.log(np.clip(qb, 1e-12, None))).sum(1)     # (NG,)
            J = R[j] + beta * (H[j] - Tn * Hb)
            gn = int(J.argmin())
            if gn != g[j]: moved += 1
            g[j] = gn; S = S_wo + Q[j, gn]
        if moved == 0:
            return g, it + 1                                          # fixed point reached
    return g, sweeps


def auc_of(Y, P, wall, n_end):
    tv = af.META["topic_valid"]
    tvw = tv[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    return float(np.mean([1.0 - ((Y[:, wall + h] - P[:, wall + h]) ** 2).sum() /
                          max(((Y[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(n_end - wall)]))


def run_wall(Y, TH, wall, betas):
    X, C, R = per_topic_grid(Y, TH, wall)
    PL = af.BODIES.index("pluto")
    Q = q_of(C, PL)
    out = {}
    for b in betas:
        g, its = select(R, Q, b)
        P = np.stack([np.maximum(X[g[j]] @ C[j, g[j]], 0.0) ** 2 for j in range(Y.shape[0])])
        cls = (np.degrees(af.GRID[g] + np.pi * (C[np.arange(len(g)), g, 1 + PL] < 0)) % 360)
        occ = np.bincount((np.round(cls, 6) % 360 // 30).astype(int), minlength=12)
        out[b] = {"auc": auc_of(Y, P, wall, wall + af.HORIZON), "occ": occ.tolist(),
                  "iters": its, "min_occ": int(occ.min()), "max_occ": int(occ.max())}
    return out


def main():
    names, Y, labels, future = af.load_lunar()
    TH, _ = af.sky_lunar(labels + future)
    n = Y.shape[1]
    walls = list(range(n - 63, n - 29, 3))
    print(f"═══ TWELVE-SIGN KL BALANCE IN THE SWEEP · τ={TAU} · {len(walls)} origins "
          f"({labels[walls[0]]}..{labels[walls[-1]]}) ═══", flush=True)
    print(f"    β chosen on the first {SELECT_N} origins ({labels[walls[0]]}..{labels[walls[SELECT_N-1]]}); "
          f"{labels[walls[SELECT_N]]}..{labels[walls[-1]]} held out", flush=True)
    res = {b: [] for b in BETAS}; occs = {}
    for w in walls:
        r = run_wall(Y, TH, w, BETAS)
        for b in BETAS: res[b].append(r[b]["auc"])
        occs[labels[w]] = {b: r[b] for b in BETAS}
        print(f"  {labels[w]}  " + "  ".join(f"β={b:<6g}{r[b]['auc']:+.4f}" for b in BETAS), flush=True)
    print(f"\n    {'β':>8s}{'select-mean':>13s}{'held-mean':>11s}{'all-mean':>10s}{'1996':>9s}"
          f"{'seasons min..max':>19s}", flush=True)
    base = np.array(res[0.0])
    for b in BETAS:
        a = np.array(res[b]); o = occs[labels[walls[-1]]][b]
        print(f"    {b:>8g}{a[:SELECT_N].mean():>+13.4f}{a[SELECT_N:].mean():>+11.4f}"
              f"{a.mean():>+10.4f}{a[-1]:>+9.4f}{o['min_occ']:>13d}..{o['max_occ']:<5d}", flush=True)
    pick = max(BETAS, key=lambda b: np.mean(res[b][:SELECT_N]))
    a = np.array(res[pick]); d = a - base
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if pick != 0.0 else 0.0
    print(f"\n  CHOSEN ON THE FIRST {SELECT_N} ORIGINS: β = {pick}", flush=True)
    if pick == 0.0:
        print("  → THE KL TERM DOES NOT HELP: the honest selection picks β=0, i.e. no term at all.", flush=True)
    else:
        print(f"  held-out origins: {a[SELECT_N:].mean():+.4f} vs β=0 {base[SELECT_N:].mean():+.4f} "
              f"→ {a[SELECT_N:].mean()-base[SELECT_N:].mean():+.4f}", flush=True)
        print(f"  headline 1996:    {a[-1]:+.4f} vs β=0 {base[-1]:+.4f} → {a[-1]-base[-1]:+.4f}", flush=True)
        print(f"  all twelve:       {a.mean():+.4f} vs {base.mean():+.4f} → {d.mean():+.4f} ± {se:.4f}, "
              f"wins {int((d>0).sum())}/12", flush=True)
    print(f"\n  season occupancy at the 1996 wall:", flush=True)
    for b in (0.0, pick if pick else BETAS[3]):
        o = occs[labels[walls[-1]]][b]["occ"]
        print(f"    β={b:<7g} " + " ".join(f"{v:2d}" for v in o) + f"   (min {min(o)}, max {max(o)})", flush=True)
    json.dump({"tau": TAU, "betas": BETAS, "origins": [labels[w] for w in walls],
               "auc": {str(b): [round(v, 4) for v in res[b]] for b in BETAS},
               "select_n": SELECT_N, "chosen_beta": pick,
               "occupancy_1996": {str(b): occs[labels[walls[-1]]][b]["occ"] for b in BETAS}},
              open("analysis/arxivtopics/kl_balance.json", "w"), indent=1)
    print("KLDONE", flush=True)


if __name__ == "__main__":
    main()
