#!/usr/bin/env python3
"""MINIMISE THE INFORMATION BETWEEN THE TWO DISTRIBUTIONS (operator 2026-08-04):

    min_U  Σ_t n_t · KL( q(·|t) ‖ p(·|t) )        p(j|t) = softmax_j( b_j + Σᵢ u_ij sinθᵢ + v_ij cosθᵢ )

KL(q‖p) = CE(q,p) − H(q), and H(q) is not the model's to change — so minimising the KL is exactly
maximum-likelihood for the multinomial with soft targets, and under the softmax link that problem is
CONVEX: one global optimum, reached deterministically from a zero start, no seed, no restarts. The
per-topic tuned waves w·sin(θ − p_j) live inside (u, v) exactly, as before.

Why this link and not the squared-amplitude pie: softmax logits built from sines are BOUNDED, so the
extrapolated distribution cannot drift to the degenerate corners that forced the amplitude models to
carry an anchor. The stabiliser is structural here, not bolted on.

Soft targets also dissolve the classifier's sample-starvation: every year constrains all 251
coordinates, so ridge becomes a small correction rather than a lifeline — λ=0 is in the grid, and the
grid is read on the nine early origins only.

Scored on the board this family already plays on: held-out KL(q‖p) per origin, against carry-forward
persistence (0.0831), the renormalised per-topic record (0.0853) and the Born multi-head (0.0880).

  python3 analysis/arxivtopics/kl_softmax.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
import arxiv_fit as af

names, Yv, labels_y, future = af.load_lunar()
TH, _ = af.sky_lunar(labels_y + future)
J, n = Yv.shape
Q = Yv / 100.0
Z_ALL = np.concatenate([np.ones((TH.shape[0], 1)), np.sin(TH), np.cos(TH)], 1)
D = Z_ALL.shape[1]
NW = np.clip(af.META["evidence"], 0, None) ** 0.75
GRID_L2 = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
WALLS = list(range(n - 63, n - 29, 3))
SEL = 9


def fit(wall, lam):
    Z = Z_ALL[:wall]; Qt = Q[:, :wall].T                     # (T, J) soft targets
    w = NW[:wall]; w = w / w.sum()
    def obj(theta):
        U = theta.reshape(J, D)
        L = Z @ U.T; L -= L.max(1, keepdims=True)
        E = np.exp(L); P = E / E.sum(1, keepdims=True)
        ce = -(w[:, None] * Qt * np.log(np.clip(P, 1e-300, None))).sum()
        G = ((P - Qt) * w[:, None]).T @ Z
        if lam > 0:
            ce += lam * (U[:, 1:] ** 2).sum(); G[:, 1:] += 2 * lam * U[:, 1:]
        return ce, G.ravel()
    r = minimize(obj, np.zeros(J * D), jac=True, method="L-BFGS-B",
                 options={"maxiter": 1000, "ftol": 1e-14})
    return r.x.reshape(J, D)


def pred(U):
    L = Z_ALL @ U.T; L -= L.max(1, keepdims=True)
    P = np.exp(L); P /= P.sum(1, keepdims=True)
    return P.T                                               # (J, ne)


def kl_window(P, lo, hi):
    w = NW[lo:hi]; w = w / w.sum()
    Hq = float(-(np.where(Q[:, lo:hi] > 0, Q[:, lo:hi] * np.log(np.clip(Q[:, lo:hi], 1e-12, None)), 0)).sum(0) @ w)
    pd = np.clip(P[:, lo:hi], 1e-12, None); pd = pd / pd.sum(0, keepdims=True)
    return float(-(Q[:, lo:hi] * np.log(pd)).sum(0) @ w) - Hq


def auc_at(Yh, wall):
    tv = af.META["topic_valid"]; tvw = tv[:, :wall].astype(float)
    mu = (Yv[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    return float(np.mean([1.0 - ((Yv[:, wall + h] - Yh[:, wall + h]) ** 2).sum() /
                max(((Yv[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(min(30, n - wall))]))


def main():
    print(f"═══ SOFTMAX KL MODEL · {J} topics · convex, global, seedless ═══", flush=True)
    t0 = time.time()
    per = {}
    for lam in GRID_L2:
        kls = []
        for w in WALLS:
            P = pred(fit(w, lam))
            kls.append(kl_window(P, w, min(w + af.HORIZON, n)))
        per[lam] = np.array(kls)
        print(f"  λ={lam:<6g} KL early(9) {per[lam][:SEL].mean():.4f} · held(3) {per[lam][SEL:].mean():.4f} · "
              f"per-origin " + " ".join(f"{v:.3f}" for v in per[lam]), flush=True)
    pick = min(GRID_L2, key=lambda l: per[l][:SEL].mean())
    k = per[pick]
    print(f"\n  CHOSEN ON THE NINE EARLY ORIGINS: λ = {pick}   [{time.time()-t0:.0f}s]", flush=True)
    aucs = []
    for w in WALLS:
        aucs.append(auc_at(100.0 * pred(fit(w, pick)), w))
    a = np.array(aucs)
    print(f"    {'model':32s} KL mean {k.mean():.4f} · held(3) {k[SEL:].mean():.4f} · 1996 {k[-1]:.4f}", flush=True)
    print(f"    {'persistence distribution':32s} KL mean 0.0831 · (board reference)", flush=True)
    print(f"    {'record renormalised':32s} KL mean 0.0853", flush=True)
    print(f"    {'Born multi-head (tuned)':32s} KL mean 0.0880", flush=True)
    print(f"    share-AUC for cross-reference: mean {a.mean():+.4f} · 1996 {a[-1]:+.4f} "
          f"(record +0.8751/+0.7990 · persistence +0.8511/+0.7344)", flush=True)
    U = fit(n - 30, pick)
    P = pred(U)
    print(f"    headline train KL {kl_window(P, 0, n-30):.4f} · held-out KL {kl_window(P, n-30, n):.4f}", flush=True)
    json.dump({"lambda": pick, "walls": [labels_y[w] for w in WALLS],
               "kl": [round(float(v), 4) for v in k], "kl_mean": round(float(k.mean()), 4),
               "auc": [round(float(v), 4) for v in a], "auc_mean": round(float(a.mean()), 4),
               "board": {"persistence": 0.0831, "record_renorm": 0.0853, "born_tuned": 0.0880}},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kl_softmax.json"), "w"), indent=1)
    print("KLSDONE", flush=True)


if __name__ == "__main__":
    main()
