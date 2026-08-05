#!/usr/bin/env python3
"""TRAIN THE KL ITSELF on y_j = (b_j + Σᵢ w_ij·sin θᵢ)² (operator 2026-08-04).

    p_j(t) = y_j(t) / Σ_k y_k(t)
    min_U  Σ_t n_t · KL( q(·|t) ‖ p(·|t) )  +  Σ_j aw_j Σ_{t∈horizon} ( a_j(t) − m_j )²

The second term is the horizon anchor INSIDE the objective this time. The earlier polish measurement
optimised bare cross-entropy from the anchored analytic start and drifted — an unfair comparison,
because it optimised a different objective than the one the analytic solve had answered. Here the KL
is trained with the anchor as part of the loss, so "analytic amplitude solve" and "true KL training"
finally answer the same question and the comparison means something.

The landscape is sign-symmetric (negating any head leaves p unchanged), so no unique global optimum
exists; optimisation is deterministic L-BFGS from the analytic amplitude solution — the canonical
branch — and from zero as a control. No seed anywhere.

  python3 analysis/arxivtopics/train_kl_born.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
import arxiv_fit as af
import multihead_ce as MH

names, Yv, labels_y = MH.names, MH.Yv, MH.labels
TH = MH.TH
J, n, ne = MH.J, MH.n, MH.ne
Q, NW = MH.Q, MH.NW
WALLS = list(range(n - 63, n - 29, 3))


def train_kl(wall, U0, kind="sin", anchored=True):
    Z = MH.features(kind)
    Zt = Z[:wall]
    w = NW[:wall] / NW[:wall].sum()
    Qt = Q[:, :wall]
    hz = min(wall + af.HORIZON, ne)
    Za = Z[wall:hz]
    tail = np.sqrt(Qt[:, max(0, wall - af.ANCHOR_K):])
    m = np.maximum(np.sqrt(Q[:, max(0, wall - af.ANCHOR_K):wall]).mean(1), 1e-4)
    aw = af.LAM_HORIZON / (m ** 2) / max(hz - wall, 1) if anchored else np.zeros(J)

    def obj(theta):
        U = theta.reshape(J, -1)
        A = U @ Zt.T                                         # (J, T)
        A2 = A ** 2 + 1e-12
        R = A2.sum(0, keepdims=True)
        P = A2 / R
        kl = -(Qt * np.log(P / np.clip(Qt, 1e-12, None))).sum(0) @ w
        gA = -2.0 * w[None, :] * (Qt / np.where(np.abs(A) > 1e-9, A, 1e-9) - A * Qt.sum(0, keepdims=True) / R)
        G = gA @ Zt
        Aa = U @ Za.T
        kl += float((aw[:, None] * (Aa - m[:, None]) ** 2).sum())
        G += (2.0 * aw[:, None] * (Aa - m[:, None])) @ Za
        return float(kl), G.ravel()

    r = minimize(obj, U0.ravel(), jac=True, method="L-BFGS-B",
                 options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-12})
    return r.x.reshape(J, -1)


def main():
    print(f"═══ TRUE KL TRAINING · phase-less sin vs free per-body phases (≡ sincos) ═══", flush=True)
    rows, aucs = {}, {}
    t0 = time.time()
    for kind, tag in (("sin", "no phases"), ("sincos", "free p_ji")):
        Z = MH.features(kind)
        for mode in ("analytic", "kl_trained") + (("kl_from_zero",) if kind == "sin" else ()):
            key = f"{tag} · {mode}"
            rows[key], aucs[key] = [], []
            for w in WALLS:
                U0 = MH.solve(Z, w, True)
                if mode == "analytic":
                    U = U0
                elif mode == "kl_trained":
                    U = train_kl(w, U0, kind, True)
                else:
                    U = train_kl(w, np.zeros_like(U0) + 1e-3, kind, True)
                P, _ = MH.predict(U, Z)
                rows[key].append(MH.kl_at(P, w)); aucs[key].append(MH.auc_at(P, w))
    print(f"  [{time.time()-t0:.0f}s]  held-out (TEST) KL per origin, lower better:", flush=True)
    for k in rows:
        r = np.array(rows[k]); a = np.array(aucs[k])
        print(f"    {k:26s} KL mean {r.mean():.4f} · 1996 {r[-1]:.4f} · share-AUC {a.mean():+.4f}"
              f"   per-origin " + " ".join(f"{v:.3f}" for v in r), flush=True)
    print(f"\n    board: persistence 0.0831 · record renormalised 0.0853 · Born tuned-head 0.0880 · softmax-KL 0.1104", flush=True)
    json.dump({k: {"kl": [round(float(v), 4) for v in rows[k]],
                   "kl_mean": round(float(np.mean(rows[k])), 4),
                   "auc_mean": round(float(np.mean(aucs[k])), 4)} for k in rows},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_kl_born.json"), "w"), indent=1)
    print("TKBDONE", flush=True)


if __name__ == "__main__":
    main()
