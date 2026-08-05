#!/usr/bin/env python3
"""CLASSIFICATION, NOT REGRESSION (operator 2026-08-04): predict THE MOST TRENDING TOPIC per date.

    label(t) = argmax_j [ q_j(t) − q_j(t−1) ]      the topic whose share of the year's citations rose most
    p(j | t) = softmax_j( b_j + Σᵢ u_ij sinθᵢ(t) + v_ij cosθᵢ(t) )        trained by cross-entropy

MODEL CLASS, and why this exact form. Multinomial logistic regression on the sky is the one member of
the cross-entropy classification family whose training problem is CONVEX — the optimum is global and
unique (given ridge), found deterministically from a zero start, no seed anywhere. And nothing is
given up to get that: each class's per-body tuned wave w·sin(θ−p) is exactly u·sinθ + v·cosθ, so the
per-topic phases of the earlier models live inside this parameterisation and can be read back out of
(u, v) after the fit.

REGULARISATION IS NOT OPTIONAL HERE, and saying so beats hiding it: 251 classes × 15 features = 3,765
parameters against ~296 training dates — unridged, the classes separate and the weights diverge. The
strength is chosen the only honest way this campaign allows: on the NINE EARLY origins (1963..1987),
then applied unchanged to 1990/1993/1996.

THE BASELINES that make the score mean something: predict the TRAINING MODE (the topic that won most
often), and PERSISTENCE — predict whichever topic won the last training year, every year. Chance is
1/251 ≈ 0.4%.

  python3 analysis/arxivtopics/trend_classifier.py
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
LAB = np.diff(Q, axis=1).argmax(0)                       # label for year index t+1, t = 0..n-2
Z_ALL = np.concatenate([np.ones((TH.shape[0], 1)), np.sin(TH), np.cos(TH)], 1)
D = Z_ALL.shape[1]
NW = np.clip(af.META["evidence"], 0, None) ** 0.75
GRID_L2 = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
WALLS = list(range(n - 63, n - 29, 3))
SEL = 9


def fit(wall, lam):
    """Convex softmax CE + ridge (intercept unpenalised), deterministic L-BFGS from zero."""
    ti = np.arange(1, wall)                              # years with a label inside the train era
    Z = Z_ALL[ti]; y = LAB[ti - 1]
    w = NW[ti]; w = w / w.sum()
    Y1 = np.zeros((len(ti), J)); Y1[np.arange(len(ti)), y] = 1.0

    def obj(theta):
        U = theta.reshape(J, D)
        L = Z @ U.T
        L -= L.max(1, keepdims=True)
        E = np.exp(L); P = E / E.sum(1, keepdims=True)
        ce = -(w * np.log(np.clip(P[np.arange(len(ti)), y], 1e-12, None))).sum()
        pen = lam * (U[:, 1:] ** 2).sum()
        G = ((P - Y1) * w[:, None]).T @ Z
        G[:, 1:] += 2 * lam * U[:, 1:]
        return ce + pen, G.ravel()

    r = minimize(obj, np.zeros(J * D), jac=True, method="L-BFGS-B",
                 options={"maxiter": 500, "ftol": 1e-12})
    return r.x.reshape(J, D)


def score(U, wall):
    lo, hi = wall, min(wall + af.HORIZON, n)
    ti = np.arange(lo, hi)                               # label years in the window
    Z = Z_ALL[ti]; y = LAB[ti - 1]
    L = Z @ U.T; L -= L.max(1, keepdims=True)
    P = np.exp(L); P /= P.sum(1, keepdims=True)
    order = np.argsort(-P, 1)
    top1 = float((order[:, 0] == y).mean())
    top3 = float((y[:, None] == order[:, :3]).any(1).mean())
    ce = float(-np.log(np.clip(P[np.arange(len(ti)), y], 1e-12, None)).mean())
    tr = LAB[np.arange(1, wall) - 1]
    mode = np.bincount(tr, minlength=J).argmax()
    b_mode = float((y == mode).mean())
    b_pers = float((y == LAB[wall - 2]).mean())          # last training winner, carried forward
    return {"top1": top1, "top3": top3, "ce": ce, "mode": b_mode, "pers": b_pers,
            "pred": order[:, 0], "true": y}


def main():
    print(f"═══ MOST-TRENDING-TOPIC CLASSIFIER · {J} classes · convex softmax CE ═══", flush=True)
    t0 = time.time()
    per_lam = {}
    for lam in GRID_L2:
        rows = [score(fit(w, lam), w) for w in WALLS]
        per_lam[lam] = rows
        a = np.array([r["top1"] for r in rows])
        print(f"  λ={lam:<6g} top-1 early(9) {a[:SEL].mean():.3f} · held(3) {a[SEL:].mean():.3f} · "
              f"per-origin " + " ".join(f"{v:.2f}" for v in a), flush=True)
    pick = max(GRID_L2, key=lambda l: np.mean([r["top1"] for r in per_lam[l][:SEL]]))
    rows = per_lam[pick]
    a1 = np.array([r["top1"] for r in rows]); a3 = np.array([r["top3"] for r in rows])
    bm = np.array([r["mode"] for r in rows]); bp = np.array([r["pers"] for r in rows])
    ce = np.array([r["ce"] for r in rows])
    print(f"\n  CHOSEN ON THE NINE EARLY ORIGINS: λ = {pick}   [{time.time()-t0:.0f}s total]", flush=True)
    print(f"    {'':22s}{'early(9)':>10s}{'held(3)':>9s}{'all(12)':>9s}", flush=True)
    for lab, v in (("model top-1", a1), ("model top-3", a3),
                   ("baseline: train mode", bm), ("baseline: persistence", bp)):
        print(f"    {lab:22s}{v[:SEL].mean():>10.3f}{v[SEL:].mean():>9.3f}{v.mean():>9.3f}", flush=True)
    print(f"    model held-out CE {ce.mean():.3f} · chance CE {np.log(J):.3f} (uniform) · chance top-1 {1/J:.4f}", flush=True)

    U = fit(n - 30, pick)
    s = score(U, n - 30)
    print(f"\n  THE HEADLINE WINDOW (fit ≤1995 → predict the winner 1996..2025):", flush=True)
    print(f"    top-1 {s['top1']:.3f} · top-3 {s['top3']:.3f} · mode {s['mode']:.3f} · persistence {s['pers']:.3f}", flush=True)
    hits = 0
    for k, t in enumerate(range(n - 30, n)):
        ok = s["pred"][k] == s["true"][k]; hits += ok
        print(f"    {labels_y[t]}  pred: {names[s['pred'][k]][:34]:36s} true: {names[s['true'][k]][:34]:36s} {'✓' if ok else ''}", flush=True)
    json.dump({"lambda": pick, "walls": [labels_y[w] for w in WALLS],
               "top1": [round(float(v), 4) for v in a1], "top3": [round(float(v), 4) for v in a3],
               "baseline_mode": [round(float(v), 4) for v in bm],
               "baseline_persistence": [round(float(v), 4) for v in bp],
               "ce": [round(float(v), 4) for v in ce],
               "headline": {"top1": round(s["top1"], 4), "top3": round(s["top3"], 4),
                             "mode": round(s["mode"], 4), "pers": round(s["pers"], 4)}},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trend_classifier.json"), "w"), indent=1)
    print("CLSDONE", flush=True)


if __name__ == "__main__":
    main()
