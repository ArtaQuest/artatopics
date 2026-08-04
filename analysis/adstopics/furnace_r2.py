#!/usr/bin/env python3
"""FURNACE REGRESSION (operator 2026-07-20): forget classification — fit the furnace DIRECTLY to
the RAW trend and maximize R²:

    yhat(t) = sum_i w_i * exp(-wrap(theta_i(t) - p)^2) + b        (12 bodies)

Per topic: w by NON-NEGATIVE least squares (all w_i >= 0 — operator; solved by projected gradient,
batched over phases x topics), b free (absorbed by centering); the phase p comes from the 5-degree
grid (72 candidates). Reported two ways:
  in-sample — fit + evaluate + pick p on ALL 210 months (how much trend the family CAN explain);
  honest    — fit w,b on train (<162), pick p on validation [162,186), score R² on test [186,210).
Baselines with comparable dof: month-of-year climatology (12 dummies) and annual harmonics
(sin/cos of 1x and 2x per year + b). Out-of-sample R² = 1 - SS_res/SS_tot with SS_tot around the
fit-window mean (forecast skill convention; can go negative).

  python3 analysis/adstopics/furnace_r2.py
"""
import importlib.util as u, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
rc = _load("analysis/adstopics/ratechange.py", "rc")
sf = _load("analysis/adstopics/svm_furnace.py", "sf")


def r2(Y, Yh, cols, mu_cols):
    """Per-topic R² on cols; SS_tot around the mean over mu_cols (fit window)."""
    res = ((Y[:, cols] - Yh[:, cols]) ** 2).sum(1)
    mu = Y[:, mu_cols].mean(1, keepdims=True)
    tot = ((Y[:, cols] - mu) ** 2).sum(1) + 1e-9
    return 1.0 - res / tot


def ols(Y, X, fit, ev, mu_cols):
    """Per-topic OLS of Y on X (n,F): coefficients from fit cols, R² on ev cols."""
    C = Y[:, fit] @ np.linalg.pinv(X[fit]).T          # (Tn,F)
    return r2(Y, C @ X.T, ev, mu_cols)


def main():
    names, Y = rc.bal.load_all(); b = rc.BenchR(Y); Y = Y.astype(np.float64)
    Tn, n = Y.shape
    TH = sf.sky12(n)                                   # (n,12)
    P = 72; phases = np.deg2rad(np.arange(P) * 5.0)
    d = TH[:, None, :] - phases[None, :, None]
    G = np.exp(-(np.arctan2(np.sin(d), np.cos(d)) ** 2))               # (n,P,12)
    moy = b.moy; CL = np.eye(12)[moy]                                  # climatology dummies (n,12)
    ang = 2 * np.pi * np.arange(n) / 12.0
    HARM = np.column_stack([np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang), np.ones(n)])
    tt = (np.arange(n) - n / 2) / n
    POLY = np.column_stack([tt ** k for k in range(6)])                # generic smooth-trend control
    FAST = [0, 1, 2, 3, 4]                                             # sun..mars (periods <= 2y)
    SLOW = [5, 6, 7, 8, 9, 10, 11]                                     # jupiter..pluto, node, lilith

    def furnace(fit, sel, ev, mu_cols, bods=None, iters=500):
        """NNLS per (phase, topic): w >= 0 by projected gradient, b free via centering.
        Per-topic phase chosen on sel cols; R² reported on ev cols. bods = body subset."""
        Gs = G if bods is None else G[:, :, bods]
        nb = Gs.shape[2]
        Gf = Gs[fit]                                   # (nf,P,nb)
        Gm = Gf.mean(0)                                # (P,12)
        Gc = Gf - Gm[None]                             # centered features
        ym = Y[:, fit].mean(1, keepdims=True)          # (Tn,1)
        yc = Y[:, fit] - ym
        Gram = np.einsum("tpi,tpj->pij", Gc, Gc)       # (P,12,12)
        GTy = np.einsum("kt,tpi->pki", yc, Gc)         # (P,Tn,12)
        Lc = np.linalg.eigvalsh(Gram).max(-1) + 1e-9   # Lipschitz per phase
        W = np.zeros((P, Tn, nb))
        for _ in range(iters):                         # projected gradient: w <- relu(w - grad/L)
            W = np.maximum(0.0, W - (np.einsum("pki,pij->pkj", W, Gram) - GTy) / Lc[:, None, None])
        B = ym.T - np.einsum("pki,pi->pk", W, Gm)      # (P,Tn) bias from centering
        r2sel = np.zeros((P, Tn)); r2ev = np.zeros((P, Tn))
        for p in range(P):
            Yh = W[p] @ Gs[:, p].T + B[p][:, None]    # (Tn,n)
            r2sel[p] = r2(Y, Yh, sel, mu_cols); r2ev[p] = r2(Y, Yh, ev, mu_cols)
        kbest = r2sel.argmax(0)                        # per-topic phase (chosen on sel)
        per_topic = r2ev[kbest, np.arange(Tn)]
        kglob = int(r2sel.mean(1).argmax())            # ONE global phase (chosen on sel)
        return per_topic, r2ev[kglob], kglob * 5, kbest

    allc = np.arange(n); tr = np.arange(b.a); va = np.arange(b.a, b.b); te = np.arange(b.b, n)
    print(f"  {Tn} topics · raw trend (0-100) · 12-body furnace · NNLS (all w>=0) + free bias · phase grid 5° (72)", flush=True)

    def table(tag, fit, sel, ev, mu_cols, show_hist=False):
        print(f"\n  == {tag} ==", flush=True)
        ft, fg, kg, kbest = furnace(fit, sel, ev, mu_cols)
        for name, r in (("furnace 12 bodies (per-topic phase)", ft),
                        (f"furnace 12 bodies (global phase {kg}° {sf.SIGNS[kg // 30]})", fg),
                        ("furnace SUN only", furnace(fit, sel, ev, mu_cols, bods=[0])[0]),
                        ("furnace FAST 5 (sun..mars)", furnace(fit, sel, ev, mu_cols, bods=FAST)[0]),
                        ("furnace SLOW 7 (jupiter..lilith)", furnace(fit, sel, ev, mu_cols, bods=SLOW)[0]),
                        ("poly-5 trend control", ols(Y, POLY, fit, ev, mu_cols)),
                        ("climatology (12 dummies)", ols(Y, CL, fit, ev, mu_cols)),
                        ("harmonics (4 + b)", ols(Y, HARM, fit, ev, mu_cols))):
            print(f"    {name:44s} mean R² {r.mean():+.4f} · median {np.median(r):+.4f}", flush=True)
        if show_hist:
            hist = np.bincount(kbest * 5 // 30, minlength=12)
            print("    per-topic phase by sign: " + " ".join(f"{sf.SIGNS[i][:3]} {hist[i]}" for i in range(12)), flush=True)

    table("IN-SAMPLE (fit + phase + R² on all 210 months)", allc, allc, allc, allc, show_hist=True)
    table("HONEST (fit w,b < 162 · phase on val [162,186) · R² on test [186,210))", tr, va, te, tr)
    mean0 = r2(Y, np.repeat(Y[:, tr].mean(1, keepdims=True), n, 1), te, tr)
    print(f"    {'train-mean forecast':44s} mean R² {mean0.mean():+.4f} (0 by definition)", flush=True)


if __name__ == "__main__":
    main()
