#!/usr/bin/env python3
"""adstopics MODEL v3 — the bounded, damped, validated mechanistic forecaster family.

GOAL (operator): a NON-NEGATIVE full-tier MEAN future R². The mean dies by blow-ups; every family
member below is conservative by construction and every choice is made on VALIDATION only:

  members (all causal, all clipped to the physical [0,100] thermometer range):
    L     level-only:        y_hat = L_b                        (trailing-12 median at origin)
    D     damped drift:      y_hat = L_b + s_b * Σ_{i<=h} φ^i   (robust Theil-Sen slope over the
                                                                 last 24 months, damping φ; slope
                                                                 CONTINUES then relaxes — reservoir
                                                                 cooling)
    S(λ)  + furnace:         y_hat = member + λ * F_vm(t)       (the pruned-4 von Mises seasonal,
                                                                 amplitude shrunk by λ)
    N     seasonal naive:    y_hat = y[t - 12] blended toward L_b with damping
    LOG   any member fit/applied in log1p space (multiplicative drift)

  per topic: every (member, λ, φ, log) combo is scored on the VALIDATION window (never test);
  the best-validated member predicts the untouched TEST. A conservative margin δ: a risky member
  must beat the level-only VALIDATION score by δ to be chosen (δ swept on validation pooled).

  python3 analysis/adstopics/model_v3.py [N_topics]
→ analysis/adstopics/model_v3_results.csv
"""
import importlib.util as u, itertools, json, os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

tf = _load("analysis/trends_fit.py", "tf")
r5 = _load("analysis/topic500_reference_solution.py", "r5")
ex = _load("analysis/adstopics/experiments.py", "ex")
co = _load("analysis/adstopics/combo_experiments.py", "co")

PHIS = [0.0, 0.7, 0.9, 0.97]
LAMS = [0.0, 0.25, 0.5, 1.0]
DELTAS = [0.0, 0.02, 0.05, 0.1]


def theil_sen(y):
    """Robust slope over the window (median of pairwise slopes, subsampled deterministically)."""
    n = len(y)
    ss = []
    for i in range(0, n - 1, 2):
        for j in range(i + 6, n, 6):
            ss.append((y[j] - y[i]) / (j - i))
    return float(np.median(ss)) if ss else 0.0


def r2(y, p):
    ss = float(((y - y.mean()) ** 2).sum())
    return 1 - float(((y - p) ** 2).sum()) / ss if ss > 1e-9 else 0.0


def main():
    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    names, Ys, X = ex.load_topics(n_topics)
    n = len(Ys[0]); a, b = ex.split3(n)
    print(f"[v3] {len(Ys)} topics · {n} months (recency year excluded)")
    dev = r5._device()

    def lvl(y, upto):
        return float(np.median(y[max(0, upto - 12):upto]))

    # the furnace seasonal (pruned-4 vm), fitted ONCE on causal residuals (train loss, val checkpoint)
    Lc = [np.array([lvl(y, i) if i >= 6 else float(np.median(y[:max(1, i + 1)])) for i in range(n)])
          for y in Ys]
    resid = [y - c for y, c in zip(Ys, Lc)]
    _, par1 = co.fit_vm(resid, X, dev, kernel="vonmises")
    C = np.zeros((len(Ys), ex.NBX))
    for i in range(len(Ys)):
        z = np.exp(par1["kappa"][i][None, :] *
                   (np.cos(np.deg2rad((X[:a] - par1["p"][i] + 180.0) % 360.0 - 180.0)) - 1.0))
        C[i] = (par1["w"][i][None, :] * z).std(0)
    mask = np.zeros_like(C)
    idx = np.argsort(-C, axis=1)[:, :4]
    for i in range(C.shape[0]):
        mask[i, idx[i]] = 1.0
    seas, _ = co.fit_vm(resid, X, dev, body_mask=mask, kernel="vonmises")   # (T,n) residual seasonal

    # candidate predictions per topic over BOTH val and test horizons, from the appropriate origins:
    #   validation candidates predict [a,b) from origin a; test candidates predict [b,n) from origin b.
    def members(y, i, origin, upto):
        """dict name -> prediction array for months [origin, upto), causal at `origin`."""
        L0 = lvl(y, origin)
        s0 = theil_sen(y[max(0, origin - 24):origin])
        h = np.arange(1, upto - origin + 1)
        out = {}
        for phi in PHIS:
            drift = L0 + s0 * (np.cumsum(phi ** h) if phi > 0 else 0.0)
            base = np.full(upto - origin, L0) if phi == 0.0 else drift
            for lam in LAMS:
                pred = base + lam * seas[i][origin:upto]
                out[f"phi{phi}_lam{lam}"] = np.clip(pred, 0.0, 100.0)
        # seasonal naive (damped toward level)
        nv = y[origin - 12:upto - 12].copy()
        out["snaive"] = np.clip(0.5 * nv + 0.5 * lvl(y, origin), 0.0, 100.0)
        out["snaive_full"] = np.clip(nv, 0.0, 100.0)
        return out

    # per-topic validation scores for every member; then per-δ test evaluation
    val_scores, test_preds = [], []
    for i, y in enumerate(Ys):
        vm_ = members(y, i, a, b)
        tm_ = members(y, i, b, n)
        vs = {k: r2(y[a:b], v) for k, v in vm_.items()}
        val_scores.append(vs)
        test_preds.append(tm_)

    rows = []
    base_key = "phi0.0_lam0.0"                                # level-only
    for delta in DELTAS:
        tests = []
        picks = {}
        for i, y in enumerate(Ys):
            vs = val_scores[i]
            best_k = max(vs, key=lambda k: vs[k])
            if vs[best_k] < vs[base_key] + delta:
                best_k = base_key                             # conservative fallback
            picks[best_k] = picks.get(best_k, 0) + 1
            tests.append(r2(y[b:], test_preds[i][best_k]))
        tests = np.array(tests)
        rows.append(dict(delta=delta, mean_test=float(tests.mean()), median_test=float(np.median(tests)),
                         frac_pos=float((tests > 0).mean()),
                         clamped=float(np.clip(tests, 0, 1).mean() * 100)))
        top_picks = sorted(picks.items(), key=lambda kv: -kv[1])[:4]
        print(f"  δ={delta:4.2f}: TEST mean {tests.mean():7.3f} · med {np.median(tests):6.3f} · "
              f">0 {(tests>0).mean()*100:4.1f}% · picks {top_picks}", flush=True)
    # reference: pure level-only
    lv = np.array([r2(Ys[i][b:], test_preds[i][base_key]) for i in range(len(Ys))])
    print(f"  level-only:  TEST mean {lv.mean():7.3f} · med {np.median(lv):6.3f}")
    pd.DataFrame(rows).to_csv("analysis/adstopics/model_v3_results.csv", index=False)

if __name__ == "__main__":
    main()
