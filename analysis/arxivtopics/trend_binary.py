#!/usr/bin/env python3
"""PER-TOPIC BINARY TRENDING CLASSIFIERS, BALANCED BY CONSTRUCTION (operator 2026-08-04).

    For each topic j, on its own history alone (fully decoupled model and data):
      Δ_j(t)   = works_j(t+1) − works_j(t)          works = PUBLICATION COUNT, the rail's works matrix
      τ_j      = median_t Δ_j(t)                     the topic's own median rise
      label(t) = 1 if Δ_j(t) > τ_j                   "trending next year" — balanced ~50/50 by definition
      features = [1, sin θᵢ(t), cos θᵢ(t)]           the sky at year t, nothing else
      model    = logistic regression (binary cross-entropy, convex, deterministic)

    Years: from the topic's continuously-non-zero start (on works) through the last full year.
    Split: SHUFFLED 90/10 of the topic's data points (RandomState(0)), as specified. A shuffled
    split on an autocorrelated series is a GENEROUS test — adjacent years land on both sides — so
    the temporal split (last 10% of years held out) is reported beside it for contrast, clearly
    labelled. The median is computed on the topic's whole Δ series, which is what makes the set
    balanced; that convention is part of the spec and is stated rather than hidden.

  python3 analysis/arxivtopics/trend_binary.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.optimize import minimize
import arxiv_fit as af

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
names, Yv, labels_y, future = af.load_lunar()
TH, _ = af.sky_lunar(labels_y + future)
n = Yv.shape[1]

# ── publication counts, aligned to the same 251 topics ────────────────────────────────────────────
_w = pd.read_csv(os.path.join(REPO, "analysis/citations/rail_works_yearly.csv"))
_c = pd.read_csv(os.path.join(REPO, "analysis/citations/citations_received_yearly.csv"))
key = "subfield_id" if "subfield_id" in _w.columns and "subfield_id" in _c.columns else None
assert key, "rail files need subfield_id to align"
_w = _w.set_index(key).loc[_c[key]].reset_index()
ycols = [c for c in _w.columns if c[:1].isdigit()][:n]      # same 1700..2025 window as the shares
W = _w[ycols].to_numpy(float)
J = W.shape[0]
Z_ALL = np.concatenate([np.ones((TH.shape[0], 1)), np.sin(TH), np.cos(TH)], 1)
D = Z_ALL.shape[1]
RNG = np.random.RandomState(0)                              # the one RNG: the specified shuffled split


def logistic(Zt, y, lam=1e-6):
    def obj(th):
        s = Zt @ th
        p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
        ce = -np.mean(y * np.log(np.clip(p, 1e-12, None)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, None)))
        g = Zt.T @ (p - y) / len(y)
        return ce + lam * (th[1:] ** 2).sum(), g + 2 * lam * np.r_[0, th[1:]]
    return minimize(obj, np.zeros(Zt.shape[1]), jac=True, method="L-BFGS-B",
                    options={"maxiter": 500}).x


def run_topic(j):
    w = W[j]
    nz = np.ones(n, bool)
    for i in range(n - 2, -1, -1): nz[i] = (w[i] > 0) & nz[i + 1]
    t0 = int(nz.argmax())
    ts = np.arange(t0, n - 1)                                # label needs t+1
    if len(ts) < 30: return None
    d = w[ts + 1] - w[ts]
    tau = np.median(d)
    y = (d > tau).astype(float)
    Z = Z_ALL[ts]
    idx = RNG.permutation(len(ts))
    k = max(1, int(round(0.1 * len(ts))))
    te, tr = idx[:k], idx[k:]
    th = logistic(Z[tr], y[tr])
    acc = lambda I: float(((Z[I] @ th > 0).astype(float) == y[I]).mean())
    # temporal contrast: same machinery, last 10% of YEARS held out
    cut = len(ts) - k
    th_t = logistic(Z[:cut], y[:cut])
    acc_t = float(((Z[cut:] @ th_t > 0).astype(float) == y[cut:]).mean())
    # today's call: the sky at the last year → trending next year?
    p_now = float(1.0 / (1.0 + np.exp(-(Z_ALL[n - 1] @ th))))
    return {"train": acc(tr), "test": acc(te), "test_temporal": acc_t,
            "balance": float(y.mean()), "n": len(ts), "p_now": p_now, "start": labels_y[t0]}


def main():
    t0 = time.time()
    res = {}
    for j in range(J):
        r = run_topic(j)
        if r: res[names[j]] = r
    tr = np.array([r["train"] for r in res.values()])
    te = np.array([r["test"] for r in res.values()])
    tt = np.array([r["test_temporal"] for r in res.values()])
    ba = np.array([r["balance"] for r in res.values()])
    nn = np.array([r["n"] for r in res.values()])
    pooled = float(np.average(te, weights=np.maximum(np.round(0.1 * nn), 1)))
    print(f"═══ {len(res)} per-topic binary trending classifiers · [{time.time()-t0:.0f}s] ═══", flush=True)
    print(f"  label balance: mean {ba.mean():.3f} (median-threshold construction; ties on integer counts pull it under .5)", flush=True)
    print(f"  data points per topic: median {int(np.median(nn))} (min {nn.min()}, max {nn.max()})", flush=True)
    print(f"\n  {'':24s}{'mean':>8s}{'median':>8s}{'>50%':>8s}", flush=True)
    for lab, v in (("TRAIN accuracy", tr), ("TEST accuracy (shuffled)", te), ("test (temporal contrast)", tt)):
        print(f"  {lab:24s}{v.mean():>8.3f}{np.median(v):>8.3f}{(v > 0.5).mean()*100:>7.0f}%", flush=True)
    print(f"  pooled shuffled-test accuracy (size-weighted): {pooled:.3f} · chance 0.500", flush=True)
    json.dump({"topics": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items()}
                          for k, r in res.items()},
               "summary": {"train_mean": round(float(tr.mean()), 4), "test_mean": round(float(te.mean()), 4),
                            "test_median": round(float(np.median(te)), 4),
                            "test_temporal_mean": round(float(tt.mean()), 4),
                            "pooled_test": round(pooled, 4), "balance_mean": round(float(ba.mean()), 4),
                            "n_topics": len(res)}},
              open(os.path.join(HERE, "trend_binary.json"), "w"), indent=1)
    print("TBDONE", flush=True)


if __name__ == "__main__":
    main()
