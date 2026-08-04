#!/usr/bin/env python3
"""SIMPLE FURNACE, linear-model heads (operator 2026-07-20).

  score(t) = sum_i w_i * exp(-wrap(theta_i(t) - p)^2) + b        (12 bodies: Sun..Pluto, Node, Lilith)

The 12 weights + bias are trained as either
  svm      — linear SVM (squared hinge + L2) on the score, or
  logistic — logistic regression: sigmoid(score), binary cross-entropy + L2;
the single GLOBAL phase p is tuned by grid search in 5-degree steps (72 candidates), chosen on
validation. Balanced-at-all-times label. Two variants per loss, same phase grid:
  global    — ONE (w, b) for the whole population (prediction depends only on the month).
  per-topic — each topic its own (w, b), all sharing the one global phase p.

  python3 analysis/adstopics/svm_furnace.py [svm|logistic] [lag]    (lag 12 = ANNUAL growth task)
"""
import sys
import importlib.util as u, os
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
rc = _load("analysis/adstopics/ratechange.py", "rc")
cl = _load("analysis/adstopics/astro_classifier.py", "cl")

BODIES = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "node", "lilith"]
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]


def sky12(n):
    E = pd.read_csv("analysis/_ephemeris_extended.csv")
    grid = pd.DatetimeIndex(pd.to_datetime(E["Time"]))
    i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
    TH = np.stack([np.deg2rad(E[f"{b}_lon"].to_numpy(float)[i0:i0 + n]) for b in BODIES], 1)  # (n,12)
    assert not np.isnan(TH).any()
    return TH


def main():
    import torch as T
    dev = "mps" if T.backends.mps.is_available() else "cpu"
    lag = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    names, Y = rc.bal.load_all(); b = rc.BenchR(Y)
    L, _ = cl.prep(b, lag)                                # balanced-at-all-times label (Tn,n) in {0,1}
    Tn, n = L.shape; TH = sky12(n)
    P = 72; phases = np.deg2rad(np.arange(P) * 5.0)
    # Gaussian features per phase: G[p,t,i] = exp(-wrap(theta_i(t)-p)^2)
    d = TH[None] - phases[:, None, None]
    G = np.exp(-(np.arctan2(np.sin(d), np.cos(d)) ** 2)).astype(np.float32)      # (P,n,12)
    y = T.tensor(L * 2.0 - 1.0, device=dev)               # (Tn,n) in {-1,+1}
    Gt = T.tensor(G, device=dev)
    tr = slice(lag, b.a); va = slice(b.a, b.b); te = slice(b.b, n)
    ytr = y[:, tr]; Ltr = T.tensor(L[:, tr], device=dev); lam = 1e-3

    def fit(per_topic, loss="svm", steps=400, lr=0.05, chunk=24):
        accv = np.zeros(P); acct = np.zeros(P); Wout = np.zeros((P, 12))
        for c0 in range(0, P, chunk):
            pc = slice(c0, min(c0 + chunk, P)); Pc = pc.stop - pc.start
            Gc = Gt[pc]                                   # (Pc,n,12)
            shape = (Pc, Tn, 12) if per_topic else (Pc, 1, 12)
            W = T.zeros(shape, device=dev, requires_grad=True)
            B = T.zeros(shape[:2], device=dev, requires_grad=True)
            opt = T.optim.Adam([W, B], lr=lr)
            for _ in range(steps):
                opt.zero_grad()
                logit = T.einsum("pki,pti->pkt", W, Gc[:, tr]) + B[:, :, None]    # (Pc,K,ntr)
                if loss == "svm":
                    data = T.relu(1 - ytr[None] * logit).pow(2).mean((1, 2))      # squared hinge (SVM)
                else:                                                              # logistic regression:
                    tgt = Ltr[None].expand(logit.shape[0], -1, -1)                 # (Pc,Tn,ntr)
                    data = T.nn.functional.binary_cross_entropy_with_logits(       # sigmoid(score) + BCE
                        logit.expand_as(tgt), tgt, reduction="none").mean((1, 2))
                (data + lam * W.pow(2).sum(-1).mean(1)).sum().backward(); opt.step()
            with T.no_grad():
                logit = T.einsum("pki,pti->pkt", W, Gc) + B[:, :, None]           # (Pc,K,n) -> broadcast
                pred = (logit > 0).float()
                if not per_topic: pred = pred.expand(-1, Tn, -1)
                Lt = T.tensor(L, device=dev)[None]
                accv[pc] = (pred[:, :, va] == Lt[:, :, va]).float().mean((1, 2)).cpu().numpy()
                acct[pc] = (pred[:, :, te] == Lt[:, :, te]).float().mean((1, 2)).cpu().numpy()
                Wout[pc] = W.mean(1).detach().cpu().numpy()
        k = int(accv.argmax()); deg = k * 5
        tag = f"{loss:8s} {'per-topic' if per_topic else 'global   '}"
        top = sorted(zip(BODIES, Wout[k]), key=lambda x: -abs(x[1]))[:4]
        print(f"  [{tag}] phase* {deg:3d}° ({SIGNS[deg // 30]}) · val {accv[k]:.4f} · TEST {acct[k]:.4f} (coin 0.5)"
              f" · top |w|: {', '.join(f'{b} {w:+.2f}' for b, w in top)}", flush=True)
        return deg, accv[k], acct[k]

    loss = (sys.argv[1] if len(sys.argv) > 1 else "logistic")
    up = L[:, te].mean(); base = max(up, 1 - up)
    print(f"  lag={lag} balanced label · test-up {up*100:.1f}% (majority base {base:.4f}) · {Tn} topics · "
          f"phase grid 5° (72) · loss {loss} + L2 · dev {dev}", flush=True)
    fit(per_topic=False, loss=loss)
    fit(per_topic=True, loss=loss)


if __name__ == "__main__":
    main()
