#!/usr/bin/env python3
"""adstopics — the MECHANISTIC RESERVOIR model: heat flows from one reservoir to another, step by step.

MECHANISM (discrete monthly steps, all flows obeying Newton-law transfer between reservoirs):

    F[t]      = SUM_i w_i * sinc( f_i * wrap(x_i[t] - p) )        the celestial FORCING (the burner):
                                                                   12 bodies (synodic moon + 11), all
                                                                   weights positive, phase p from 12
                                                                   sign-centre initialisations
    R_1[t+1]  = R_1[t] + a_1 * ( F[t] - R_1[t] )                   reservoir 1 takes heat from the
                                                                   forcing at conductance a_1
    R_k[t+1]  = R_k[t] + a_k * ( R_{k-1}[t] - R_k[t] )             heat steps down the chain,
                                                                   reservoir to reservoir
    y_hat[t]  = R_K[t]                                             observed interest = the last
                                                                   reservoir's temperature

  - K reservoirs (arms: K = 1, 2, 3); conductances a_k in (0,1) via sigmoid — positive, stable
  - reservoir INITIAL STATES R_k[0] are trained parameters (positive) — the mechanism carries LEVEL
    as state, so the future is predicted by ROLLING THE SAME DYNAMICS FORWARD, never by re-anchoring
  - parameter count (K=2): 24 forcing weights + 1 phase + 2 conductances + 2 initial states = 29

PROTOCOL: identical judge to experiments.py — recency year excluded entirely; train 70% (simulate +
fit) | validation 15% (checkpoint/selection, reached by pure forward simulation) | TEST 15% (pure
forward simulation into untouched future months). Metric: future out-of-sample R².

  python3 analysis/adstopics/mechanistic.py [N_topics]
→ analysis/adstopics/mechanistic_results.csv
"""
import importlib.util as u, json, os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

tf = _load("analysis/trends_fit.py", "tf")
r5 = _load("analysis/topic500_reference_solution.py", "r5")
ex = _load("analysis/adstopics/experiments.py", "ex")      # reuse load_topics/split3 (same gate/protocol)

NBX = ex.NBX
FMAX = ex.FMAX
SIGN_CENTERS = ex.SIGN_CENTERS
STEPS = 1200
LR = 0.03
_D2R = float(np.pi / 180.0)


def fit_reservoir(Ys, X, K, dev, seed=7, assimilate=False, forcing=True):
    """Batched mechanistic fit. assimilate=True: while observations exist (train+val months), the
    LAST reservoir is nudged toward each observed y with a trained gain g in (0,1) — state
    estimation, mechanistically "the thermometer is read as time passes" — and beyond the forecast
    origin the SAME dynamics free-run into the untouched future. forcing=False switches the
    celestial furnace off (w = 0): the assimilation-only mechanistic null."""
    import torch
    DT = torch.float32
    T = len(Ys); n = len(Ys[0]); S = 12
    a, b = ex.split3(n)
    Yp = np.stack(Ys)
    mu = Yp[:, :a].mean(1); sd = np.maximum(Yp[:, :a].std(1), 1e-6)
    Ysn = (Yp - mu[:, None]) / sd[:, None] + 1.0            # shift so typical levels are positive
    Xt = torch.tensor(X, dtype=DT, device=dev)
    Yt = torch.tensor(Ysn, dtype=DT, device=dev)
    fitm = torch.zeros(n, dtype=DT, device=dev); fitm[:a] = 1.0 / a
    ckm = torch.zeros(n, dtype=DT, device=dev); ckm[a:b] = 1.0 / (b - a)

    p = torch.tensor(SIGN_CENTERS, dtype=DT, device=dev).repeat(T, 1).clone().requires_grad_(True)
    w = torch.full((T, S, NBX), 1.0 / NBX, dtype=DT, device=dev, requires_grad=True)
    logf = torch.full((T, S, NBX), float(np.log(1.0 / (FMAX - 1.0))), dtype=DT, device=dev, requires_grad=True)
    la = torch.zeros((T, S, K), dtype=DT, device=dev, requires_grad=True)      # sigmoid -> a_k in (0,1)
    r0 = torch.ones((T, S, K), dtype=DT, device=dev, requires_grad=True)       # initial states (>=0 clamp)
    lg = torch.zeros((T, S), dtype=DT, device=dev, requires_grad=True)         # assimilation gain (sigmoid)
    opt = torch.optim.Adam([p, w, logf, la, r0, lg], lr=LR)

    def simulate(P, W, LF, LA, R0, LG):
        d = torch.remainder(Xt[None, None, :, :] - P[:, :, None, None] + 180.0, 360.0) - 180.0
        F = torch.einsum("tsmk,tsk->tsm", torch.sinc(d * _D2R * (FMAX * torch.sigmoid(LF))[:, :, None, :]), W)
        if not forcing:
            F = F * 0.0
        A = torch.sigmoid(LA)                               # (T,S,K)
        G = torch.sigmoid(LG)                               # (T,S) assimilation gain
        R = [R0[:, :, k].clamp(min=0.0) for k in range(K)]
        outs = []
        for t in range(n):                                  # step-by-step heat transfer
            outs.append(R[K - 1])                           # the one-step-ahead PREDICTION (pre-assimilation)
            up = F[:, :, t]
            newR = []
            for k in range(K):
                Rk = R[k] + A[:, :, k] * (up - R[k])
                newR.append(Rk)
                up = R[k]                                   # downstream sees the PRE-update upstream temp
            if assimilate and t < b:                        # thermometer read while data exists (train+val)
                newR[K - 1] = newR[K - 1] + G * (Yt[:, t][:, None] - newR[K - 1])
            R = newR
        return torch.stack(outs, dim=2)                     # (T,S,n)

    best_l = torch.full((T, S), 1e18, device=dev)
    snap = None
    for step in range(STEPS):
        opt.zero_grad()
        pr = simulate(p, w, logf, la, r0, lg)
        loss = ((pr - Yt[:, None, :]) ** 2 * fitm[None, None, :]).sum(2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w.clamp_(min=0.0)
            if step % 20 == 19 or step == STEPS - 1:
                cur = simulate(p, w, logf, la, r0, lg).detach()
                cl = ((cur - Yt[:, None, :]) ** 2 * ckm[None, None, :]).sum(2)
                better = cl < best_l - 1e-7
                best_l = torch.where(better, cl, best_l)
                snap = cur if snap is None else torch.where(better[:, :, None], cur, snap)
    sel = best_l.argmin(dim=1)
    ar = torch.arange(T, device=dev)
    pred = (snap[ar, sel].cpu().numpy() - 1.0) * sd[:, None] + mu[:, None]
    val = np.array([1 - ((Yp[i, a:b] - pred[i, a:b]) ** 2).sum() /
                    max(((Yp[i, a:b] - Yp[i, a:b].mean()) ** 2).sum(), 1e-9) for i in range(T)])
    test = np.array([1 - ((Yp[i, b:] - pred[i, b:]) ** 2).sum() /
                     max(((Yp[i, b:] - Yp[i, b:].mean()) ** 2).sum(), 1e-9) for i in range(T)])
    return val, test


def main():
    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    names, Ys, X = ex.load_topics(n_topics)
    print(f"[mechanistic] {len(Ys)} topics · {len(Ys[0])} months (recency year excluded)")
    dev = r5._device()
    rows = []
    for K, ass, forc, name in ((1, False, True, "reservoir-K1"), (2, False, True, "reservoir-K2"),
                               (3, False, True, "reservoir-K3"),
                               (1, True, True, "assim-K1+forcing"), (2, True, True, "assim-K2+forcing"),
                               (1, True, False, "assim-K1 NULL (no forcing)"),
                               (2, True, False, "assim-K2 NULL (no forcing)")):
        val, test = fit_reservoir(Ys, X, K, dev, assimilate=ass, forcing=forc)
        rows.append(dict(arm=name, median_val=float(np.median(val)), median_test=float(np.median(test)),
                         clamped_mean_test=float(np.clip(test, 0, 1).mean() * 100),
                         frac_test_pos=float((test > 0).mean())))
        print(f"  {name:26s} val_med {np.median(val):7.3f} · TEST med {np.median(test):7.3f} · "
              f"clamped {np.clip(test,0,1).mean()*100:5.1f} · >0: {(test>0).mean()*100:4.1f}%", flush=True)
    pd.DataFrame(rows).to_csv("analysis/adstopics/mechanistic_results.csv", index=False)

if __name__ == "__main__":
    main()
