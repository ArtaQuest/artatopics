#!/usr/bin/env python3
"""SKY -> ALL-TOPIC SCORES forward model, attention form (operator 2026-07-20, final spec):

    y_j(t) = sum_i  a_ij * exp( -wrap(theta_i(t) - p_j)^2 )  +  b_j

  p_j, b_j : free per-topic PARAMETERS (phase via a 2-vector -> atan2, no wrap issues)
  a_ij     : ATTENTION SCORES — topic query q_j attends over body keys k_i:
             a_ij = softmax_i( q_j . k_i / sqrt(d) )   (positive, sums to 1 over the 12 bodies)
  s        : one global amplitude (softplus, raw 0-100 units)  ->  yhat = s * Sum + b_j

Given ONLY the astro inputs (the 12 body angles theta(t), known for any date), the model emits the
search score of ALL 3,021 topics at that time TOGETHER. Trained end-to-end by MSE on the raw
series; scored by R² per topic (temporal) and per month (cross-sectional, vs the bias-only
baseline). Honest walls: parameters fit on train (<162), early stop on val [162,186), R² reported
on test [186,210) — pure ephemeris extrapolation.

  python3 analysis/adstopics/astro_forward.py
"""
import importlib.util as u, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
rc = _load("analysis/adstopics/ratechange.py", "rc")
sf = _load("analysis/adstopics/svm_furnace.py", "sf")


def r2_rows(Y, Yh, cols, mu_cols):
    res = ((Y[:, cols] - Yh[:, cols]) ** 2).sum(1)
    tot = ((Y[:, cols] - Y[:, mu_cols].mean(1, keepdims=True)) ** 2).sum(1) + 1e-9
    return 1.0 - res / tot


def r2_xsec(Y, Yh, cols):
    res = ((Y[:, cols] - Yh[:, cols]) ** 2).sum(0)
    tot = ((Y[:, cols] - Y[:, cols].mean(0, keepdims=True)) ** 2).sum(0) + 1e-9
    return 1.0 - res / tot


def phase_init(Y, TH, fit):
    """Coarse 5° grid: per-topic phase maximizing train correlation under UNIFORM attention."""
    P = 72; phases = np.deg2rad(np.arange(P) * 5.0)
    d = TH[fit][:, None, :] - phases[None, :, None]
    GU = np.exp(-(np.arctan2(np.sin(d), np.cos(d)) ** 2)).mean(2).T        # (P, nf) uniform-attention curve
    GUc = GU - GU.mean(1, keepdims=True)
    Yc = Y[:, fit] - Y[:, fit].mean(1, keepdims=True)
    score = (Yc @ GUc.T) / (np.linalg.norm(GUc, axis=1)[None] + 1e-9)      # (Tn, P)
    return phases[score.argmax(1)]


def run(b, Y, TH, loss_end, mode="softmax", seed=7, dq=16, steps=4000, lr=2e-2, device="cpu"):
    """Fit all parameters on months < loss_end (full batch); early stop on val when walls are on.
    Returns Yh (Tn,n) from the sky alone, attention A (Tn,12), phases p (Tn)."""
    import torch as T
    T.manual_seed(seed); rng = np.random.RandomState(seed); Tn, n = Y.shape
    honest = loss_end < n
    p0 = phase_init(Y, TH, np.arange(loss_end))
    U = T.tensor(np.stack([np.sin(p0), np.cos(p0)], 1), dtype=T.float32, device=device, requires_grad=True)
    Bp = T.tensor(Y[:, :loss_end].mean(1), dtype=T.float32, device=device, requires_grad=True)
    Q = T.tensor(rng.randn(Tn, dq) * 0.05, dtype=T.float32, device=device, requires_grad=True)
    K = T.tensor(rng.randn(12, dq) * 0.05, dtype=T.float32, device=device, requires_grad=True)
    s_raw = T.tensor(5.0, device=device, requires_grad=True)
    THd = T.tensor(TH.T, dtype=T.float32, device=device)                   # (12, n)
    Yt = T.tensor(Y, dtype=T.float32, device=device)
    opt = T.optim.Adam([U, Bp, Q, K, s_raw], lr=lr)
    va = np.arange(b.a, b.b)

    def forward(cols):
        p = T.atan2(U[:, 0], U[:, 1])                                      # (Tn) phases
        dphi = THd[None, :, cols] - p[:, None, None]
        g = T.exp(-(T.atan2(T.sin(dphi), T.cos(dphi)) ** 2))               # (Tn,12,|cols|)
        sc = Q @ K.T / dq ** 0.5                                           # (Tn,12) attention scores
        A = T.softmax(sc, 1) if mode == "softmax" else T.nn.functional.softplus(sc)
        return T.nn.functional.softplus(s_raw) * (A[:, :, None] * g).sum(1) + Bp[:, None], A, p

    colsT = T.arange(n, device=device); loss_cols = T.arange(loss_end, device=device)
    best, stall, state = -1e18, 0, None
    for it in range(steps):
        yh, _, _ = forward(loss_cols)
        loss = ((yh - Yt[:, :loss_end]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 99:
            with T.no_grad():
                Yh, _, _ = forward(colsT)
                sel = -((Yh[:, va] - Yt[:, va]) ** 2).mean().item() if honest else -loss.item()
            if sel > best + 1e-9:
                best, stall = sel, 0
                state = [x.detach().clone() for x in (U, Bp, Q, K, s_raw)]
            else:
                stall += 1
                if stall >= 8: break
    with T.no_grad():
        for x, sv in zip((U, Bp, Q, K, s_raw), state): x.copy_(sv)
        Yh, A, p = forward(colsT)
    return Yh.cpu().numpy(), A.cpu().numpy(), p.cpu().numpy()


def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "softmax"
    dev = "cpu"
    try:
        import torch as T
        if T.backends.mps.is_available(): dev = "mps"
    except Exception: pass
    names, Y = rc.bal.load_all(); b = rc.BenchR(Y); Y = Y.astype(np.float64)
    Tn, n = Y.shape; TH = sf.sky12(n)
    print(f"  {Tn} topics · y_j(t)=Σ_i a_ij·exp(-wrap(θ_i(t)-p_j)²)+b_j · a_ij=softmax(q_j·k_i/√d) "
          f"attention[{mode}] · p_j,b_j free params · dev {dev}", flush=True)
    allc = np.arange(n); tr = np.arange(b.a); te = np.arange(b.b, n)
    bias_only = np.repeat(Y[:, :b.a].mean(1, keepdims=True), n, 1)

    print("\n  == IN-SAMPLE (fit + R² on all 210 months) ==", flush=True)
    Yh, A, p = run(b, Y, TH, loss_end=n, mode=mode, device=dev)
    pt = r2_rows(Y, Yh, allc, allc); xs = r2_xsec(Y, Yh, allc)
    print(f"    per-topic R² mean {pt.mean():+.4f} · median {np.median(pt):+.4f} · xsec/month R² mean {xs.mean():+.4f}", flush=True)
    mass = A.mean(0); topm = sorted(zip(sf.BODIES, mass), key=lambda x: -x[1])[:5]
    print(f"    attention mass: {', '.join(f'{k} {v:.3f}' for k, v in topm)}", flush=True)
    hist = np.bincount(((np.rad2deg(p) % 360) // 30).astype(int), minlength=12)
    print("    phases by sign: " + " ".join(f"{sf.SIGNS[i][:3]} {hist[i]}" for i in range(12)), flush=True)

    print("\n  == HONEST (fit < 162 · early stop on val · R² on test [186,210)) ==", flush=True)
    Yh, A, p = run(b, Y, TH, loss_end=b.a, mode=mode, device=dev)
    pt = r2_rows(Y, Yh, te, tr); xs = r2_xsec(Y, Yh, te); xb = r2_xsec(Y, bias_only, te)
    print(f"    per-topic R² mean {pt.mean():+.4f} · median {np.median(pt):+.4f} · "
          f"xsec/month R² mean {xs.mean():+.4f} (bias-only baseline {xb.mean():+.4f})", flush=True)
    mass = A.mean(0); topm = sorted(zip(sf.BODIES, mass), key=lambda x: -x[1])[:5]
    print(f"    attention mass: {', '.join(f'{k} {v:.3f}' for k, v in topm)}", flush=True)


if __name__ == "__main__":
    main()
