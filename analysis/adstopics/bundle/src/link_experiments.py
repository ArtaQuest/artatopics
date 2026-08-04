#!/usr/bin/env python3
"""adstopics — LINK-FUNCTION experiments: the aperture through which an aligned body delivers heat.

Same machine as the spec (12 lamps, one window p, positive wattages w_i, per-body shape s_i, 12
sign-centre starts, 25 parameters), different TRANSMISSION PATTERNS K(Δ; s_i) for misalignment Δ:

  sinc      sinc(s·Δ)                 single-slit diffraction AMPLITUDE (the baseline; side lobes ±)
  sinc2     sinc²(s·Δ)                single-slit INTENSITY (Fejér kernel — physically nonnegative)
  gauss     exp(−s·Δ²)                Gaussian bump, exp(−(x−μ)²/v) with v = 1/s (operator's example)
  vonmises  exp(κ(cosΔ − 1))          the circular Gaussian — the natural bump on a wheel
  laplace   exp(−s·|Δ|)               exponential decay from alignment
  rcos      ((1+cosΔ)/2)^m            raised-cosine aperture, sharpness m

All shape parameters positive (softplus/sigmoid), all weights positive (projection). Identical
protocol: recency year excluded; train 70 | val 15 (checkpoint) | TEST 15 untouched future; judged
on future out-of-sample R² (median, clamped mean, frac>0).

  python3 analysis/adstopics/link_experiments.py [N_topics]
→ analysis/adstopics/link_results.csv
"""
import importlib.util as u, json, os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

tf = _load("analysis/trends_fit.py", "tf")
r5 = _load("analysis/topic500_reference_solution.py", "r5")
ex = _load("analysis/adstopics/experiments.py", "ex")

NBX = ex.NBX
FMAX = ex.FMAX
SIGN_CENTERS = ex.SIGN_CENTERS
STEPS = 2400
LR = 0.03
_D2R = float(np.pi / 180.0)
KMAX = 50.0                                                 # cap for gauss/vonmises/laplace sharpness


def fit_link(Ys, X, kernel, dev, seed=7):
    import torch
    DT = torch.float32
    T = len(Ys); n = len(Ys[0]); S = 12
    a, b = ex.split3(n)
    Yp = np.stack(Ys)
    mu = Yp[:, :a].mean(1); sd = np.maximum(Yp[:, :a].std(1), 1e-6)
    Ysn = (Yp - mu[:, None]) / sd[:, None]
    Xt = torch.tensor(X, dtype=DT, device=dev)
    Yt = torch.tensor(Ysn, dtype=DT, device=dev)
    fitm = torch.zeros(n, dtype=DT, device=dev); fitm[:a] = 1.0 / a
    ckm = torch.zeros(n, dtype=DT, device=dev); ckm[a:b] = 1.0 / (b - a)

    p = torch.tensor(SIGN_CENTERS, dtype=DT, device=dev).repeat(T, 1).clone().requires_grad_(True)
    w = torch.full((T, S, NBX), 1.0 / NBX, dtype=DT, device=dev, requires_grad=True)
    ls = torch.zeros((T, S, NBX), dtype=DT, device=dev, requires_grad=True)   # shape param (latent)
    bi = torch.zeros((T, S), dtype=DT, device=dev, requires_grad=True)        # intercept (fair for
    opt = torch.optim.Adam([p, w, ls, bi], lr=LR)                             # nonneg kernels)

    def forward():
        d = torch.remainder(Xt[None, None, :, :] - p[:, :, None, None] + 180.0, 360.0) - 180.0
        z = d * _D2R                                        # radians in (-pi, pi]
        if kernel == "sinc":
            s = FMAX * torch.sigmoid(ls)
            K = torch.sinc(z * s[:, :, None, :])
        elif kernel == "sinc2":
            s = FMAX * torch.sigmoid(ls)
            K = torch.sinc(z * s[:, :, None, :]) ** 2
        elif kernel == "gauss":
            s = KMAX * torch.sigmoid(ls)
            K = torch.exp(-s[:, :, None, :] * z ** 2)
        elif kernel == "vonmises":
            s = KMAX * torch.sigmoid(ls)
            K = torch.exp(s[:, :, None, :] * (torch.cos(z) - 1.0))
        elif kernel == "laplace":
            s = KMAX * torch.sigmoid(ls)
            K = torch.exp(-s[:, :, None, :] * z.abs())
        elif kernel == "rcos":
            s = KMAX * torch.sigmoid(ls)
            K = ((1.0 + torch.cos(z)) / 2.0) ** s[:, :, None, :]
        else:
            raise ValueError(kernel)
        return torch.einsum("tsmk,tsk->tsm", K, w) + bi[:, :, None]

    best_l = torch.full((T, S), 1e18, device=dev)
    snap = None
    for step in range(STEPS):
        opt.zero_grad()
        pr = forward()
        loss = ((pr - Yt[:, None, :]) ** 2 * fitm[None, None, :]).sum(2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w.clamp_(min=0.0)
            if step % 20 == 19 or step == STEPS - 1:
                cur = forward().detach()
                cl = ((cur - Yt[:, None, :]) ** 2 * ckm[None, None, :]).sum(2)
                better = cl < best_l - 1e-7
                best_l = torch.where(better, cl, best_l)
                snap = cur if snap is None else torch.where(better[:, :, None], cur, snap)
    sel = best_l.argmin(dim=1)
    ar = torch.arange(T, device=dev)
    pred = snap[ar, sel].cpu().numpy() * sd[:, None] + mu[:, None]
    val = np.array([1 - ((Yp[i, a:b] - pred[i, a:b]) ** 2).sum() /
                    max(((Yp[i, a:b] - Yp[i, a:b].mean()) ** 2).sum(), 1e-9) for i in range(T)])
    test = np.array([1 - ((Yp[i, b:] - pred[i, b:]) ** 2).sum() /
                     max(((Yp[i, b:] - Yp[i, b:].mean()) ** 2).sum(), 1e-9) for i in range(T)])
    return val, test


def main():
    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    names, Ys, X = ex.load_topics(n_topics)
    print(f"[links] {len(Ys)} topics · {len(Ys[0])} months (recency year excluded)")
    dev = r5._device()
    rows = []
    for kernel in ("sinc", "sinc2", "gauss", "vonmises", "laplace", "rcos"):
        val, test = fit_link(Ys, X, kernel, dev)
        rows.append(dict(arm=f"link:{kernel}", median_val=float(np.median(val)),
                         median_test=float(np.median(test)),
                         clamped_mean_test=float(np.clip(test, 0, 1).mean() * 100),
                         frac_test_pos=float((test > 0).mean())))
        print(f"  {kernel:9s} val_med {np.median(val):7.3f} · TEST med {np.median(test):7.3f} · "
              f"clamped {np.clip(test,0,1).mean()*100:5.1f} · >0: {(test>0).mean()*100:4.1f}%", flush=True)
    pd.DataFrame(rows).to_csv("analysis/adstopics/link_results.csv", index=False)

if __name__ == "__main__":
    main()
