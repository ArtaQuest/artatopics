#!/usr/bin/env python3
"""adstopics — FINAL COMBINATION arms: the three winning ingredients assembled.

MECHANISM: the reservoir's temperature is estimated by a causal trailing thermometer (the
trailing-12-month median — the level the system actually holds); the celestial furnace shines
through VON MISES windows (the circular Gaussian aperture, the best link OOS) onto the RESIDUAL
heat; the least-contributing lamps are unscrewed and the survivors refit (pruning, the best
generalization move within the family). Judge: untouched future test R², recency year excluded.

Arms: nowcast+vm12 · nowcast+vm-prune8 · nowcast+vm-prune6 · nowcast+vm-prune4 · (references:
level-only, nowcast+sinc12)

  python3 analysis/adstopics/combo_experiments.py [N_topics]
→ analysis/adstopics/combo_results.csv
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
SIGN_CENTERS = ex.SIGN_CENTERS
STEPS = int(os.environ.get("AQ_FIT_STEPS", 2400))
LR = 0.03
KMAX = 50.0
_D2R = float(np.pi / 180.0)


def fit_vm(Ys_resid, X, dev, body_mask=None, kernel="vonmises", loss="mse", fixed_f=None, intercept=True, split=None, per_lamp_phase=False, recency_gamma=None):
    """Von-Mises (or sinc) furnace fit on residual series; TRAIN loss, VAL checkpoint. Returns
    (pred_resid (T,n), params w/f/p) at the winning start."""
    import torch
    DT = torch.float32
    T = len(Ys_resid); n = len(Ys_resid[0]); S = 12
    a, b = split if split is not None else ex.split3(n)
    Yp = np.stack(Ys_resid)
    sd = np.maximum(Yp[:, :a].std(1), 1e-6)
    Ysn = Yp / sd[:, None]
    Xt = torch.tensor(X, dtype=DT, device=dev)
    Yt = torch.tensor(Ysn, dtype=DT, device=dev)
    fitm = torch.zeros(n, dtype=DT, device=dev)
    if recency_gamma is None:
        fitm[:a] = 1.0 / a
    else:                                                     # newer train months weigh more
        wts = torch.tensor([recency_gamma ** ((a - 1 - i) / 12.0) for i in range(a)], dtype=DT, device=dev)
        fitm[:a] = wts / wts.sum()
    ckm = torch.zeros(n, dtype=DT, device=dev); ckm[a:b] = 1.0 / (b - a)
    M = torch.ones((T, 1, NBX), dtype=DT, device=dev) if body_mask is None else \
        torch.tensor(np.asarray(body_mask, float)[:, None, :], dtype=DT, device=dev)
    if per_lamp_phase:                                        # every lamp gets its own window direction
        p = torch.tensor(SIGN_CENTERS, dtype=DT, device=dev)[None, :, None].repeat(T, 1, NBX).clone().requires_grad_(True)
    else:
        p = torch.tensor(SIGN_CENTERS, dtype=DT, device=dev).repeat(T, 1).clone().requires_grad_(True)
    w = torch.full((T, S, NBX), 1.0 / NBX, dtype=DT, device=dev, requires_grad=True)
    if fixed_f is None:
        ls = torch.zeros((T, S, NBX), dtype=DT, device=dev, requires_grad=True)
    else:                                                      # freeze every f at fixed_f (13-param spec)
        fv = float(np.log(fixed_f / (KMAX - fixed_f)) if kernel == "vonmises"
                   else np.log(fixed_f / (ex.FMAX - fixed_f)))
        ls = torch.full((T, S, NBX), fv, dtype=DT, device=dev)
    bi = torch.zeros((T, S), dtype=DT, device=dev, requires_grad=intercept)
    pars = [p, w] + ([ls] if fixed_f is None else []) + ([bi] if intercept else [])
    opt = torch.optim.Adam(pars, lr=LR)

    def forward():
        pp = p[:, :, None, :] if per_lamp_phase else p[:, :, None, None]
        d = torch.remainder(Xt[None, None, :, :] - pp + 180.0, 360.0) - 180.0
        z = d * _D2R
        if kernel == "vonmises":
            K = torch.exp((KMAX * torch.sigmoid(ls))[:, :, None, :] * (torch.cos(z) - 1.0))
        elif kernel == "cos":
            K = torch.cos(z)
        elif kernel == "gauss":
            K = torch.exp(-(z * (ex.FMAX * torch.sigmoid(ls))[:, :, None, :]) ** 2)
        elif kernel == "laplace":
            K = torch.exp(-z.abs() * (ex.FMAX * torch.sigmoid(ls))[:, :, None, :])
        elif kernel == "rcos":
            K = 0.5 * (1.0 + torch.cos(z))
        elif kernel == "sinc2":
            K = torch.sinc(z * (ex.FMAX * torch.sigmoid(ls))[:, :, None, :]) ** 2
        else:
            K = torch.sinc(z * (ex.FMAX * torch.sigmoid(ls))[:, :, None, :])
        return torch.einsum("tsmk,tsk->tsm", K, w * M) + bi[:, :, None]

    best_l = torch.full((T, S), 1e18, device=dev)
    stall = 0
    snap = None
    bw = w.detach().clone(); bls = ls.detach().clone(); bp = p.detach().clone()
    for step in range(STEPS):
        opt.zero_grad()
        pr = forward()
        e = pr - Yt[:, None, :]
        pe = e ** 2 if loss == "mse" else torch.where(e.abs() <= 1.0, e ** 2, 2.0 * e.abs() - 1.0)
        l = (pe * fitm[None, None, :]).sum(2).mean()
        l.backward()
        opt.step()
        with torch.no_grad():
            w.clamp_(min=0.0)
            if step % 20 == 19 or step == STEPS - 1:
                cur = forward().detach()
                cl = ((cur - Yt[:, None, :]) ** 2 * ckm[None, None, :]).sum(2)
                better = cl < best_l - 1e-7
                stall = stall + 1 if not bool(better.any()) else 0
                best_l = torch.where(better, cl, best_l)
                snap = cur if snap is None else torch.where(better[:, :, None], cur, snap)
                bw = torch.where(better[:, :, None], w.detach(), bw)
                bls = torch.where(better[:, :, None], ls.detach(), bls)
                bp = torch.where(better[:, :, None] if per_lamp_phase else better, p.detach(), bp)
                if stall >= 25:                              # 500 steps with zero improvement anywhere
                    break
    import torch as _t
    sel = best_l.argmin(dim=1)
    ar = _t.arange(T, device=dev)
    pred = snap[ar, sel].cpu().numpy() * sd[:, None]
    scale = KMAX if kernel == "vonmises" else ex.FMAX          # shape recorded at the kernel's true scale
    params = {"w": (bw[ar, sel] * M[:, 0, :]).cpu().numpy() * sd[:, None],
              "kappa": (scale / (1 + np.exp(-bls[ar, sel].cpu().numpy()))),
              "p": (bp[ar, sel]).cpu().numpy() % 360.0}          # (T,) shared or (T,NBX) per-lamp
    return pred, params


def main():
    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 220
    names, Ys, X = ex.load_topics(n_topics)
    n = len(Ys[0]); a, b = ex.split3(n)
    print(f"[combo] {len(Ys)} topics · {n} months (recency year excluded)")
    dev = r5._device()

    def lvl(y, upto):
        return float(np.median(y[max(0, upto - 12):upto]))

    Lc, Lb, resid = [], [], []
    for y in Ys:
        c = np.array([lvl(y, i) if i >= 6 else float(np.median(y[:max(1, i + 1)])) for i in range(n)])
        Lc.append(c); Lb.append(lvl(y, b)); resid.append(y - c)

    def score(pred_resid, tag, rows):
        val, test = [], []
        for i, y in enumerate(Ys):
            pr = np.where(np.arange(n) < b, Lc[i] + pred_resid[i], Lb[i] + pred_resid[i])
            sstv = max(((y[a:b] - y[a:b].mean()) ** 2).sum(), 1e-9)
            sst = max(((y[b:] - y[b:].mean()) ** 2).sum(), 1e-9)
            val.append(1 - ((y[a:b] - pr[a:b]) ** 2).sum() / sstv)
            test.append(1 - ((y[b:] - pr[b:]) ** 2).sum() / sst)
        val, test = np.array(val), np.array(test)
        rows.append(dict(arm=tag, median_val=float(np.median(val)), median_test=float(np.median(test)),
                         clamped_mean_test=float(np.clip(test, 0, 1).mean() * 100),
                         frac_test_pos=float((test > 0).mean())))
        print(f"  {tag:24s} val_med {np.median(val):7.3f} · TEST med {np.median(test):7.3f} · "
              f"clamped {np.clip(test,0,1).mean()*100:5.1f} · >0: {(test>0).mean()*100:4.1f}%", flush=True)
        return test

    rows = []
    # references
    lv = np.array([1 - ((y[b:] - Lb[i]) ** 2).sum() / max(((y[b:] - y[b:].mean()) ** 2).sum(), 1e-9)
                   for i, y in enumerate(Ys)])
    rows.append(dict(arm="level-only", median_val=None, median_test=float(np.median(lv)),
                     clamped_mean_test=float(np.clip(lv, 0, 1).mean() * 100),
                     frac_test_pos=float((lv > 0).mean())))
    print(f"  {'level-only':24s} TEST med {np.median(lv):7.3f} · clamped {np.clip(lv,0,1).mean()*100:5.1f} · "
          f">0: {(lv>0).mean()*100:4.1f}%", flush=True)

    pred_s, _ = fit_vm(resid, X, dev, kernel="sinc")
    score(pred_s, "nowcast+sinc12", rows)
    pred_v, par_v = fit_vm(resid, X, dev, kernel="vonmises")
    score(pred_v, "nowcast+vm12", rows)

    # pruning on the vm fit: rank bodies by train residual-variance contribution, refit survivors
    def contributions(par):
        C = np.zeros((len(Ys), NBX))
        for i in range(len(Ys)):
            z = np.exp(par["kappa"][i][None, :] * (np.cos(np.deg2rad((X[:a] - par["p"][i] + 180.0) % 360.0 - 180.0)) - 1.0))
            C[i] = (par["w"][i][None, :] * z).std(0)
        return C

    C = contributions(par_v)
    for k in (8, 6, 4):
        m = np.zeros_like(C)
        idx = np.argsort(-C, axis=1)[:, :k]
        for i in range(C.shape[0]):
            m[i, idx[i]] = 1.0
        pk, _ = fit_vm(resid, X, dev, body_mask=m, kernel="vonmises")
        score(pk, f"nowcast+vm-prune{k}", rows)

    pd.DataFrame(rows).to_csv("analysis/adstopics/combo_results.csv", index=False)

if __name__ == "__main__":
    main()
