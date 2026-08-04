#!/usr/bin/env python3
"""adstopics MODEL 2 — the platform sinc phase model, upgraded against overfitting.

Changes vs the topic-500 atlas fit (each addressing a documented failure mode):

1. EXPLICIT TREND BASIS. y_hat = w·sinc(f·wrap(x−p)) + b + c1·τ + c2·τ² (τ = centred, scaled month
   index). On trend-heavy topics the old fit spent its slow-body sinc terms on the secular trend and
   the global phase drifted; giving the trend its own two coefficients frees the phase to be purely
   seasonal.
2. HONEST THREE-WAY TIME SPLIT. train 70% | validation 15% | test 15% (time-blocked, in that order).
   Weight decay is selected per topic on VALIDATION (sweep over WD_GRID), with validation
   early-checkpointing inside each fit; the reported out-of-sample R² comes from the untouched TEST
   tail. The final classification fit refits on train+validation at the selected decay (test still
   untouched), and the atlas phase is read from a last full-window refit only AFTER test metrics are
   frozen.
3. REGULARISATION. Adam weight_decay on the seasonal weights and frequencies (never on trend/bias),
   swept on validation — the anti-overfitting knob the old fit lacked.

Everything is deterministic: fixed 12 sign-centre starts, fixed steps, fixed split boundaries, no
randomness. API mirrors topic500_reference_solution: fit_many2(Ys, Xs) -> per-topic dict with
par, phase, sign, r2_test (OOS), r2_train, decay, and baselines (harmonic-3 OOS R², raw peak sign).
"""
import math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))          # repo root on path if needed
sys.path.insert(0, os.path.join(os.path.dirname(HERE)))     # analysis/
import importlib.util as _u
_s = _u.spec_from_file_location("r5", os.path.join(os.path.dirname(HERE), "topic500_reference_solution.py"))
r5 = _u.module_from_spec(_s); _s.loader.exec_module(_s and r5)

SIGNS = r5.SIGNS
NB = r5.NB
SIGN_CENTERS = r5.SIGN_CENTERS
FMAX = r5.FMAX
STEPS = 2400
LR = 0.03
WD_GRID = [0.0, 3e-3, 3e-2]                                  # weight-decay sweep (validation-selected)
TRAIN_F, VAL_F = 0.70, 0.15                                  # the rest = TEST
_D2R = float(np.pi / 180.0)


def split3(n):
    a = int(round(TRAIN_F * n)); b = int(round((TRAIN_F + VAL_F) * n))
    return a, b


def predict2(par, X, tau):
    par = np.asarray(par, float)
    w = par[:NB]; f = par[NB:2 * NB]; p = par[2 * NB]; b = par[2 * NB + 1]
    c1, c2 = par[2 * NB + 2], par[2 * NB + 3]
    z = np.sinc(np.deg2rad(r5.wrap(np.asarray(X, float) - p)) * f[None, :])
    return z @ w + b + c1 * tau + c2 * tau ** 2


def _fit_chunk2(Ys, Xs, mode_masks, wd, dev):
    """One batched torch fit at weight decay wd. mode_masks = (fit_w, ck_w) row-normalised weights."""
    import torch
    DT = torch.float32
    T = len(Ys)
    ns = np.array([len(y) for y in Ys], int)
    n_max = int(ns.max()); S = 12
    Yp = np.zeros((T, n_max)); Xp = np.zeros((T, n_max, NB)); Tau = np.zeros((T, n_max))
    for i, (y, X) in enumerate(zip(Ys, Xs)):
        o = r5.canon_order(X)
        Yp[i, :ns[i]] = y[o]; Xp[i, :ns[i], :] = X[o]
        Tau[i, :ns[i]] = (np.arange(ns[i]) - ns[i] / 2.0) / max(1.0, ns[i] / 2.0)
    mu = np.array([Yp[i, :ns[i]].mean() for i in range(T)])
    sd = np.array([max(Yp[i, :ns[i]].std(), 1e-6) for i in range(T)])
    Ysn = (Yp - mu[:, None]) / sd[:, None]
    fitw, ckw = mode_masks
    Xt = torch.tensor(Xp, dtype=DT, device=dev)
    Yt = torch.tensor(Ysn, dtype=DT, device=dev)
    Tt = torch.tensor(Tau, dtype=DT, device=dev)
    FW = torch.tensor(fitw, dtype=DT, device=dev)
    CW = torch.tensor(ckw, dtype=DT, device=dev)
    p = torch.tensor(SIGN_CENTERS, dtype=DT, device=dev).repeat(T, 1).clone().requires_grad_(True)
    w = torch.full((T, S, NB), 1.0 / NB, dtype=DT, device=dev, requires_grad=True)
    logf0 = float(np.log(1.0 / (FMAX - 1.0)))
    logf = torch.full((T, S, NB), logf0, dtype=DT, device=dev, requires_grad=True)
    b = torch.zeros((T, S), dtype=DT, device=dev, requires_grad=True)
    c1 = torch.zeros((T, S), dtype=DT, device=dev, requires_grad=True)
    c2 = torch.zeros((T, S), dtype=DT, device=dev, requires_grad=True)
    opt = torch.optim.Adam([
        {"params": [w, logf], "weight_decay": wd},           # regularise the SEASONAL machinery only
        {"params": [p, b, c1, c2], "weight_decay": 0.0},
    ], lr=LR)

    def forward(P, W, LF, B, C1, C2):
        d = torch.remainder(Xt[:, None, :, :] - P[:, :, None, None] + 180.0, 360.0) - 180.0
        z = torch.sinc(d * _D2R * (FMAX * torch.sigmoid(LF))[:, :, None, :])
        return (torch.einsum("tsmk,tsk->tsm", z, W) + B[:, :, None]
                + C1[:, :, None] * Tt[:, None, :] + C2[:, :, None] * Tt[:, None, :] ** 2)

    best_l = torch.full((T, S), 1e18, device=dev)
    best = {k: v.detach().clone() for k, v in (("p", p), ("w", w), ("logf", logf), ("b", b), ("c1", c1), ("c2", c2))}
    for step in range(STEPS):
        opt.zero_grad()
        pr = forward(p, w, logf, b, c1, c2)
        loss = ((pr - Yt[:, None, :]) ** 2 * FW[:, None, :]).sum(2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w.clamp_(min=0.0)
            if step % 20 == 19 or step == STEPS - 1:
                cl = ((forward(p, w, logf, b, c1, c2) - Yt[:, None, :]) ** 2 * CW[:, None, :]).sum(2)
                better = cl < best_l - 1e-7
                best_l = torch.where(better, cl, best_l)
                for k, cur in (("p", p), ("w", w), ("logf", logf), ("b", b), ("c1", c1), ("c2", c2)):
                    m = better[..., None] if cur.dim() == 3 else better
                    best[k] = torch.where(m, cur.detach(), best[k])
    sel = best_l.argmin(dim=1)
    ar = torch.arange(T, device=dev)
    W = (best["w"][ar, sel]).cpu().numpy() * sd[:, None]
    F = (FMAX * torch.sigmoid(best["logf"][ar, sel])).cpu().numpy()
    P = (best["p"][ar, sel]).cpu().numpy() % 360.0
    B = (best["b"][ar, sel]).cpu().numpy() * sd + mu
    C1 = (best["c1"][ar, sel]).cpu().numpy() * sd
    C2 = (best["c2"][ar, sel]).cpu().numpy() * sd
    return np.column_stack([W, F, P[:, None], B[:, None], C1[:, None], C2[:, None]])


def masks(ns, lo_frac, hi_frac, ck_lo, ck_hi):
    """Row-normalised fit + checkpoint masks over each topic's CANONICAL order (fractions of n)."""
    T = len(ns); n_max = max(ns)
    fit = np.zeros((T, n_max)); ck = np.zeros((T, n_max))
    for i, n in enumerate(ns):
        a, b = int(round(lo_frac * n)), int(round(hi_frac * n))
        ca, cb = int(round(ck_lo * n)), int(round(ck_hi * n))
        fit[i, a:b] = 1.0; ck[i, ca:cb] = 1.0
    fit /= np.maximum(fit.sum(1, keepdims=True), 1e-9)
    ck /= np.maximum(ck.sum(1, keepdims=True), 1e-9)
    return fit, ck


def _r2(y, p):
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - p) ** 2)) / ss if ss > 1e-9 else 0.0


def fit_many2(Ys, Xs, chunk=200, device=None, progress=False):
    """Full protocol per topic: WD selected on VALIDATION, OOS R² from the untouched TEST tail,
    final phase from a train+val refit at the selected WD (then a full-window refit for the atlas
    parameters, after metrics are frozen). Returns a list of per-topic dicts."""
    import torch  # noqa
    dev = device or r5._device()
    T = len(Ys)
    ns = [len(y) for y in Ys]
    out = [dict() for _ in range(T)]
    for lo in range(0, T, chunk):
        hi = min(T, lo + chunk)
        ys = [np.asarray(Ys[i], float) for i in range(lo, hi)]
        xs = [np.asarray(Xs[i], float) for i in range(lo, hi)]
        nn = [len(y) for y in ys]
        # 1) sweep WD: fit on TRAIN, checkpoint on VAL
        val_best = None
        pars_by_wd = {}
        for wd in WD_GRID:
            par = _fit_chunk2(ys, xs, masks(nn, 0.0, TRAIN_F, TRAIN_F, TRAIN_F + VAL_F), wd, dev)
            pars_by_wd[wd] = par
            # validation R² per topic
            vr = []
            for i in range(len(ys)):
                o = r5.canon_order(xs[i]); y = ys[i][o]; X = xs[i][o]
                tau = (np.arange(nn[i]) - nn[i] / 2.0) / max(1.0, nn[i] / 2.0)
                a, b = split3(nn[i])
                pred = predict2(par[i], X, tau)
                vr.append(_r2(y[a:b], pred[a:b]))
            vr = np.array(vr)
            val_best = vr if val_best is None else val_best
            if wd == WD_GRID[0]:
                sel_wd = np.full(len(ys), wd); sel_val = vr
            else:
                better = vr > sel_val
                sel_wd = np.where(better, wd, sel_wd); sel_val = np.where(better, vr, sel_val)
        # 2) per selected WD: refit on TRAIN+VAL (checkpoint on VAL), score the untouched TEST
        for wd in WD_GRID:
            idx = [i for i in range(len(ys)) if sel_wd[i] == wd]
            if not idx: continue
            par = _fit_chunk2([ys[i] for i in idx], [xs[i] for i in idx],
                              masks([nn[i] for i in idx], 0.0, TRAIN_F + VAL_F, TRAIN_F, TRAIN_F + VAL_F), wd, dev)
            # 3) full-window refit for the final atlas parameters (after test metrics)
            par_full = _fit_chunk2([ys[i] for i in idx], [xs[i] for i in idx],
                                   masks([nn[i] for i in idx], 0.0, 1.0, 0.0, 1.0), wd, dev)
            for j, i in enumerate(idx):
                o = r5.canon_order(xs[i]); y = ys[i][o]; X = xs[i][o]
                tau = (np.arange(nn[i]) - nn[i] / 2.0) / max(1.0, nn[i] / 2.0)
                a, b = split3(nn[i])
                pred = predict2(par[j], X, tau)
                predf = predict2(par_full[j], X, tau)
                g = out[lo + i]
                g["decay"] = float(wd)
                g["r2_val"] = round(float(sel_val[i]), 4)
                g["r2_test"] = round(_r2(y[b:], pred[b:]), 4)          # the honest OOS number
                g["r2_train"] = round(_r2(y[:a], pred[:a]), 4)
                g["r2_full_insample"] = round(_r2(y, predf), 4)
                g["par"] = [round(float(v), 6) for v in par_full[j]]
                g["phase"] = round(float(par_full[j][2 * NB] % 360.0), 2)
                g["sign"] = SIGNS[int(par_full[j][2 * NB] % 360.0 // 30) % 12]
                # harmonic-3 baseline (annual cos+sin+mean, closed form on train+val, scored on test)
                m = np.arange(nn[i])
                H = np.column_stack([np.ones(nn[i]), np.cos(2 * np.pi * m / 12), np.sin(2 * np.pi * m / 12),
                                     tau, tau ** 2])
                coef, *_ = np.linalg.lstsq(H[:b], y[:b], rcond=None)
                g["r2_test_harmonic"] = round(_r2(y[b:], H[b:] @ coef), 4)
        if progress:
            print(f"  model2 fit {hi}/{T}", flush=True)
    return out
