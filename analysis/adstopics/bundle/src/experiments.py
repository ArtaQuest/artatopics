#!/usr/bin/env python3
"""adstopics — MODEL EXPERIMENTS on the operator-specified baseline class.

BASELINE (the spec):   y_hat = SUM_i  w_i * sinc( f_i * wrap(x_i - p) )      (i over 12 bodies,
  the Moon included as its SYNODIC phase — elongation from the Sun, the platform's moon clock)
  - 24 weights (w_i, f_i) + 1 phase p = 25 parameters; NO intercept, NO trend term
  - ALL parameters positive: w_i >= 0 (projected), f_i in (0, FMAX] (sigmoid), p in [0,360)
  - every fit tries 12 models: p initialised at each sign centre (15, 45, ..., 345), best kept

EXPERIMENT ARMS (each deterministic; per-topic-per-start inits from fixed-seed RNG):
  - w init: const 1/11, or |N(0, sigma_w)| for sigma_w in {0.05, 0.2, 0.5}
  - f init: const 1.0, or |N(1, sigma_f)| for sigma_f in {0.25, 0.75}, or U(0.2, 4)
  - optional arms beyond the spec (candidate improvements, same judge): +intercept, +linear/quad
    trend, weight decay {3e-3, 3e-2}

PROTOCOL: the last 12 fetched months (the recency year) are EXCLUDED from the study entirely
(platform DROP_LAST); on the remaining clean window (2008-01..2025-06, 210 months) the FUTURE
out-of-sample split is train = first 70%, validation = next 15% (arm/init selection), TEST = the
final 15% — future months, untouched by any choice. The ultimate metric is mean/median TEST R².

  python3 analysis/adstopics/experiments.py [N_topics]
→ analysis/adstopics/experiments_results.csv + experiments_summary.json
"""
import importlib.util as u, itertools, json, os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

tf = _load("analysis/trends_fit.py", "tf")
r5 = _load("analysis/topic500_reference_solution.py", "r5")

NBX = r5.NB                                            # 11 bodies — MOON REMOVED (operator 2026-07-15: monthly-aliased noise)
FMAX = r5.FMAX
SIGN_CENTERS = np.arange(15.0, 360.0, 30.0)
STEPS = 2400
LR = 0.03
TRAIN_F, VAL_F = 0.70, 0.15
_D2R = float(np.pi / 180.0)


def split3(n):
    a = int(round(TRAIN_F * n)); b = int(round((TRAIN_F + VAL_F) * n))
    return a, b


def load_topics(n_max):
    grid = pd.DatetimeIndex(tf.GRID)
    i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
    iend = len(grid) - tf.DROP_LAST                      # the recency year excluded ENTIRELY
    lon = tf.ephemeris()
    X = np.column_stack([np.asarray(lon[b], float)[i0:iend] for b in tf.BODIES])   # 11 bodies, NO moon
    vocab = sorted(json.load(open("analysis/adstopics/vocabulary.json")))
    _bl = set(json.load(open("analysis/adstopics/blacklist.json")).get("excluded_topics", []))
    vocab = [t for t in vocab if t not in _bl]
    Ys, names = [], []
    for t in vocab:
        p = f"analysis/adstopics/series/{tf.slug(t)}.csv"
        if not os.path.exists(p): continue
        df = pd.read_csv(p); df["Time"] = pd.to_datetime(df["Time"])
        v = pd.to_numeric(df.drop_duplicates("Time").set_index("Time")["v"].reindex(grid[i0:iend]), errors="coerce")
        if v.notna().sum() < (iend - i0) * 0.5: continue
        y = v.interpolate(limit_direction="both").to_numpy(float)
        if not np.isfinite(y).all(): continue
        a, b = split3(len(y))
        # EXPERIMENT INCLUSION GATE: enough volume + variance that R² is meaningful in every block
        if float(y.max()) < 10 or float(y[:a].std()) < 1.0 or float(y[b:].std()) < 1.0: continue
        Ys.append(y); names.append(t)
        if len(Ys) >= n_max: break
    return names, Ys, X


def fit_arm(Ys, X, arm, dev, body_mask=None, level_offsets=None, raw_Ys=None):
    """Batched fit of ALL topics under one arm config. Fits on TRAIN (checkpoint on VALIDATION);
    returns (val_r2, test_r2, params) with the TEST tail untouched by any selection. body_mask
    (T, NBX) of 0/1 restricts each topic to its surviving bodies (pruning experiments)."""
    import torch
    DT = torch.float32
    T = len(Ys)
    n = len(Ys[0])
    a, b = split3(n)
    S = 12
    Yp = np.stack(Ys)                                     # chronological; canonical order irrelevant here
    mu = Yp[:, :b].mean(1); sd = np.maximum(Yp[:, :b].std(1), 1e-6)
    Ysn = (Yp - mu[:, None]) / sd[:, None]
    tau = (np.arange(n) - n / 2.0) / (n / 2.0)
    arm.setdefault("recency", False)
    rng = np.random.default_rng(arm["seed"])

    # inits (positive by construction)
    if arm["w_init"] == "const":
        w0 = np.full((T, S, NBX), 1.0 / NBX)
    else:
        w0 = np.abs(rng.normal(0.0, arm["w_sigma"], (T, S, NBX)))
    if arm["f_init"] == "const":
        f0 = np.full((T, S, NBX), 1.0)
    elif arm["f_init"] == "normal":
        f0 = np.clip(np.abs(rng.normal(1.0, arm["f_sigma"], (T, S, NBX))), 0.05, FMAX - 0.05)
    else:                                                  # uniform
        f0 = rng.uniform(0.2, 4.0, (T, S, NBX))
    logf0 = np.log(f0 / (FMAX - f0))                       # sigmoid^-1(f/FMAX)

    Xt = torch.tensor(X, dtype=DT, device=dev)             # (n, NB) shared sky
    Yt = torch.tensor(Ysn, dtype=DT, device=dev)
    Tt = torch.tensor(tau, dtype=DT, device=dev)
    if arm.get("recency"):
        hl = 60.0
        rw = 0.5 ** ((a - 1 - np.arange(n)) / hl); rw[a:] = 0.0
        fitm = torch.tensor(rw / rw.sum(), dtype=DT, device=dev)
    else:
        fitm = torch.zeros(n, dtype=DT, device=dev); fitm[:a] = 1.0 / a
    ckm = torch.zeros(n, dtype=DT, device=dev); ckm[a:b] = 1.0 / (b - a)

    M = torch.ones((T, 1, NBX), dtype=DT, device=dev) if body_mask is None else \
        torch.tensor(np.asarray(body_mask, float)[:, None, :], dtype=DT, device=dev)
    p = torch.tensor(SIGN_CENTERS, dtype=DT, device=dev).repeat(T, 1).clone().requires_grad_(True)
    w = torch.tensor(w0, dtype=DT, device=dev, requires_grad=True)
    logf = torch.tensor(logf0, dtype=DT, device=dev, requires_grad=True)
    extra = []
    if arm["intercept"]:
        bpar = torch.zeros((T, S), dtype=DT, device=dev, requires_grad=True); extra.append(bpar)
    if arm["trend"]:
        c1 = torch.zeros((T, S), dtype=DT, device=dev, requires_grad=True)
        c2 = torch.zeros((T, S), dtype=DT, device=dev, requires_grad=True)
        extra += [c1, c2]
    opt = torch.optim.Adam([
        {"params": [w, logf], "weight_decay": arm["decay"]},
        {"params": [p] + extra, "weight_decay": 0.0},
    ], lr=LR)

    def forward():
        d = torch.remainder(Xt[None, None, :, :] - p[:, :, None, None] + 180.0, 360.0) - 180.0
        z = torch.sinc(d * _D2R * (FMAX * torch.sigmoid(logf))[:, :, None, :])
        out = torch.einsum("tsmk,tsk->tsm", z, w * M)
        if arm["intercept"]:
            out = out + bpar[:, :, None]
        if arm["trend"]:
            out = out + c1[:, :, None] * Tt[None, None, :] + c2[:, :, None] * Tt[None, None, :] ** 2
        return out

    best_l = torch.full((T, S), 1e18, device=dev)
    snap = None
    bw = w.detach().clone(); bf = logf.detach().clone(); bp = p.detach().clone()
    for step in range(STEPS):
        opt.zero_grad()
        pr = forward()
        loss = ((pr - Yt[:, None, :]) ** 2 * fitm[None, None, :]).sum(2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w.clamp_(min=0.0)
            if step % 20 == 19 or step == STEPS - 1:
                cl = ((forward() - Yt[:, None, :]) ** 2 * ckm[None, None, :]).sum(2)
                better = cl < best_l - 1e-7
                best_l = torch.where(better, cl, best_l)
                cur = forward().detach()
                snap = cur if snap is None else torch.where(better[:, :, None], cur, snap)
                bw = torch.where(better[:, :, None], w.detach(), bw)
                bf = torch.where(better[:, :, None], logf.detach(), bf)
                bp = torch.where(better, p.detach(), bp)
    sel = best_l.argmin(dim=1)
    ar = torch.arange(T, device=dev)
    pred = snap[ar, sel].cpu().numpy() * sd[:, None] + mu[:, None]
    if level_offsets is not None:
        Yraw = np.stack(raw_Ys)
        pred2 = pred + np.array([lo for lo, _ in level_offsets])[:, None]
        val_r2 = np.array([1 - ((Yraw[i, a:b] - pred2[i, a:b]) ** 2).sum() /
                           max(((Yraw[i, a:b] - Yraw[i, a:b].mean()) ** 2).sum(), 1e-9) for i in range(T)])
        test_r2 = np.array([1 - ((Yraw[i, b:] - pred2[i, b:]) ** 2).sum() /
                            max(((Yraw[i, b:] - Yraw[i, b:].mean()) ** 2).sum(), 1e-9) for i in range(T)])
        params = {"w": None}
        return val_r2, test_r2, params
    val_r2 = np.array([1 - ((Yp[i, a:b] - pred[i, a:b]) ** 2).sum() /
                       max(((Yp[i, a:b] - Yp[i, a:b].mean()) ** 2).sum(), 1e-9) for i in range(T)])
    test_r2 = np.array([1 - ((Yp[i, b:] - pred[i, b:]) ** 2).sum() /
                        max(((Yp[i, b:] - Yp[i, b:].mean()) ** 2).sum(), 1e-9) for i in range(T)])
    params = {"w": (bw[ar, sel] * M[:, 0, :]).cpu().numpy() * sd[:, None],
              "f": (FMAX * torch.sigmoid(bf[ar, sel])).cpu().numpy(),
              "p": (bp[ar, sel]).cpu().numpy() % 360.0}
    return val_r2, test_r2, params


def main():
    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    names, Ys, X = load_topics(n_topics)
    print(f"[experiments] {len(Ys)} topics · window {len(Ys[0])} months (recency year excluded)")
    dev = r5._device()

    arms = []
    # the SPEC baseline grid: init experiments, no intercept/trend/decay
    for w_init, w_sigma in (("const", 0.0), ("normal", 0.05), ("normal", 0.2), ("normal", 0.5)):
        for f_init, f_sigma in (("const", 0.0), ("normal", 0.25), ("normal", 0.75), ("uniform", 0.0)):
            arms.append(dict(name=f"spec w={w_init}{w_sigma or ''} f={f_init}{f_sigma or ''}",
                             w_init=w_init, w_sigma=w_sigma, f_init=f_init, f_sigma=f_sigma,
                             intercept=False, trend=False, decay=0.0, seed=7))
    # candidate improvements (same judge)
    base = dict(w_init="const", w_sigma=0.0, f_init="const", f_sigma=0.0, seed=7)
    arms.append(dict(name="spec+intercept", **base, intercept=True, trend=False, decay=0.0))
    arms.append(dict(name="spec+trend", **base, intercept=True, trend=True, decay=0.0))
    arms.append(dict(name="spec+trend+wd3e-3", **base, intercept=True, trend=True, decay=3e-3))
    arms.append(dict(name="spec+trend+wd3e-2", **base, intercept=True, trend=True, decay=3e-2))
    arms.append(dict(name="spec+recency-loss", **base, intercept=False, trend=False, decay=0.0, recency=True))
    arms.append(dict(name="spec+intercept+recency", **base, intercept=True, trend=False, decay=0.0, recency=True))
    arms.append(dict(name="spec+int+trend+recency", **base, intercept=True, trend=True, decay=0.0, recency=True))

    rows = []
    for arm in arms:
        val, test, _ = fit_arm(Ys, X, arm, dev)
        cl = np.clip(test, 0, 1)
        rows.append(dict(arm=arm["name"], median_val=float(np.median(val)),
                         median_test=float(np.median(test)),
                         clamped_mean_test=float(cl.mean() * 100),
                         frac_test_pos=float((test > 0).mean())))
        print(f"  {arm['name']:34s} val_med {np.median(val):7.3f} · TEST med {np.median(test):7.3f} · "
              f"clamped {cl.mean()*100:5.1f} · >0: {(test>0).mean()*100:4.1f}%", flush=True)
    # ── LEVEL-ANCHORED designs (the topic-500 duel lesson: level is ~random-walk; anchor it) ──
    n = len(Ys[0]); a, b = split3(n)

    def lvl(y, upto):
        return float(np.median(y[max(0, upto - 12):upto]))          # trailing-12-month median level

    lv_only, nowseas, nowharm = [], [], []
    m = np.arange(n)
    H = np.column_stack([np.cos(2 * np.pi * m / 12), np.sin(2 * np.pi * m / 12)])
    resid_Ys = []
    for y in Ys:
        L_train_end = lvl(y, a)                                      # level known at train end
        # causal trailing level curve over the train window (for the residual target)
        Lc = np.array([lvl(y, i) if i >= 6 else float(np.median(y[:max(1, i + 1)])) for i in range(n)])
        resid_Ys.append(y - Lc)
        sstv = max(((y[a:b] - y[a:b].mean()) ** 2).sum(), 1e-9)
        sst = max(((y[b:] - y[b:].mean()) ** 2).sum(), 1e-9)
        L_val_end = lvl(y, b)
        lv_only.append(1 - ((y[b:] - L_val_end) ** 2).sum() / sst)
        # nowcast + harmonic seasonal (closed form on train residual)
        coef, *_ = np.linalg.lstsq(H[:a], (y - Lc)[:a], rcond=None)
        pr = L_val_end + H @ coef
        nowharm.append(1 - ((y[b:] - pr[b:]) ** 2).sum() / sst)
    print(f"  {'LEVEL-only (trailing-12 median)':34s} TEST med {np.median(lv_only):7.3f} · "
          f"clamped {np.clip(lv_only,0,1).mean()*100:5.1f} · >0: {(np.array(lv_only)>0).mean()*100:4.1f}%", flush=True)
    print(f"  {'nowcast+harmonic (closed form)':34s} TEST med {np.median(nowharm):7.3f} · "
          f"clamped {np.clip(nowharm,0,1).mean()*100:5.1f} · >0: {(np.array(nowharm)>0).mean()*100:4.1f}%", flush=True)
    rows.append(dict(arm="level-only", median_val=None, median_test=float(np.median(lv_only)),
                     clamped_mean_test=float(np.clip(lv_only,0,1).mean()*100),
                     frac_test_pos=float((np.array(lv_only)>0).mean())))
    rows.append(dict(arm="nowcast+harmonic", median_val=None, median_test=float(np.median(nowharm)),
                     clamped_mean_test=float(np.clip(nowharm,0,1).mean()*100),
                     frac_test_pos=float((np.array(nowharm)>0).mean())))
    # nowcast + SINC seasonal on the causal residual (the spec model fit to the detrended series;
    # test prediction = val-end level + fitted seasonal)
    base_ns = dict(name="nowcast+sinc", w_init="const", w_sigma=0.0, f_init="const", f_sigma=0.0,
                   intercept=False, trend=False, decay=0.0, seed=7)
    vns, tns, _ = fit_arm(resid_Ys, X, base_ns, dev, level_offsets=[(lvl(y, b), b) for y in Ys], raw_Ys=Ys)
    rows.append(dict(arm="nowcast+sinc", median_val=float(np.median(vns)), median_test=float(np.median(tns)),
                     clamped_mean_test=float(np.clip(tns,0,1).mean()*100),
                     frac_test_pos=float((tns>0).mean())))
    print(f"  {'nowcast+sinc (spec on residual)':34s} val_med {np.median(vns):7.3f} · TEST med {np.median(tns):7.3f} · "
          f"clamped {np.clip(tns,0,1).mean()*100:5.1f} · >0: {(tns>0).mean()*100:4.1f}%", flush=True)

    # ── MULTI-STAGE PRUNING (fit all 12 -> prune least-important bodies on TRAIN -> refit) ──
    base = dict(name="prune-base", w_init="const", w_sigma=0.0, f_init="const", f_sigma=0.0,
                intercept=False, trend=False, decay=0.0, seed=7)
    n = len(Ys[0]); a, _b2 = split3(n)
    val0, test0, par0 = fit_arm(Ys, X, base, dev)

    def contributions(par, mask=None):
        """Per-topic per-body TRAIN variance contribution Var(w_i * z_i) under the fitted params."""
        C = np.zeros((len(Ys), NBX))
        for i in range(len(Ys)):
            z = np.sinc(np.deg2rad((X[:a] - par["p"][i] + 180.0) % 360.0 - 180.0) * par["f"][i][None, :])
            C[i] = (par["w"][i][None, :] * z).std(0)
            if mask is not None:
                C[i] *= mask[i]
        return C

    def top_mask(C, k):
        m = np.zeros_like(C)
        idx = np.argsort(-C, axis=1)[:, :k]
        for i in range(C.shape[0]):
            m[i, idx[i]] = 1.0
        return m

    C0 = contributions(par0)
    for k in (8, 6, 4, 2):
        mk = top_mask(C0, k)
        v, t, _ = fit_arm(Ys, X, dict(base, name=f"prune12->{k}"), dev, body_mask=mk)
        rows.append(dict(arm=f"prune 12->{k}", median_val=float(np.median(v)), median_test=float(np.median(t)),
                         clamped_mean_test=float(np.clip(t, 0, 1).mean() * 100),
                         frac_test_pos=float((t > 0).mean())))
        print(f"  prune 12->{k}:  val_med {np.median(v):7.3f} · TEST med {np.median(t):7.3f} · clamped {np.clip(t,0,1).mean()*100:5.1f}", flush=True)
    # multi-stage 12 -> 8 -> 4
    m8 = top_mask(C0, 8)
    v8, t8, par8 = fit_arm(Ys, X, dict(base, name="stage8"), dev, body_mask=m8)
    C8 = contributions(par8, m8)
    m4 = top_mask(C8, 4)
    v4, t4, _ = fit_arm(Ys, X, dict(base, name="stage4"), dev, body_mask=m4)
    rows.append(dict(arm="prune 12->8->4", median_val=float(np.median(v4)), median_test=float(np.median(t4)),
                     clamped_mean_test=float(np.clip(t4, 0, 1).mean() * 100),
                     frac_test_pos=float((t4 > 0).mean())))
    print(f"  prune 12->8->4: val_med {np.median(v4):7.3f} · TEST med {np.median(t4):7.3f} · clamped {np.clip(t4,0,1).mean()*100:5.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("analysis/adstopics/experiments_results.csv", index=False)
    # harmonic+trend closed-form reference
    n = len(Ys[0]); a, b = split3(n)
    m = np.arange(n); tau = (m - n / 2.0) / (n / 2.0)
    H = np.column_stack([np.ones(n), np.cos(2 * np.pi * m / 12), np.sin(2 * np.pi * m / 12), tau, tau ** 2])
    ht = []
    for y in Ys:
        coef, *_ = np.linalg.lstsq(H[:b], y[:b], rcond=None)
        pr = H @ coef
        sst = max(((y[b:] - y[b:].mean()) ** 2).sum(), 1e-9)
        ht.append(1 - ((y[b:] - pr[b:]) ** 2).sum() / sst)
    print(f"  {'harmonic+trend (closed form)':34s} "
          f"TEST med {np.median(ht):7.3f} · clamped {np.clip(ht,0,1).mean()*100:5.1f} · >0: {(np.array(ht)>0).mean()*100:4.1f}%")
    json.dump({"n_topics": len(Ys), "arms": rows,
               "harmonic_ref": {"mean_test": float(np.mean(ht)), "median_test": float(np.median(ht))}},
              open("analysis/adstopics/experiments_summary.json", "w"), indent=1)

if __name__ == "__main__":
    main()
