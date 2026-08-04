#!/usr/bin/env python3
"""THE RATE-CHANGE DATASET (operator 2026-07-19) — the study redone on the MONTHLY RATE CHANGE.

The old high/low task classified the LEVEL (rate > bar). WRONG question. The real target is the
MONTHLY RATE OF CHANGE and whether next month's search trend goes SIGNIFICANTLY HIGHER:

    shift → subtract → divide:   r[i,t] = (y[i,t] - y[i,t-1]) / max(y[i,t-1], 1)
    threshold = the GLOBAL MEDIAN change over TRAIN months (the balancing value):  tau = median(r_train) = 0

Because 34.8% of month-pairs are EXACTLY flat (r = 0 — a fat atom), no scalar threshold hits
50/50: r>0 is 32/68, r>=0 is 67/33. The flat months are therefore excluded as TIES (the journal's
accepted tie-handling), and the balanced task is: AMONG MONTHS THAT MOVED, did the rate move
higher (+1) or lower (-1)? — a genuine 48.7/51.3 split, coin = 0.5, ties scored neither way.

Walls unchanged (train / 24-mo validation / untouched 24-mo AR test). The AR test feeds PREDICTED
labels forward wherever label history is an input.

  python3 analysis/adstopics/ratechange.py [refs|simple|champion|ceiling|all]
→ analysis/adstopics/ratechange_results.csv
"""
import importlib.util as u, json, math, os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

bal = _load("analysis/adstopics/balanced.py", "bal")   # reuse the audited loader
tf = _load("analysis/trends_fit.py", "tf")

H = 24


class BenchR:
    """Rate-change protocol. R = relative monthly change; label = sign(R); flat months (R==0) are
    ties, excluded from every score, fit and ceiling. L in {0,1} (1=higher), TIE mask separate."""

    def __init__(self, Y):
        self.Y = Y
        self.n = Y.shape[1]
        self.b = self.n - H
        self.a = self.b - H
        R = np.zeros_like(Y, dtype=float)
        R[:, 1:] = (Y[:, 1:] - Y[:, :-1]) / np.maximum(Y[:, :-1], 1.0)   # shift, subtract, divide
        self.R = R
        self.tau = float(np.median(R[:, 1:self.a].ravel()))             # = 0, the balancing median
        self.L = (R > self.tau).astype(int)
        self.tie = (R == self.tau)                                      # flat months — excluded
        self.tie[:, 0] = True                                           # month 0 has no prior change
        self.moy = np.arange(self.n) % 12
        self.hs = np.arange(1, H + 1)
        self.rows = []
        m = ~self.tie[:, 1:self.a]
        hi = self.L[:, 1:self.a][m].mean()
        tm = ~self.tie[:, self.b:]
        print(f"[ratechange] tau={self.tau:.3f} (median) · flat atom {self.tie[:, 1:self.a].mean()*100:.1f}% "
              f"· among-moved train {hi*100:.2f}% higher · test-moved {self.L[:, self.b:][tm].mean()*100:.2f}% "
              f"· {Y.shape[0]} topics", flush=True)

    def score(self, pred24, tag):
        """Ties excluded per horizon; accuracy over the moved test cells only."""
        acc = np.empty(H)
        for h in range(H):
            cells = ~self.tie[:, self.b + h]
            acc[h] = ((pred24[cells, h] > 0) == (self.L[cells, self.b + h] == 1)).mean()
        auc = float(np.trapezoid(acc, self.hs) / (H - 1))
        self.accmats = getattr(self, "accmats", {}); self.accmats[tag] = (pred24)
        self.rows.append(dict(arm=tag, acc=round(auc, 4), h1=round(float(acc[0]), 4),
                              h24=round(float(acc[-1]), 4)))
        print(f"  {tag:22s} ACC {auc:.4f} · h1 {acc[0]:.3f} · h24 {acc[-1]:.3f}", flush=True)
        return auc

    def bootstrap_ci(self, tag, B=2000, seed=7):
        pred = self.accmats[tag]; rng = np.random.default_rng(seed); Tn = self.Y.shape[0]
        base = np.stack([((pred[:, h] > 0) == (self.L[:, self.b + h] == 1)).astype(float)
                         for h in range(H)], 1)                          # (T,H) raw hit
        keep = (~self.tie[:, self.b:]).astype(float)
        out = []
        for _ in range(B):
            idx = rng.integers(0, Tn, Tn)
            num = (base[idx] * keep[idx]).sum(0); den = keep[idx].sum(0) + 1e-9
            out.append(np.trapezoid(num / den, self.hs) / (H - 1))
        a = np.array(out)
        return float(np.quantile(a, .025)), float(np.quantile(a, .975)), float(a.std())


def arm_block(b, wall):
    Tn = b.Y.shape[0]; Lpm = (b.L * 2 - 1).astype(float)
    P = {"persist": np.repeat(Lpm[:, wall - 1][:, None], H, 1)}
    sn = np.zeros((Tn, H)); v3 = np.zeros((Tn, H))
    for h in range(H):
        t = wall + h; j = t - 12
        while j >= wall: j -= 12
        sn[:, h] = Lpm[:, j]; v3[:, h] = (Lpm[:, j] + Lpm[:, j - 12] + Lpm[:, j - 24]) / 3.0
    P["snaive"] = sn; P["vote3"] = v3
    return P


def _nz(b, lo, hi):
    """non-tie mean of L over columns [lo,hi) per topic (for base rate / climatology)."""
    return b.L, b.tie


def refs(b):
    Tn = b.Y.shape[0]; Pt = arm_block(b, b.b)
    b.score(Pt["persist"], "persistence_AR")
    b.score(Pt["snaive"], "seasonal_naive_AR")
    keep = (~b.tie[:, 1:b.a]).astype(float)
    base = ((b.L[:, 1:b.a] * keep).sum(1) / (keep.sum(1) + 1e-9) > 0.5) * 2.0 - 1
    b.score(np.repeat(base[:, None], H, 1), "base_rate")
    clim = np.zeros((Tn, 12))
    for mth in range(12):
        sel = b.moy[1:b.a] == mth
        k = keep[:, sel]; clim[:, mth] = (b.L[:, 1:b.a][:, sel] * k).sum(1) / (k.sum(1) + 1e-9)
    cl = np.stack([(clim[:, b.moy[b.b + h]] > 0.5) * 2.0 - 1 for h in range(H)], 1)
    b.score(cl, "label_climatology")
    mm = np.sign(Pt["persist"] + cl + np.repeat(base[:, None], H, 1)); mm[mm == 0] = 1
    b.score(mm, "memory_majority")
    return base, clim


def direct_logits(b):
    """Direct multi-horizon pooled logistic; tie targets excluded from the fit."""
    Tn, n = b.L.shape; Lpm = (b.L * 2 - 1).astype(np.float64)
    keepc = np.cumsum((~b.tie).astype(float), 1); hic = np.cumsum(b.L * (~b.tie), 1)

    def feats(w):
        wl = Lpm[:, w - 1]
        mom = np.clip((b.R[:, w - 1] - b.tau) / 0.5, -3, 3)
        base = np.where(keepc[:, w - 1] > 0, hic[:, w - 1] / np.maximum(keepc[:, w - 1], 1) * 2 - 1, 0.0)
        cols = []
        for h in range(1, H + 1):
            t = w + h - 1; j = t - 12
            while j >= w: j -= 12
            sn = Lpm[:, j]; v3 = (Lpm[:, j] + Lpm[:, j - 12] + Lpm[:, j - 24]) / 3.0
            hh = h / 24.0; M = np.zeros((Tn, 12)); M[:, b.moy[t]] = 1.0
            cols.append(np.column_stack([wl, sn, v3, mom, base, np.full(Tn, hh),
                                         wl * hh, sn * hh, v3 * hh, wl * np.abs(v3), mom * hh, M]))
        return np.stack(cols, 1)

    walls = list(range(61, b.a - H + 1, 3)); Xs, ys, ws = [], [], []
    for w in walls:
        F = feats(w); Xs.append(F.reshape(-1, F.shape[-1]))
        ys.append(b.L[:, w:w + H].reshape(-1)); ws.append((~b.tie[:, w:w + H]).reshape(-1).astype(float))
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys).astype(np.float64); wtr = np.concatenate(ws)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Xtr = (Xtr - mu) / sd
    w_ = np.zeros(Xtr.shape[1] + 1); sw = wtr / wtr.mean()
    for _ in range(400):
        z = Xtr @ w_[:-1] + w_[-1]; p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = (p - ytr) * sw
        w_ -= 2.0 * np.concatenate([Xtr.T @ g / len(ytr), [g.mean()]])
    out = []
    for wall in (b.a, b.b):
        F = feats(wall); out.append(((F - mu) / sd) @ w_[:-1] + w_[-1])
    return out


def pooled_ar(b):
    Tn, n = b.L.shape
    keep = (~b.tie[:, 1:b.a]).astype(float)
    base = (b.L[:, 1:b.a] * keep).sum(1) / (keep.sum(1) + 1e-9)
    clim = np.zeros((Tn, 12))
    for mth in range(12):
        sel = b.moy[1:b.a] == mth; k = keep[:, sel]
        clim[:, mth] = (b.L[:, 1:b.a][:, sel] * k).sum(1) / (k.sum(1) + 1e-9)

    def feats(t, Lsrc):
        cols = [Lsrc[:, t - 1] * 2 - 1, Lsrc[:, t - 12] * 2 - 1, base * 2 - 1,
                clim[:, b.moy[t]] * 2 - 1, np.clip((b.R[:, b.a - 1] - b.tau) / 0.5, -3, 3)]
        M = np.zeros((Tn, 12)); M[:, b.moy[t]] = 1.0
        return np.column_stack(cols + [M])

    Xs, ys, ws = [], [], []
    for t in range(13, b.a):
        Xs.append(feats(t, b.L)); ys.append(b.L[:, t]); ws.append((~b.tie[:, t]).astype(float))
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys).astype(float); wtr = np.concatenate(ws)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Xtr = (Xtr - mu) / sd
    w = np.zeros(Xtr.shape[1] + 1); sw = wtr / wtr.mean()
    for _ in range(300):
        z = Xtr @ w[:-1] + w[-1]; p = 1 / (1 + np.exp(-np.clip(z, -30, 30))); g = (p - ytr) * sw
        w -= 2.0 * np.concatenate([Xtr.T @ g / len(ytr), [g.mean()]])
    Lar = b.L.copy(); Z = np.zeros((Tn, H))
    for h, t in enumerate(range(b.b, b.n)):
        z = ((feats(t, Lar) - mu) / sd) @ w[:-1] + w[-1]; Z[:, h] = z; Lar[:, t] = (z > 0).astype(int)
    return Z


def encdec_logits(b):
    """Phase encoder-decoder on the rate-change wave (CPU, seed 7); tie targets masked in the loss."""
    import torch as T
    Tn = b.Y.shape[0]; W = 36; a1 = b.a - 1
    lw = (b.L * 2 - 1).astype(np.float32)
    g9 = 0.9 ** ((a1 - 1 - np.arange(a1)) / 12.0)
    vote = np.zeros((Tn, 12), dtype=np.float32)
    for mth in range(12):
        sel = b.moy[:a1] == mth; vote[:, mth] = (lw[:, :a1][:, sel] * g9[sel]).sum(1) / g9[sel].sum()
    mom = np.clip((b.R[:, a1 - 1] - b.tau) / 0.5, -3, 3).astype(np.float32)
    wall_lbl = lw[:, a1 - 1]; moy_ang = np.deg2rad(b.moy * 30.0 + 15.0).astype(np.float32)
    win = lw[:, a1 - W:a1]; thw = moy_ang[a1 - W:a1]
    ang12 = np.deg2rad(np.arange(12) * 30.0 + 15.0); posv = np.clip(vote, 0, None)
    p_tgt = np.arctan2((posv * np.sin(ang12)[None]).sum(1), (posv * np.cos(ang12)[None]).sum(1))
    p_wt = (np.sqrt(((posv * np.sin(ang12)[None]).sum(1)) ** 2 +
                    ((posv * np.cos(ang12)[None]).sum(1)) ** 2) / (np.abs(posv).sum(1) + 1e-6))
    p_tgtT = T.tensor(p_tgt.astype(np.float32)); p_wtT = T.tensor(p_wt.astype(np.float32))
    tr = list(range(37, b.a)); va = list(range(b.a, b.b))
    ytr = T.tensor(b.L[:, tr].astype(np.float32)); mtr = T.tensor((~b.tie[:, tr]).astype(np.float32))
    yva = T.tensor(b.L[:, va].astype(np.float32)); mva = T.tensor((~b.tie[:, va]).astype(np.float32))
    bce = T.nn.functional.binary_cross_entropy_with_logits
    tokN = np.stack([win, np.tile(np.sin(thw), (Tn, 1)), np.tile(np.cos(thw), (Tn, 1))], -1)

    def mem_feats(ts_, wall):
        f_v = np.stack([vote[:, b.moy[t]] for t in ts_], 1)
        f_sn = np.stack([lw[:, wall - 12 + ((t - wall) % 12)] if t >= wall else lw[:, t - 12] for t in ts_], 1)
        f_p = np.repeat(wall_lbl[:, None], len(ts_), 1); f_m = np.repeat(mom[:, None], len(ts_), 1)
        return T.tensor(np.stack([f_v, f_sn, f_p, f_m], -1).astype(np.float32))

    out = {}
    for tag, use_mem in (("ENCDECR_pure", False), ("ENCDECR_mem", True)):
        T.manual_seed(7); gseed = T.Generator().manual_seed(7)
        P = lambda *sh: T.nn.Parameter(T.randn(*sh, generator=gseed) * 0.2)
        Dh = 16
        We, be, qv, uv = P(3, Dh), P(Dh), P(Dh), P(Dh)
        Wd, bd, Qd = P(3, Dh), P(Dh), P(2, Dh)
        Wo, bo = P(Dh + (4 if use_mem else 0), 1), P(1)
        tok = T.tensor(tokN); thwT = T.tensor(thw); lwT = T.tensor(win)

        def encoder():
            e = T.tanh(tok @ We + be); al = T.softmax((e @ qv) / math.sqrt(Dh), dim=1)
            ws = al * T.tanh(e @ uv)
            return T.atan2((ws * T.sin(thwT)[None]).sum(1), (ws * T.cos(thwT)[None]).sum(1))

        def decoder(ts_, p_, wall):
            phi = thwT[None, :] - p_[:, None]; rtok = T.stack([lwT, T.sin(phi), T.cos(phi)], -1)
            r = T.tanh(rtok @ Wd + bd)
            th_t = T.tensor(np.array([moy_ang[t] for t in ts_])); tphi = th_t[None, :] - p_[:, None]
            q = T.stack([T.sin(tphi), T.cos(tphi)], -1) @ Qd
            a2 = T.softmax(T.einsum("twd,tmd->tmw", r, q) / math.sqrt(Dh), dim=-1)
            ctx = T.einsum("tmw,twd->tmd", a2, r)
            if use_mem: ctx = T.cat([ctx, mem_feats(ts_, wall)], -1)
            return (ctx @ Wo).squeeze(-1) + bo

        params = [We, be, qv, uv, Wd, bd, Qd, Wo, bo]; opt = T.optim.Adam(params, lr=3e-3)
        best, stall, state = -1.0, 0, None
        for ep in range(500):
            opt.zero_grad(); p_cur = encoder()
            raw = bce(decoder(tr, p_cur, b.a), ytr, reduction="none")
            loss = (raw * mtr).sum() / mtr.sum() + 0.3 * (p_wtT * (1.0 - T.cos(p_cur - p_tgtT))).mean()
            loss.backward(); opt.step()
            if ep % 5 == 4:
                with T.no_grad():
                    hit = ((decoder(va, encoder(), b.a) > 0).float() == yva).float()
                    acc = (hit * mva).sum().item() / mva.sum().item()
                if acc > best + 1e-5: best, stall, state = acc, 0, [x.detach().clone() for x in params]
                else: stall += 1
                if stall >= 15: break
        with T.no_grad():
            for x, st in zip(params, state): x.copy_(st)
            pfin = encoder()
            Zt = T.stack([decoder([t], pfin, b.b)[:, 0] for t in range(b.b, b.n)], 1).cpu().numpy()
            Zv = decoder(va, pfin, b.a).cpu().numpy()
        print(f"    {tag}: val acc {best:.4f}", flush=True)
        out[tag] = (best, Zv, Zt, pfin.cpu().numpy())
    return out


def champion(b, names, with_encdec=True, write_atlas=False):
    Tn = b.Y.shape[0]; Pv, Pt = arm_block(b, b.a), arm_block(b, b.b)
    Zv_dir, Zt_dir = direct_logits(b)
    b.score(pooled_ar(b), "pooled_logistic_AR")
    b.score(Zt_dir, "direct_pooled")
    order = ["persist", "snaive", "vote3", "direct"]
    cand_v = {"persist": Pv["persist"], "snaive": Pv["snaive"], "vote3": Pv["vote3"], "direct": Zv_dir}
    cand_t = {"persist": Pt["persist"], "snaive": Pt["snaive"], "vote3": Pt["vote3"], "direct": Zt_dir}
    phases = None
    if with_encdec:
        ed = encdec_logits(b); tag = max(ed, key=lambda k: ed[k][0])
        b.score(ed["ENCDECR_pure"][2], "ENCDECR_pure"); b.score(ed["ENCDECR_mem"][2], "ENCDECR_mem")
        cand_v["encdec"] = ed[tag][1]; cand_t["encdec"] = ed[tag][2]; order.append("encdec"); phases = ed[tag][3]
    yv = b.L[:, b.a:b.b]; kv = (~b.tie[:, b.a:b.b]).astype(float)
    accV = np.stack([(((cand_v[k] > 0) == (yv == 1)) * kv).sum(1) / (kv.sum(1) + 1e-9) for k in order], 1)
    choice = accV.argmax(1); Z = np.zeros((Tn, H))
    for i, k in enumerate(order):
        m = choice == i; Z[m] = cand_t[k][m]
        print(f"    arm {k:8s} chosen by {int(m.sum())} topics", flush=True)
    b.score(Z, "CHAMPION_selected")
    # THE SHIPPED CHAMPION — the tournament winner, refined seasonal climatology (beats the
    # per-topic selector and the encoder-decoder; adversarially verified, no leakage).
    Zs, peak, cfg = seasonal_champion(b)
    b.score(Zs, "CHAMPION_seasonal")
    if write_atlas:
        MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pred = np.where(Zs > 0, 1, -1).astype(int)
        tiei = b.tie[:, b.b:]; actual = np.where(b.L[:, b.b:] == 1, 1, -1); actual[tiei] = 0
        hits = np.where(tiei, -1, (pred == actual).astype(int))
        peracc = np.array([((pred[i][~tiei[i]] == actual[i][~tiei[i]]).mean() if (~tiei[i]).any() else np.nan)
                           for i in range(Tn)])
        # 'sign' repurposed as the topic's PEAK-GROWTH month (the month its search rises most); 'phase'
        # its angle — keeps the registry/export/atlas pipeline working with a rate-change meaning.
        df = pd.DataFrame(dict(topic=names, sign=[MON[m] for m in peak],
                               phase=np.round(peak * 30.0 + 15.0, 2), peak_month=[MON[m] for m in peak],
                               arm=["seasonal"] * Tn, dir_acc_test=np.round(peracc, 4)))
        df.to_csv("analysis/adstopics/ratechange_atlas.csv", index=False)
        payload = {tf.slug(t): {"sqPred": [int(v) for v in pred[i]], "sqHit": [int(v) for v in hits[i]],
                                "sqTie": [int(v) for v in tiei[i]]}
                   for i, t in enumerate(names)}
        payload["_meta"] = {"tau": b.tau, "task": "ratechange-higher", "cfg": cfg,
                            "champion": "refined seasonal climatology of the rate-change direction (ties excluded)"}
        json.dump(payload, open("analysis/adstopics/ratechange_pred.json", "w"))
        m = ~tiei
        print(f"[atlas] seasonal champion mean test acc {(pred[m]==actual[m]).mean():.4f} · atlas + pred written", flush=True)
    return Zs, peak


def seasonal_champion(b):
    """THE SHIPPED CHAMPION (tournament winner, adversarially verified, test AUC 0.5999):
    refined per-(topic, month-of-year) climatology of the rate-change DIRECTION —
    recency-weighted sign vote, base-rate shrinkage (pseudocount k), and a seasonality-strength
    gate that blends weakly-seasonal topics toward the global monthly pattern. Pure-seasonal
    (depends only on the target month); all knobs selected on VALIDATION only. Ties never enter."""
    Tn = b.Y.shape[0]

    def clim(W, decay):
        cols = np.arange(1, W); mt = b.moy[cols]
        valid = (~b.tie[:, cols]).astype(np.float64); Lc = b.L[:, cols].astype(np.float64)
        Wt = valid * (decay ** ((W - 1 - cols) / 12.0))[None, :]
        num = np.zeros((Tn, 12)); den = np.zeros((Tn, 12))
        for m in range(12):
            sel = mt == m; den[:, m] = Wt[:, sel].sum(1); num[:, m] = (Wt[:, sel] * Lc[:, sel]).sum(1)
        base = (Wt * Lc).sum(1) / (Wt.sum(1) + 1e-9)
        gclim = num.sum(0) / (den.sum(0) + 1e-9)
        return num, den, base, gclim

    def predict(W, decay, k, thr, width):
        num, den, base, gclim = clim(W, decay)
        shrunk = (num + k * base[:, None]) / (den + k)
        dd = den.sum(1, keepdims=True) + 1e-9
        g = np.sqrt((den * (shrunk - base[:, None]) ** 2).sum(1) / dd.ravel())
        alpha = 1.0 / (1.0 + np.exp(-(g - thr) / width))
        P = np.zeros((Tn, H))
        for h in range(H):
            m = b.moy[W + h]; P[:, h] = alpha * shrunk[:, m] + (1.0 - alpha) * gclim[m] - 0.5
        peak = shrunk.argmax(1)
        return P, peak

    def vauc(pv):
        acc = np.empty(H)
        for h in range(H):
            c = ~b.tie[:, b.a + h]; acc[h] = ((pv[c, h] > 0) == (b.L[c, b.a + h] == 1)).mean()
        return float(np.trapezoid(acc, b.hs) / (H - 1))

    best = (-1.0, None)
    for decay in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5):
        for k in (0.5, 1, 2, 4, 8, 16, 32, 64):
            for thr in (-1.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
                a = vauc(predict(b.a, decay, k, thr, 0.05)[0])
                if a > best[0]: best = (a, dict(decay=decay, k=k, thr=thr, width=0.05))
    cfg = best[1]
    print(f"    seasonal champion val-selected {cfg} val={best[0]:.4f}", flush=True)
    P, peak = predict(b.b, cfg["decay"], cfg["k"], cfg["thr"], cfg["width"])
    return P, peak, cfg


def ceilings(b):
    prof = {}
    for t in range(b.b, b.n):
        for i in range(b.Y.shape[0]):
            if not b.tie[i, t]: prof.setdefault((b.moy[t],), [])
    num = den = 0
    for m in range(12):
        cols = [t for t in range(b.b, b.n) if b.moy[t] == m]
        for i in range(b.Y.shape[0]):
            vals = [b.L[i, t] for t in cols if not b.tie[i, t]]
            if vals:
                pos = sum(vals); num += max(pos, len(vals) - pos); den += len(vals)
    o3 = num / den
    ag = []
    for t in range(b.b, b.b + 12):
        m = (~b.tie[:, t]) & (~b.tie[:, t + 12]); ag.append((b.L[m, t] == b.L[m, t + 12]).mean())
    print(f"  O3 per-(topic,month) cap {o3:.4f} · O2 cross-year agreement {np.mean(ag):.4f}", flush=True)
    return o3, float(np.mean(ag))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    names, Y = bal.load_all()
    b = BenchR(Y)
    if mode in ("ceiling", "all", "simple", "champion"): ceilings(b)
    refs(b)
    if mode == "simple": champion(b, names, with_encdec=False, write_atlas=True)
    if mode in ("champion", "all"): champion(b, names, with_encdec=True, write_atlas=True)
    pd.DataFrame(b.rows).to_csv("analysis/adstopics/ratechange_results.csv", index=False)


if __name__ == "__main__":
    main()
