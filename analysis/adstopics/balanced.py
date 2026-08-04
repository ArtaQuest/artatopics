#!/usr/bin/env python3
"""THE BALANCED DATASET (operator 2026-07-17) — the study redone on a 50/50 target.

Label(topic, month) = [ monthly rate > tau ], where tau is THE global threshold that splits the
whole dataset into 50/50 classes: the median of the monthly interest rate over ALL topics x
TRAIN months (train-side only — no leakage; the achieved split is reported). The series becomes
a HIGH/LOW square wave. Walls unchanged (train / 24-mo validation / untouched 24-mo AR test);
the AR test feeds PREDICTED labels forward wherever label history is an input.

Rerun of the whole program on the new target:
  refs     : persistence (last month's label), seasonal naive (t-12), per-topic base rate,
             label climatology, majority
  models   : pooled logistic · encoder-decoder with GD phase on the high/low wave (the study's
             core architecture, retargeted)
  ceiling  : test-window oracle + cross-year agreement on the new labels

  python3 analysis/adstopics/balanced.py [refs|pooled|encdec|ceiling|all]
→ analysis/adstopics/balanced_results.csv
"""
import importlib.util as u, json, math, os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

tf = _load("analysis/trends_fit.py", "tf")
r5 = _load("analysis/topic500_reference_solution.py", "r5")

H = 24


def load_all():
    grid = pd.DatetimeIndex(tf.GRID)
    i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
    iend = len(grid) - tf.DROP_LAST
    vocab = json.load(open("analysis/adstopics/vocabulary.json"))
    bl = set(json.load(open("analysis/adstopics/blacklist.json")).get("excluded_topics", []))
    names, Ys = [], []
    for t in sorted(vocab):
        if t in bl: continue
        p = f"analysis/adstopics/series/{tf.slug(t)}.csv"
        if not os.path.exists(p): continue
        df = pd.read_csv(p); df["Time"] = pd.to_datetime(df["Time"])
        v = pd.to_numeric(df.drop_duplicates("Time").set_index("Time")["v"]
                          .reindex(grid[i0:iend]), errors="coerce")
        if v.notna().sum() < (iend - i0) * 0.5: continue
        y = v.interpolate(limit_direction="both").to_numpy(float)
        if not np.isfinite(y).all() or y.max() <= 0: continue
        names.append(t); Ys.append(y)
    return names, np.stack(Ys)


class BenchB:
    """The balanced-label protocol. L[t, m] in {0,1}; tau from TRAIN cells only."""

    def __init__(self, Y):
        self.Y = Y
        self.n = Y.shape[1]
        self.b = self.n - H
        self.a = self.b - H
        train_cells = Y[:, :self.a].ravel()
        self.tau = float(np.median(train_cells))
        self.L = (Y > self.tau).astype(int)
        tr_split = self.L[:, :self.a].mean()
        te_split = self.L[:, self.b:].mean()
        self.moy = np.arange(self.n) % 12
        self.hs = np.arange(1, H + 1)
        self.rows = []
        print(f"[balanced] tau = {self.tau:.1f} · train split {tr_split*100:.2f}% high · "
              f"test split {te_split*100:.2f}% high · {Y.shape[0]} topics")

    def curve(self, dirfun):
        acc = []
        for h in range(1, H + 1):
            t = self.b + h - 1
            acc.append(float((dirfun(t) == (self.L[:, t] == 1)).mean()))
        return np.array(acc)

    def score(self, dirfun, tag):
        acc = self.curve(dirfun)
        auc = float(np.trapezoid(acc, self.hs) / (H - 1))
        self.rows.append(dict(arm=tag, auc=round(auc, 4), acc_h1=round(acc[0], 4),
                              acc_mean=round(float(acc.mean()), 4), acc_h24=round(acc[-1], 4)))
        print(f"  {tag:20s} ACC {auc:.4f} · h1 {acc[0]:.3f} · h24 {acc[-1]:.3f}", flush=True)
        return acc


def refs(b):
    # AR-CORRECT persistence: beyond the wall, persistence = the last TRUE pre-wall label
    # carried forward through its own predictions => constant per topic (the wall label).
    wall = b.L[:, b.b - 1] == 1
    b.score(lambda t: wall, "persistence_AR")
    base = b.L[:, :b.a].mean(1) > 0.5
    b.score(lambda t: base, "base_rate")
    def snaive_ar(t):
        # tiled last observed pre-wall year: label at the same month-of-year, 12 back (wall-tiled)
        j = t - 12
        while j >= b.b: j -= 12
        return b.L[:, j] == 1
    b.score(snaive_ar, "seasonal_naive_AR")
    clim = np.zeros((b.Y.shape[0], 12))
    for m in range(12):
        sel = (b.moy[:b.a] == m)
        clim[:, m] = b.L[:, :b.a][:, sel].mean(1)
    b.score(lambda t: clim[:, b.moy[t]] > 0.5, "label_climatology")
    b.score(lambda t: (wall.astype(int) + (clim[:, b.moy[t]] > 0.5) + base) >= 2, "memory_majority")
    return wall, base, clim


def pooled(b, wall, base, clim):
    Tn = b.Y.shape[0]

    def feats(t, Lsrc):
        cols = [Lsrc[:, t - 1] * 2 - 1, Lsrc[:, t - 12] * 2 - 1, base.astype(float) * 2 - 1,
                clim[:, b.moy[t]] * 2 - 1,
                np.clip((b.Y[:, b.a - 1] - b.tau) / (b.tau + 1), -3, 3)]
        M = np.zeros((Tn, 12)); M[:, b.moy[t]] = 1.0
        return np.column_stack(cols + [M])

    Xs, ys = [], []
    for t in range(13, b.a):
        Xs.append(feats(t, b.L)); ys.append(b.L[:, t])
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys).astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    w = np.zeros(Xtr.shape[1] + 1)
    for _ in range(300):
        z = Xtr @ w[:-1] + w[-1]
        p = 1 / (1 + np.exp(-z))
        w -= 2.0 * np.concatenate([Xtr.T @ (p - ytr) / len(ytr), [(p - ytr).mean()]])
    # AR rollout: predicted labels feed the lag features forward
    Lar = b.L.copy()
    lo = {}
    for t in range(b.b, b.n):
        z = ((feats(t, Lar) - mu) / sd) @ w[:-1] + w[-1]
        lo[t] = z
        Lar[:, t] = (z > 0).astype(int)
    b.score(lambda t: lo[t] > 0, "pooled_logistic_AR")
    return lo


def simple(b):
    """Per-topic arm selection on VALIDATION among AR-safe memory arms."""
    Tn = b.Y.shape[0]

    def arm_preds(wall, gamma):
        """Each arm -> (Tn, H) predictions for months [wall, wall+H), using data < wall only."""
        P = {}
        P["persist"] = np.repeat(b.L[:, wall - 1][:, None], H, axis=1)
        P["snaive"] = np.stack([b.L[:, wall - 12 + (h % 12)] for h in range(H)], axis=1)
        vote = np.zeros((Tn, H))
        for h in range(H):
            num = den = 0.0
            for k, wgt in enumerate([1.0, gamma, gamma * gamma]):
                idx = wall + h - 12 * (k + 1)
                while idx >= wall: idx -= 12
                num = num + wgt * b.L[:, idx]; den += wgt
            vote[:, h] = num / den
        P["vote"] = (vote > 0.5).astype(int)
        return P

    # global gamma picked on validation for the vote arm
    val_true = b.L[:, b.a:b.b]
    best_g, best_acc = 1.0, -1
    for g in (1.0, 0.8, 0.6, 0.4):
        acc = (arm_preds(b.a, g)["vote"] == val_true).mean()
        if acc > best_acc: best_acc, best_g = acc, g
    print(f"  vote gamma={best_g} (val {best_acc:.4f})")

    Pval = arm_preds(b.a, best_g)
    order = ["persist", "snaive", "vote"]
    val_acc = np.stack([(Pval[k] == val_true).mean(1) for k in order], axis=1)
    choice = val_acc.argmax(1)  # argmax ties -> earliest = persist
    Ptest = arm_preds(b.b, best_g)
    sel = np.zeros((Tn, H), dtype=int)
    for i, k in enumerate(order):
        m = choice == i
        sel[m] = Ptest[k][m]
        print(f"    arm {k:8s} chosen by {int(m.sum())} topics")
    b.score(lambda t: Ptest["vote"][:, t - b.b] == 1, f"same_month_vote_g{best_g}")
    b.score(lambda t: sel[:, t - b.b] == 1, "per_topic_selected")
    return sel


def encdec(b, names):
    """The study's core architecture retargeted: encoder estimates the phase of the HIGH/LOW
    wave; decoder classifies the rotated data. Memory features strictly AR-safe (pre-wall
    only: wall label, tiled seasonal naive, recency vote profile, wall level vs tau)."""
    import torch as T
    dev_ = r5._device()
    Tn = b.Y.shape[0]
    W = 36
    a1 = b.a - 1
    lw = (b.L * 2 - 1).astype(np.float32)            # the ±1 high/low wave
    # recency-weighted per-month-of-year profile (pre-wall)
    g9 = 0.9 ** ((a1 - 1 - np.arange(a1)) / 12.0)
    vote = np.zeros((Tn, 12), dtype=np.float32)
    for mth in range(12):
        sel = b.moy[:a1] == mth
        vote[:, mth] = (lw[:, :a1][:, sel] * g9[sel]).sum(1) / g9[sel].sum()
    lvl = np.clip((b.Y[:, a1 - 1] - b.tau) / (b.tau + 1.0), -3, 3).astype(np.float32)
    wall_lbl = lw[:, a1 - 1]
    moy_ang = np.deg2rad(b.moy * 30.0 + 15.0).astype(np.float32)
    win = lw[:, a1 - W:a1]
    thw = moy_ang[a1 - W:a1]
    ang12 = np.deg2rad(np.arange(12) * 30.0 + 15.0)
    posv = np.clip(vote, 0, None)
    p_tgt = np.arctan2((posv * np.sin(ang12)[None]).sum(1), (posv * np.cos(ang12)[None]).sum(1))
    p_wt = (np.sqrt(((posv * np.sin(ang12)[None]).sum(1)) ** 2 +
                    ((posv * np.cos(ang12)[None]).sum(1)) ** 2) / (np.abs(posv).sum(1) + 1e-6))
    p_tgtT = T.tensor(p_tgt.astype(np.float32), device=dev_)
    p_wtT = T.tensor(p_wt.astype(np.float32), device=dev_)
    tr = list(range(37, b.a)); va = list(range(b.a, b.b))
    ytr = T.tensor(b.L[:, tr].astype(np.float32), device=dev_)
    yva = T.tensor(b.L[:, va].astype(np.float32), device=dev_)
    bce = T.nn.functional.binary_cross_entropy_with_logits
    tokN = np.stack([win, np.tile(np.sin(thw), (Tn, 1)), np.tile(np.cos(thw), (Tn, 1))], -1)

    def mem_feats(ts_, wall):
        f_v = np.stack([vote[:, b.moy[t]] for t in ts_], 1)
        f_sn = np.stack([lw[:, wall - 12 + ((t - wall) % 12)] if t >= wall else lw[:, t - 12]
                         for t in ts_], 1)
        f_p = np.repeat(wall_lbl[:, None], len(ts_), 1)
        f_l = np.repeat(lvl[:, None], len(ts_), 1)
        return T.tensor(np.stack([f_v, f_sn, f_p, f_l], -1).astype(np.float32), device=dev_)

    keep = {}
    for tag, use_mem in (("ENCDECB_pure", False), ("ENCDECB_mem", True)):
        T.manual_seed(7)
        gseed = T.Generator().manual_seed(7)
        P = lambda *sh: T.nn.Parameter(T.randn(*sh, generator=gseed).to(dev_) * 0.2)
        Dh = 16
        We, be, qv, uv = P(3, Dh), P(Dh), P(Dh), P(Dh)
        Wd, bd, Qd = P(3, Dh), P(Dh), P(2, Dh)
        Wo, bo = P(Dh + (4 if use_mem else 0), 1), P(1)
        tok = T.tensor(tokN, device=dev_)
        thwT = T.tensor(thw, device=dev_)
        lwT = T.tensor(win, device=dev_)

        def encoder():
            e = T.tanh(tok @ We + be)
            al = T.softmax((e @ qv) / math.sqrt(Dh), dim=1)
            ws = al * T.tanh(e @ uv)
            return T.atan2((ws * T.sin(thwT)[None]).sum(1), (ws * T.cos(thwT)[None]).sum(1))

        def decoder(ts_, p_, wall):
            phi = thwT[None, :] - p_[:, None]
            rtok = T.stack([lwT, T.sin(phi), T.cos(phi)], -1)
            r = T.tanh(rtok @ Wd + bd)
            th_t = T.tensor(np.array([moy_ang[t] for t in ts_]), device=dev_)
            tphi = th_t[None, :] - p_[:, None]
            q = T.stack([T.sin(tphi), T.cos(tphi)], -1) @ Qd
            a2 = T.softmax(T.einsum("twd,tmd->tmw", r, q) / math.sqrt(Dh), dim=-1)
            ctx = T.einsum("tmw,twd->tmd", a2, r)
            if use_mem:
                ctx = T.cat([ctx, mem_feats(ts_, wall)], -1)
            return (ctx @ Wo).squeeze(-1) + bo

        params = [We, be, qv, uv, Wd, bd, Qd, Wo, bo]
        opt = T.optim.Adam(params, lr=3e-3)
        best, stall, state = -1.0, 0, None
        for ep in range(500):
            opt.zero_grad()
            p_cur = encoder()
            loss = bce(decoder(tr, p_cur, b.a), ytr)
            loss = loss + 0.3 * (p_wtT * (1.0 - T.cos(p_cur - p_tgtT))).mean()
            loss.backward(); opt.step()
            if ep % 5 == 4:
                with T.no_grad():
                    acc = ((decoder(va, encoder(), b.a) > 0).float() == yva).float().mean().item()
                if acc > best + 1e-5:
                    best, stall, state = acc, 0, [x.detach().clone() for x in params]
                else:
                    stall += 1
                if stall >= 15: break
        with T.no_grad():
            for x, st in zip(params, state): x.copy_(st)
            pfin = encoder()
            lo = {t: decoder([t], pfin, b.b)[:, 0].cpu().numpy() for t in range(b.b, b.n)}
            vlo = decoder(va, pfin, b.a).cpu().numpy()
        print(f"    {tag}: val acc {best:.4f}", flush=True)
        b.score(lambda t, l=lo: l[t] > 0, tag)
        keep[tag] = (best, lo, pfin.cpu().numpy(), vlo)
        d_ = pfin.cpu().numpy() - p_tgt
        strong = p_wt > np.quantile(p_wt, 0.75)
        Rg = abs(np.exp(1j * d_[strong]).mean())
        print(f"    {tag}: strong-anchor R={Rg:.3f}", flush=True)
    best_tag = max(keep, key=lambda k: keep[k][0])
    print(f"    winner on val: {best_tag}")
    return keep


def direct(b):
    """DIRECT multi-horizon pooled logistic: for every horizon h in 1..24 predict the label at
    wall+h-1 from PRE-WALL data only (no rollout, no compounding). Trained over many simulated
    walls; one model with horizon features + interactions."""
    Tn, n = b.L.shape
    Lpm = (b.L * 2 - 1).astype(np.float64)
    csum = np.cumsum(b.L, axis=1)  # for pre-wall base rate

    def feats(w):
        """(Tn, H, F) features for targets w..w+H-1 using data < w only — IDENTICAL to the
        published contract's direct_logits (bundle-balanced/reproduce.py); one code path."""
        wl = Lpm[:, w - 1]
        lvl = np.clip((b.Y[:, w - 1] - b.tau) / (b.tau + 1.0), -3, 3)
        base = (csum[:, w - 1] / w) * 2 - 1
        cols = []
        for h in range(1, H + 1):
            t = w + h - 1
            j = t - 12
            while j >= w: j -= 12
            sn = Lpm[:, j]
            v3 = (Lpm[:, j] + Lpm[:, j - 12] + Lpm[:, j - 24]) / 3.0
            hh = h / 24.0
            M = np.zeros((Tn, 12)); M[:, b.moy[t]] = 1.0
            cols.append(np.column_stack([wl, sn, v3, lvl, base,
                                         np.full(Tn, hh), wl * hh, sn * hh, v3 * hh,
                                         wl * np.abs(v3), lvl * hh, M]))
        return np.stack(cols, axis=1)  # (Tn, H, F)

    walls = list(range(61, b.a - H + 1, 3))
    Xs, ys = [], []
    for w in walls:
        F = feats(w)
        Xs.append(F.reshape(-1, F.shape[-1]))
        ys.append(b.L[:, w:w + H].reshape(-1))
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys).astype(np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    w_ = np.zeros(Xtr.shape[1] + 1)
    for _ in range(400):
        z = Xtr @ w_[:-1] + w_[-1]
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        w_ -= 2.0 * np.concatenate([Xtr.T @ (p - ytr) / len(ytr), [(p - ytr).mean()]])
    Ft = feats(b.b)
    Z = ((Ft - mu) / sd) @ w_[:-1] + w_[-1]           # (Tn, H)
    b.score(lambda t: Z[:, t - b.b] > 0, "direct_pooled")
    return Z


def direct_logits(b):
    """The direct pooled model's logits at wall=a (val) and wall=b (test)."""
    Tn, n = b.L.shape
    Lpm = (b.L * 2 - 1).astype(np.float64)
    csum = np.cumsum(b.L, axis=1)

    def feats(w):
        wl = Lpm[:, w - 1]
        lvl = np.clip((b.Y[:, w - 1] - b.tau) / (b.tau + 1.0), -3, 3)
        base = (csum[:, w - 1] / w) * 2 - 1
        cols = []
        for h in range(1, H + 1):
            t = w + h - 1
            j = t - 12
            while j >= w: j -= 12
            sn = Lpm[:, j]
            v3 = (Lpm[:, j] + Lpm[:, j - 12] + Lpm[:, j - 24]) / 3.0
            hh = h / 24.0
            M = np.zeros((Tn, 12)); M[:, b.moy[t]] = 1.0
            cols.append(np.column_stack([wl, sn, v3, lvl, base,
                                         np.full(Tn, hh), wl * hh, sn * hh, v3 * hh,
                                         wl * np.abs(v3), lvl * hh, M]))
        return np.stack(cols, axis=1)

    walls = list(range(61, b.a - H + 1, 3))
    Xs, ys = [], []
    for w in walls:
        F = feats(w)
        Xs.append(F.reshape(-1, F.shape[-1]))
        ys.append(b.L[:, w:w + H].reshape(-1))
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys).astype(np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    w_ = np.zeros(Xtr.shape[1] + 1)
    for _ in range(400):
        z = Xtr @ w_[:-1] + w_[-1]
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        w_ -= 2.0 * np.concatenate([Xtr.T @ (p - ytr) / len(ytr), [(p - ytr).mean()]])
    out = []
    for wall in (b.a, b.b):
        F = feats(wall)
        out.append(((F - mu) / sd) @ w_[:-1] + w_[-1])
    return out[0], out[1]


def champion(b, names):
    """Honest final stage. All arms emit predictions at wall=a (val) and wall=b (test).
    val-A (first 12 months) fits the per-topic selector + the pooled stacker; val-B
    (last 12) compares ALL candidates; the val-B winner ALONE touches the test window."""
    Tn = b.Y.shape[0]
    Lpm = (b.L * 2 - 1).astype(np.float64)

    def arm_block(wall):
        P = {"persist": np.repeat(Lpm[:, wall - 1][:, None], H, 1)}
        sn = np.zeros((Tn, H)); v3 = np.zeros((Tn, H))
        for h in range(H):
            t = wall + h
            j = t - 12
            while j >= wall: j -= 12
            sn[:, h] = Lpm[:, j]
            v3[:, h] = (Lpm[:, j] + Lpm[:, j - 12] + Lpm[:, j - 24]) / 3.0
        P["snaive"] = sn; P["vote3"] = v3
        return P

    Pv, Pt = arm_block(b.a), arm_block(b.b)
    Zv_dir, Zt_dir = direct_logits(b)
    keep = encdec(b, names)
    tag = max(keep, key=lambda k: keep[k][0])
    Zv_ed = keep[tag][3]
    Zt_ed = np.stack([keep[tag][1][t] for t in range(b.b, b.n)], 1)
    lvl_v = np.clip((b.Y[:, b.a - 1] - b.tau) / (b.tau + 1), -3, 3)
    lvl_t = np.clip((b.Y[:, b.b - 1] - b.tau) / (b.tau + 1), -3, 3)

    def stack_feats(P, Zd, Ze, lvl):
        cols = [P["persist"], P["snaive"], P["vote3"], np.tanh(Zd), np.tanh(Ze),
                np.abs(P["vote3"]), np.repeat(lvl[:, None], H, 1),
                np.tile((np.arange(H) / 24.0)[None, :], (Tn, 1))]
        return np.stack(cols, -1)

    Fv = stack_feats(Pv, Zv_dir, Zv_ed, lvl_v)
    Ft = stack_feats(Pt, Zt_dir, Zt_ed, lvl_t)
    yv = b.L[:, b.a:b.b].astype(np.float64)
    A, Bs = slice(0, 12), slice(12, 24)

    XA = Fv[:, A].reshape(-1, Fv.shape[-1]); yA = yv[:, A].reshape(-1)
    mu, sd = XA.mean(0), XA.std(0) + 1e-9
    XA = (XA - mu) / sd
    w_ = np.zeros(XA.shape[1] + 1)
    for _ in range(400):
        z = XA @ w_[:-1] + w_[-1]
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        w_ -= 2.0 * np.concatenate([XA.T @ (p - yA) / len(yA), [(p - yA).mean()]])
    stk = lambda F: ((F - mu) / sd) @ w_[:-1] + w_[-1]

    cand_v = {"persist": Pv["persist"], "snaive": Pv["snaive"], "vote3": Pv["vote3"],
              "direct": Zv_dir, "encdec": Zv_ed, "stacker": stk(Fv)}
    order = ["persist", "snaive", "vote3", "direct", "encdec"]
    accA = np.stack([((cand_v[k][:, A] > 0) == (yv[:, A] == 1)).mean(1) for k in order], 1)
    choice = accA.argmax(1)
    selv = np.zeros((Tn, H))
    for i, k in enumerate(order): selv[choice == i] = cand_v[k][choice == i]
    cand_v["selected"] = selv
    accB = {k: float(((v[:, Bs] > 0) == (yv[:, Bs] == 1)).mean()) for k, v in cand_v.items()}
    for k, v in sorted(accB.items(), key=lambda kv: -kv[1]):
        print(f"    val-B {k:9s} {v:.4f}", flush=True)
    win = max(accB, key=accB.get)
    print(f"    champion by val-B: {win}", flush=True)

    cand_t = {"persist": Pt["persist"], "snaive": Pt["snaive"], "vote3": Pt["vote3"],
              "direct": Zt_dir, "encdec": Zt_ed, "stacker": stk(Ft)}
    selt = np.zeros((Tn, H))
    for i, k in enumerate(order): selt[choice == i] = cand_t[k][choice == i]
    cand_t["selected"] = selt
    Zwin = cand_t[win]
    b.score(lambda t: Zwin[:, t - b.b] > 0, f"CHAMPION_{win}")
    np.save("analysis/adstopics/balanced_champion_logits.npy", Zwin)
    return win, Zwin, keep


def final(b, names):
    """THE SHIPPED CHAMPION (declared protocol, seed 7): per-topic arm selection over all five
    arms {persist, snaive, vote3, direct, encdec} fit on the ENTIRE 24-month validation window;
    the selected ensemble is evaluated once on test and exported as the prod atlas."""
    import json as _json
    Tn = b.Y.shape[0]
    Lpm = (b.L * 2 - 1).astype(np.float64)

    def arm_block(wall):
        P = {"persist": np.repeat(Lpm[:, wall - 1][:, None], H, 1)}
        sn = np.zeros((Tn, H)); v3 = np.zeros((Tn, H))
        for h in range(H):
            t = wall + h
            j = t - 12
            while j >= wall: j -= 12
            sn[:, h] = Lpm[:, j]
            v3[:, h] = (Lpm[:, j] + Lpm[:, j - 12] + Lpm[:, j - 24]) / 3.0
        P["snaive"] = sn; P["vote3"] = v3
        return P

    Pv, Pt = arm_block(b.a), arm_block(b.b)
    Zv_dir, Zt_dir = direct_logits(b)
    keep = encdec(b, names)
    tag = max(keep, key=lambda k: keep[k][0])
    Zv_ed = keep[tag][3]
    Zt_ed = np.stack([keep[tag][1][t] for t in range(b.b, b.n)], 1)
    phases = keep[tag][2]

    yv = b.L[:, b.a:b.b].astype(np.float64)
    cand_v = {"persist": Pv["persist"], "snaive": Pv["snaive"], "vote3": Pv["vote3"],
              "direct": Zv_dir, "encdec": Zv_ed}
    cand_t = {"persist": Pt["persist"], "snaive": Pt["snaive"], "vote3": Pt["vote3"],
              "direct": Zt_dir, "encdec": Zt_ed}
    order = ["persist", "snaive", "vote3", "direct", "encdec"]
    accV = np.stack([((cand_v[k] > 0) == (yv == 1)).mean(1) for k in order], 1)
    choice = accV.argmax(1)
    Zfin = np.zeros((Tn, H))
    for i, k in enumerate(order):
        m = choice == i
        Zfin[m] = cand_t[k][m]
        print(f"    arm {k:8s} chosen by {int(m.sum())} topics", flush=True)
    b.score(lambda t: Zfin[:, t - b.b] > 0, "FINAL_selected5")

    # ---- atlas + prod prediction export (balanced high/low wave) ----
    SIGNS_ = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio",
              "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
    ph_deg = np.degrees(phases) % 360.0
    pred = np.where(Zfin > 0, 1, -1).astype(int)
    act = (b.L[:, b.b:] * 2 - 1).astype(int)
    hits = (pred == act).astype(int)
    df = pd.DataFrame(dict(topic=names, phase=np.round(ph_deg, 2),
                           sign=[SIGNS_[int(x // 30) % 12] for x in ph_deg],
                           arm=[order[c] for c in choice],
                           dir_acc_test=np.round(hits.mean(1), 4)))
    df.to_csv("analysis/adstopics/balanced_atlas.csv", index=False)
    payload = {tf.slug(t): {"sqPred": [int(v) for v in pred[i]],
                            "sqHit": [int(v) for v in hits[i]]}
               for i, t in enumerate(names)}
    payload["_meta"] = {"tau": b.tau, "task": "balanced-highlow",
                        "champion": "per-topic selection over 5 arms (full-val fit, seed 7)"}
    _json.dump(payload, open("analysis/adstopics/astro_ts_pred.json", "w"))
    print(f"[balanced-final] mean test acc {hits.mean():.4f} · atlas + pred json written", flush=True)


def ceiling(b):
    prof = {}
    for t in range(b.b, b.n):
        prof.setdefault(b.moy[t], []).append(b.L[:, t])
    num = den = 0
    for m, v in prof.items():
        V = np.stack(v)
        pos = V.sum(0); neg = V.shape[0] - pos
        num += np.maximum(pos, neg).sum(); den += V.shape[0] * V.shape[1]
    print(f"  O3 per-(topic,month) cap: {num/den:.4f}")
    agree = []
    for t in range(b.b, b.b + 12):
        agree.append((b.L[:, t] == b.L[:, t + 12]).mean())
    print(f"  O2 cross-year agreement:  {np.mean(agree):.4f}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    names, Y = load_all()
    b = BenchB(Y)
    if mode in ("ceiling", "all"):
        ceiling(b)
    wall, base, clim = refs(b)
    if mode in ("pooled", "all"):
        pooled(b, wall, base, clim)
    if mode in ("simple", "all"):
        simple(b)
    if mode in ("encdec", "all"):
        encdec(b, names)
    if mode in ("direct", "all"):
        direct(b)
    if mode in ("champion", "all"):
        champion(b, names)
    if mode == "final":
        final(b, names)
    pd.DataFrame(b.rows).to_csv("analysis/adstopics/balanced_results.csv", index=False)

if __name__ == "__main__":
    main()
