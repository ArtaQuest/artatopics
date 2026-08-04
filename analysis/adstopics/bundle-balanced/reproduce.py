#!/usr/bin/env python3
"""REPRODUCTION CONTRACT — "High or low: the balanced seasonality of the advertising taxonomy
in YouTube search". Every headline number regenerates from the REGISTERED OPEN DATASET alone:

    python3 reproduce.py <adstopics-dataset-balanced.zip | unzipped-dir>          # the scoreboard
    python3 reproduce.py <dataset> --expected      # (authors) write expected.json
    python3 reproduce.py <dataset> --fig1          # accuracy-by-horizon figure
    python3 reproduce.py <dataset> --fig2          # champion arm shares + per-arm accuracy
    python3 reproduce.py <dataset> --selftest      # planted-label sanity (no dataset model leak)

Deterministic and CPU-CANONICAL: torch arms force CPU + seed 7; the scoreboard is compared to
expected.json at 4 decimals. No network, no repo — only the dataset and this file.
Dependencies: numpy, pandas, torch, matplotlib (figures only).
"""
import json, math, os, re, sys, tempfile, zipfile
import numpy as np
import pandas as pd

H = 24
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- dataset
def dataset_dir(arg):
    if os.path.isdir(arg):
        return arg
    d = tempfile.mkdtemp(prefix="adstopics-balanced-")
    with zipfile.ZipFile(arg) as z:
        z.extractall(d)
    return d


def load_all(root):
    """The clean monthly window 2008-01..2025-06 (210 months) for every audit-clean topic —
    the same gates the dataset was built with (>=50% native coverage, positive, finite)."""
    grid = pd.date_range("2008-01-01", "2025-06-01", freq="MS")
    vocab = json.load(open(os.path.join(root, "vocabulary.json")))
    bl = set(json.load(open(os.path.join(root, "blacklist.json"))).get("excluded_topics", []))
    slug = lambda t: re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")
    names, Ys = [], []
    for t in sorted(vocab):
        if t in bl:
            continue
        p = os.path.join(root, "series", slug(t) + ".csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df["Time"] = pd.to_datetime(df["Time"])
        v = pd.to_numeric(df.drop_duplicates("Time").set_index("Time")["v"].reindex(grid),
                          errors="coerce")
        if v.notna().sum() < len(grid) * 0.5:
            continue
        y = v.interpolate(limit_direction="both").to_numpy(float)
        if not np.isfinite(y).all() or y.max() <= 0:
            continue
        names.append(t)
        Ys.append(y)
    return names, np.stack(Ys)


class Bench:
    """The balanced protocol: tau = median rate over ALL topics x TRAIN months (never the
    held-out years); label = rate > tau; walls train / 24-month val / untouched 24-month test."""

    def __init__(self, Y):
        self.Y = Y
        self.n = Y.shape[1]
        self.b = self.n - H
        self.a = self.b - H
        self.tau = float(np.median(Y[:, :self.a].ravel()))
        self.L = (Y > self.tau).astype(int)
        self.moy = np.arange(self.n) % 12
        self.hs = np.arange(1, H + 1)
        self.rows = []
        print(f"[bench] tau={self.tau:.1f} · train {self.L[:, :self.a].mean()*100:.2f}% high · "
              f"{Y.shape[0]} topics x {self.n} months", flush=True)

    def score(self, pred24, tag):
        """pred24: (T, 24) — +1/-1 or logits; scored by sign against the last 24 months."""
        hits = ((pred24 > 0) == (self.L[:, self.b:] == 1)).astype(float)   # (T, 24)
        acc = hits.mean(0)
        auc = float(np.trapezoid(acc, self.hs) / (H - 1))
        self.hitmats = getattr(self, "hitmats", {})
        self.hitmats[tag] = hits
        self.rows.append(dict(arm=tag, acc=round(auc, 4), h1=round(acc[0], 4),
                              h24=round(acc[-1], 4)))
        print(f"  {tag:22s} ACC {auc:.4f} · h1 {acc[0]:.3f} · h24 {acc[-1]:.3f}", flush=True)
        return auc

    def bootstrap_ci(self, tag, B=2000, seed=7):
        """Topic-level bootstrap 95% CI of the horizon-AUC (resample the topics)."""
        hits = self.hitmats[tag]
        rng = np.random.default_rng(seed)
        Tn = hits.shape[0]
        aucs = np.empty(B)
        for i in range(B):
            idx = rng.integers(0, Tn, Tn)
            aucs[i] = np.trapezoid(hits[idx].mean(0), self.hs) / (H - 1)
        lo, hi = np.quantile(aucs, [0.025, 0.975])
        return float(lo), float(hi), float(aucs.std())


# ---------------------------------------------------------------- memory arms
def arm_block(b, wall):
    Tn = b.Y.shape[0]
    Lpm = (b.L * 2 - 1).astype(float)
    P = {"persist": np.repeat(Lpm[:, wall - 1][:, None], H, 1)}
    sn = np.zeros((Tn, H)); v3 = np.zeros((Tn, H))
    for h in range(H):
        t = wall + h
        j = t - 12
        while j >= wall:
            j -= 12
        sn[:, h] = Lpm[:, j]
        v3[:, h] = (Lpm[:, j] + Lpm[:, j - 12] + Lpm[:, j - 24]) / 3.0
    P["snaive"] = sn
    P["vote3"] = v3
    return P


def refs(b):
    Tn = b.Y.shape[0]
    Pt = arm_block(b, b.b)
    b.score(Pt["persist"], "persistence_AR")
    b.score(Pt["snaive"], "seasonal_naive_AR")
    base = (b.L[:, :b.a].mean(1) > 0.5) * 2.0 - 1
    b.score(np.repeat(base[:, None], H, 1), "base_rate")
    clim = np.zeros((Tn, 12))
    for m in range(12):
        sel = b.moy[:b.a] == m
        clim[:, m] = b.L[:, :b.a][:, sel].mean(1)
    cl = np.stack([(clim[:, b.moy[b.b + h]] > 0.5) * 2.0 - 1 for h in range(H)], 1)
    b.score(cl, "label_climatology")
    # majority of three memories: the wall state, the monthly climatology, the base rate
    mm = np.sign(Pt["persist"] + cl + np.repeat(base[:, None], H, 1))
    b.score(mm, "memory_majority")
    b.score(pooled_ar(b), "pooled_logistic_AR")


def pooled_ar(b):
    """Pooled logistic with an AUTOREGRESSIVE rollout (predicted labels feed the lag features
    forward) — the classic iterated strategy, kept as the honest contrast to `direct`."""
    Tn, n = b.L.shape
    base = b.L[:, :b.a].mean(1)
    clim = np.zeros((Tn, 12))
    for m in range(12):
        sel = b.moy[:b.a] == m
        clim[:, m] = b.L[:, :b.a][:, sel].mean(1)

    def feats(t, Lsrc):
        cols = [Lsrc[:, t - 1] * 2 - 1, Lsrc[:, t - 12] * 2 - 1, base * 2 - 1,
                clim[:, b.moy[t]] * 2 - 1,
                np.clip((b.Y[:, b.a - 1] - b.tau) / (b.tau + 1), -3, 3)]
        M = np.zeros((Tn, 12)); M[:, b.moy[t]] = 1.0
        return np.column_stack(cols + [M])

    Xs, ys = [], []
    for t in range(13, b.a):
        Xs.append(feats(t, b.L))
        ys.append(b.L[:, t])
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys).astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    w = np.zeros(Xtr.shape[1] + 1)
    for _ in range(300):
        z = Xtr @ w[:-1] + w[-1]
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        w -= 2.0 * np.concatenate([Xtr.T @ (p - ytr) / len(ytr), [(p - ytr).mean()]])
    Lar = b.L.copy()
    Z = np.zeros((Tn, H))
    for h, t in enumerate(range(b.b, b.n)):
        z = ((feats(t, Lar) - mu) / sd) @ w[:-1] + w[-1]
        Z[:, h] = z
        Lar[:, t] = (z > 0).astype(int)
    return Z


def direct_logits(b):
    """DIRECT multi-horizon pooled logistic — pre-wall features only, no rollout."""
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
            while j >= w:
                j -= 12
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
    return out


def encdec_logits(b):
    """The phase ENCODER-DECODER on the high/low wave (CPU, seed 7): encoder attends over the
    W=36 pre-wall wave and returns a circular phase; decoder attends over the rotated tokens,
    queried by the rotated target angle; strictly AR-safe memory features at the output."""
    import torch as T
    T.use_deterministic_algorithms(True)
    dev_ = "cpu"
    Tn = b.Y.shape[0]
    W = 36
    a1 = b.a - 1
    lw = (b.L * 2 - 1).astype(np.float32)
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

    out = {}
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
            loss.backward()
            opt.step()
            if ep % 5 == 4:
                with T.no_grad():
                    acc = ((decoder(va, encoder(), b.a) > 0).float() == yva).float().mean().item()
                if acc > best + 1e-5:
                    best, stall, state = acc, 0, [x.detach().clone() for x in params]
                else:
                    stall += 1
                if stall >= 15:
                    break
        with T.no_grad():
            for x, st in zip(params, state):
                x.copy_(st)
            pfin = encoder()
            Zt = T.stack([decoder([t], pfin, b.b)[:, 0] for t in range(b.b, b.n)], 1).cpu().numpy()
            Zv = decoder(va, pfin, b.a).cpu().numpy()
        print(f"    {tag}: val acc {best:.4f}", flush=True)
        out[tag] = (best, Zv, Zt, pfin.cpu().numpy())
    return out


def champion(b, write_preds=None):
    """THE CHAMPION: per-topic arm selection over five arms fit on the ENTIRE validation window;
    the selected ensemble touches test once."""
    Tn = b.Y.shape[0]
    Pv, Pt = arm_block(b, b.a), arm_block(b, b.b)
    Zv_dir, Zt_dir = direct_logits(b)
    ed = encdec_logits(b)
    tag = max(ed, key=lambda k: ed[k][0])
    Zv_ed, Zt_ed = ed[tag][1], ed[tag][2]
    b.score(ed["ENCDECB_pure"][2], "ENCDECB_pure")
    b.score(ed["ENCDECB_mem"][2], "ENCDECB_mem")
    b.score(Zt_dir, "direct_pooled")
    yv = b.L[:, b.a:b.b].astype(float)
    cand_v = {"persist": Pv["persist"], "snaive": Pv["snaive"], "vote3": Pv["vote3"],
              "direct": Zv_dir, "encdec": Zv_ed}
    cand_t = {"persist": Pt["persist"], "snaive": Pt["snaive"], "vote3": Pt["vote3"],
              "direct": Zt_dir, "encdec": Zt_ed}
    order = ["persist", "snaive", "vote3", "direct", "encdec"]
    accV = np.stack([((cand_v[k] > 0) == (yv == 1)).mean(1) for k in order], 1)
    choice = accV.argmax(1)
    Z = np.zeros((Tn, H))
    shares = {}
    for i, k in enumerate(order):
        m = choice == i
        Z[m] = cand_t[k][m]
        shares[k] = int(m.sum())
        print(f"    arm {k:8s} chosen by {int(m.sum())} topics", flush=True)
    auc = b.score(Z, "CHAMPION_selected5")
    # per-arm generalisation: on the topics each arm was chosen for, does it beat carrying
    # persistence there? (Only the seasonal copy survives the wall — the paper's Table 2.)
    ch_hits = ((Z > 0) == (b.L[:, b.b:] == 1)).astype(float)
    p_hits = ((Pt["persist"] > 0) == (b.L[:, b.b:] == 1)).astype(float)
    print("  [arm-generalisation] arm chosen-for champion persistence delta", flush=True)
    for i, k in enumerate(order):
        mm = choice == i
        c, p = ch_hits[mm].mean(), p_hits[mm].mean()
        print(f"    {k:8s} {int(mm.sum()):5d} {c:.4f} {p:.4f} {100*(c-p):+.1f}pp", flush=True)
    sw = choice != 0
    print(f"    swapped  {int(sw.sum()):5d} {ch_hits[sw].mean():.4f} {p_hits[sw].mean():.4f} "
          f"{100*(ch_hits[sw].mean()-p_hits[sw].mean()):+.1f}pp", flush=True)
    if write_preds:
        np.save(write_preds, np.where(Z > 0, 1, -1).astype(int))
    return auc, shares, choice, Z


def ceilings(b):
    prof = {}
    for t in range(b.b, b.n):
        prof.setdefault(b.moy[t], []).append(b.L[:, t])
    num = den = 0
    for m, v in prof.items():
        V = np.stack(v)
        pos = V.sum(0)
        num += np.maximum(pos, V.shape[0] - pos).sum()
        den += V.shape[0] * V.shape[1]
    o3 = num / den
    agree = np.mean([(b.L[:, t] == b.L[:, t + 12]).mean() for t in range(b.b, b.b + 12)])
    print(f"  O3 per-(topic,month) cap {o3:.4f} · O2 cross-year agreement {agree:.4f}", flush=True)
    return o3, agree


def selftest(b):
    """Planted sanity: relabel with a KNOWN per-topic seasonal rule and check the memory arms
    recover it — protocol wiring, not model quality."""
    rng = np.random.default_rng(7)
    Tn = b.Y.shape[0]
    peak = rng.integers(0, 12, Tn)
    Lp = np.zeros_like(b.L)
    for i in range(Tn):
        Lp[i] = (((b.moy - peak[i]) % 12) < 6).astype(int)
    b2 = Bench(b.Y)
    b2.L = Lp
    Pt = arm_block(b2, b2.b)
    acc = ((Pt["snaive"] > 0) == (Lp[:, b2.b:] == 1)).mean()
    print(f"  planted-rule snaive recovery: {acc:.4f} (expect 1.0)", flush=True)
    return acc


def tau_sensitivity(b, names):
    """The ladder at the ADJACENT threshold (the other side of the integer atom): references,
    direct and the 4-arm selector (the encoder--decoder is omitted here for cost, declared).
    One robustness line: the ordering and gaps survive the only other defensible tau."""
    tau2 = 18.0    # label = rate > 18  (train split 50.85% high — the atom's other side)
    b2 = Bench(b.Y)
    b2.tau = tau2
    b2.L = (b.Y > tau2).astype(int)
    print(f"[tau-sens] tau>{tau2:.0f} · train {b2.L[:, :b2.a].mean()*100:.2f}% high", flush=True)
    refs(b2)
    Zv_dir, Zt_dir = direct_logits(b2)
    b2.score(Zt_dir, "direct_pooled")
    Pv, Pt = arm_block(b2, b2.a), arm_block(b2, b2.b)
    yv = b2.L[:, b2.a:b2.b].astype(float)
    order = ["persist", "snaive", "vote3", "direct"]
    cand_v = {**{k: Pv[k] for k in ["persist", "snaive", "vote3"]}, "direct": Zv_dir}
    cand_t = {**{k: Pt[k] for k in ["persist", "snaive", "vote3"]}, "direct": Zt_dir}
    accV = np.stack([((cand_v[k] > 0) == (yv == 1)).mean(1) for k in order], 1)
    ch = accV.argmax(1)
    Z = np.zeros((b.Y.shape[0], H))
    for i, k in enumerate(order):
        Z[ch == i] = cand_t[k][ch == i]
    b2.score(Z, "selected4_tau18")
    return {r["arm"]: r["acc"] for r in b2.rows}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = dataset_dir(sys.argv[1])
    names, Y = load_all(root)
    b = Bench(Y)
    if "--selftest" in sys.argv:
        selftest(b)
        return
    if "--tau-sens" in sys.argv:
        out = tau_sensitivity(b, names)
        json.dump(out, open(os.path.join(HERE, "tau_sensitivity.json"), "w"), indent=1)
        return
    test_high = float(b.L[:, b.b:].mean())
    print(f"  realised TEST split: {test_high*100:.2f}% high", flush=True)
    o3, o2 = ceilings(b)
    refs(b)
    auc, shares, choice, Z = champion(b, write_preds=os.path.join(HERE, "champion_preds.npy"))
    board = {r["arm"]: r["acc"] for r in b.rows}
    board["ceiling_O3"] = round(float(o3), 4)
    board["cross_year_O2"] = round(float(o2), 4)
    board["tau"] = b.tau
    board["topics"] = int(Y.shape[0])
    board["test_high_pct"] = round(test_high, 4)
    cis = {}
    for tag in ("CHAMPION_selected5", "persistence_AR", "seasonal_naive_AR", "direct_pooled"):
        lo, hi, se = b.bootstrap_ci(tag)
        cis[tag] = {"lo": round(lo, 4), "hi": round(hi, 4), "se": round(se, 5)}
        print(f"  CI95 {tag:22s} [{lo:.4f}, {hi:.4f}] (topic bootstrap, B=2000)", flush=True)
    board["ci95"] = cis
    json.dump(board, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("[results] written: results.json", flush=True)
    exp_path = os.path.join(HERE, "expected.json")
    if "--expected" in sys.argv:
        json.dump(board, open(exp_path, "w"), indent=1)
        print(f"[expected] written: {exp_path}")
    elif os.path.exists(exp_path):
        exp = json.load(open(exp_path))
        bad = [k for k, v in exp.items()
               if isinstance(v, float) and abs(board.get(k, 1e9) - v) > 5e-5]
        print("[contract] MATCH — every number reproduces to 4 dp" if not bad
              else f"[contract] MISMATCH: {bad}")
    # one scoreboard, one code path: the dataset's ladder CSV is THIS board
    pd.DataFrame(b.rows).to_csv(os.path.join(HERE, "balanced_results.csv"), index=False)
    # bit-identity vs the platform's shipped predictions (when the dataset carries them)
    pp_path = os.path.join(root, "platform_predictions.json")
    if os.path.exists(pp_path):
        pp = json.load(open(pp_path))
        slug = lambda t: re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")
        mine = np.where(Z > 0, 1, -1).astype(int)
        diff = sum(int(mine[i, h]) != int(pp[slug(t)]["sqPred"][h])
                   for i, t in enumerate(names) for h in range(H) if slug(t) in pp)
        print(f"[platform-identity] {diff} of {mine.size} cells differ from the dataset's "
              f"platform_predictions.json", flush=True)
    if any(f in sys.argv for f in ("--fig1", "--fig2", "--fig3")):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if "--fig1" in sys.argv:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            Pt = arm_block(b, b.b)
            arms1 = (("champion (selected-5)", Z, "-"),
                     ("persistence", Pt["persist"], "--"),
                     ("seasonal naive", Pt["snaive"], ":"))
            hs = range(1, H + 1)
            for tag, P, st in arms1:
                acc = [((P[:, h] > 0) == (b.L[:, b.b + h] == 1)).mean() for h in range(H)]
                ax.plot(hs, acc, st, label=tag)
            ch_hits = b.hitmats["CHAMPION_selected5"]
            rng = np.random.default_rng(7)
            band = np.stack([ch_hits[rng.integers(0, len(ch_hits), len(ch_hits))].mean(0)
                             for _ in range(400)])
            ax.fill_between(hs, np.quantile(band, 0.025, 0), np.quantile(band, 0.975, 0),
                            alpha=0.15, color="#1746DC",
                            label="champion 95% band (topic bootstrap)")
            ax.axhline(o3, color="gray", lw=0.8, ls="-.")
            ax.annotate(f"ceiling {o3:.4f}", (1.2, o3), fontsize=7, color="gray",
                        va="bottom")
            ax.set_ylim(0.88, 0.97)
            ax.set_xlabel("months ahead")
            ax.set_ylabel("accuracy (coin = 0.5, off-scale)")
            ax.legend(fontsize=8, loc="lower left")
            fig.tight_layout()
            fig.savefig(os.path.join(HERE, "accuracy.png"), dpi=160)
            print("fig1 -> accuracy.png")
        if "--fig2" in sys.argv:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ks = list(shares)
            ax.bar(ks, [shares[k] for k in ks], color="#1746DC")
            ax.set_yscale("log")
            ax.set_ylabel("topics (log scale)")
            for i, k in enumerate(ks):
                ax.text(i, shares[k], str(shares[k]), ha="center", va="bottom", fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(HERE, "arm_shares.png"), dpi=160)
            print("fig2 -> arm_shares.png")
        if "--fig3" in sys.argv:
            # the noise floor made visible: per-topic accuracy vs distance from the bar
            per_topic = b.hitmats["CHAMPION_selected5"].mean(1)
            dist = np.abs(b.Y[:, b.b:] - b.tau).mean(1)
            fig, (axl, axr) = plt.subplots(1, 2, figsize=(9, 3.6))
            axl.hist(per_topic, bins=25, color="#1746DC", alpha=0.85)
            axl.set_xlabel("per-topic test accuracy (24 months)")
            axl.set_ylabel("topics")
            edges = np.quantile(dist, np.linspace(0, 1, 11))
            mids, means = [], []
            for i in range(10):
                m = (dist >= edges[i]) & (dist <= edges[i + 1])
                if m.sum():
                    mids.append(dist[m].mean())
                    means.append(per_topic[m].mean())
            axr.plot(mids, means, "o-", color="#1746DC")
            axr.set_xlabel(r"mean $|$rate $- \tau|$ over the test window")
            axr.set_ylabel("mean per-topic accuracy")
            axr.axhline(0.5, color="gray", lw=0.8)
            fig.tight_layout()
            fig.savefig(os.path.join(HERE, "noise_floor.png"), dpi=160)
            print("fig3 -> noise_floor.png")


if __name__ == "__main__":
    main()
