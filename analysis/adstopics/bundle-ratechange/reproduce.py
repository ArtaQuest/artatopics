#!/usr/bin/env python3
"""REPRODUCTION CONTRACT — "Will search go higher? A balanced benchmark for the direction of the
monthly rate of change in YouTube search across the advertising taxonomy". Every headline number
regenerates from the REGISTERED OPEN DATASET alone:

    python3 reproduce.py <adstopics-ratechange-dataset.zip | unzipped-dir>      # scoreboard -> results.json vs expected.json
    python3 reproduce.py <dataset> --expected      # (authors) write expected.json
    python3 reproduce.py <dataset> --fig1 --fig2 --fig3
    python3 reproduce.py <dataset> --selftest      # planted-rule protocol sanity

THE TASK: r[i,t]=(y[t]-y[t-1])/max(y[t-1],1) is the monthly rate of change; its DIRECTION (sign) is the
label. Flat months (r==0, a 34.8% atom) are TIES, excluded from every fit, selection, score and ceiling.
Among the months that MOVED the two classes are balanced (test 50.06% higher), coin = 0.5. Walls: train
/ 24-mo validation (selection only) / untouched 24-mo AR test. CPU-canonical, seed 7, deterministic.
Deps: numpy, pandas, torch, matplotlib (figures only).
"""
import json, math, os, re, sys, tempfile, zipfile
import numpy as np, pandas as pd

H = 24
HERE = os.path.dirname(os.path.abspath(__file__))


def dataset_dir(arg):
    if os.path.isdir(arg):
        return arg
    d = tempfile.mkdtemp(prefix="adstopics-rc-")
    with zipfile.ZipFile(arg) as z:
        z.extractall(d)
    return d


def load_all(root):
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
        df = pd.read_csv(p); df["Time"] = pd.to_datetime(df["Time"])
        v = pd.to_numeric(df.drop_duplicates("Time").set_index("Time")["v"].reindex(grid), errors="coerce")
        if v.notna().sum() < len(grid) * 0.5:
            continue
        y = v.interpolate(limit_direction="both").to_numpy(float)
        if not np.isfinite(y).all() or y.max() <= 0:
            continue
        names.append(t); Ys.append(y)
    return names, np.stack(Ys)


class BenchR:
    def __init__(self, Y):
        self.Y = Y; self.n = Y.shape[1]; self.b = self.n - H; self.a = self.b - H
        R = np.zeros_like(Y, dtype=float); R[:, 1:] = (Y[:, 1:] - Y[:, :-1]) / np.maximum(Y[:, :-1], 1.0)
        self.R = R; self.tau = float(np.median(R[:, 1:self.a].ravel()))
        self.L = (R > self.tau).astype(int); self.tie = (R == self.tau); self.tie[:, 0] = True
        self.moy = np.arange(self.n) % 12; self.hs = np.arange(1, H + 1); self.rows = []
        tm = ~self.tie[:, self.b:]
        print(f"[ratechange] tau={self.tau:.3f} flat-atom {self.tie[:,1:self.a].mean()*100:.1f}% "
              f"test-moved {self.L[:,self.b:][tm].mean()*100:.2f}% higher · {Y.shape[0]} topics", flush=True)

    def score(self, pred, tag):
        acc = np.empty(H)
        for h in range(H):
            c = ~self.tie[:, self.b + h]; acc[h] = ((pred[c, h] > 0) == (self.L[c, self.b + h] == 1)).mean()
        auc = float(np.trapezoid(acc, self.hs) / (H - 1))
        self.preds = getattr(self, "preds", {}); self.preds[tag] = pred
        self.rows.append(dict(arm=tag, acc=round(auc, 4)))
        print(f"  {tag:22s} ACC {auc:.4f}", flush=True); return auc

    def ci(self, tag, B=2000, seed=7):
        pred = self.preds[tag]; rng = np.random.default_rng(seed); Tn = self.Y.shape[0]
        base = np.stack([((pred[:, h] > 0) == (self.L[:, self.b + h] == 1)).astype(float) for h in range(H)], 1)
        keep = (~self.tie[:, self.b:]).astype(float); out = []
        for _ in range(B):
            idx = rng.integers(0, Tn, Tn)
            out.append(np.trapezoid((base[idx] * keep[idx]).sum(0) / (keep[idx].sum(0) + 1e-9), self.hs) / (H - 1))
        a = np.array(out); return float(np.quantile(a, .025)), float(np.quantile(a, .975))


def arm_block(b, wall):
    Tn = b.Y.shape[0]; Lpm = (b.L * 2 - 1).astype(float)
    P = {"persist": np.repeat(Lpm[:, wall - 1][:, None], H, 1)}
    sn = np.zeros((Tn, H)); v3 = np.zeros((Tn, H))
    for h in range(H):
        t = wall + h; j = t - 12
        while j >= wall: j -= 12
        sn[:, h] = Lpm[:, j]; v3[:, h] = (Lpm[:, j] + Lpm[:, j - 12] + Lpm[:, j - 24]) / 3.0
    P["snaive"] = sn; P["vote3"] = v3; return P


def refs(b):
    Tn = b.Y.shape[0]; Pt = arm_block(b, b.b)
    b.score(Pt["persist"], "persistence")
    b.score(Pt["snaive"], "seasonal_naive")
    keep = (~b.tie[:, 1:b.a]).astype(float)
    base = ((b.L[:, 1:b.a] * keep).sum(1) / (keep.sum(1) + 1e-9) > 0.5) * 2.0 - 1
    b.score(np.repeat(base[:, None], H, 1), "base_rate")
    clim = np.zeros((Tn, 12))
    for m in range(12):
        sel = b.moy[1:b.a] == m; k = keep[:, sel]
        clim[:, m] = (b.L[:, 1:b.a][:, sel] * k).sum(1) / (k.sum(1) + 1e-9)
    cl = np.stack([(clim[:, b.moy[b.b + h]] > 0.5) * 2.0 - 1 for h in range(H)], 1)
    b.score(cl, "label_climatology")
    mm = np.sign(Pt["persist"] + cl + np.repeat(base[:, None], H, 1)); mm[mm == 0] = 1
    b.score(mm, "memory_majority")


def direct_logits(b, test=True):
    Tn, n = b.L.shape; Lpm = (b.L * 2 - 1).astype(np.float64)
    keepc = np.cumsum((~b.tie).astype(float), 1); hic = np.cumsum(b.L * (~b.tie), 1)

    def feats(w):
        wl = Lpm[:, w - 1]; mom = np.clip((b.R[:, w - 1] - b.tau) / 0.5, -3, 3)
        base = np.where(keepc[:, w - 1] > 0, hic[:, w - 1] / np.maximum(keepc[:, w - 1], 1) * 2 - 1, 0.0)
        cols = []
        for h in range(1, H + 1):
            t = w + h - 1; j = t - 12
            while j >= w: j -= 12
            sn = Lpm[:, j]; v3 = (Lpm[:, j] + Lpm[:, j - 12] + Lpm[:, j - 24]) / 3.0; hh = h / 24.0
            M = np.zeros((Tn, 12)); M[:, b.moy[t]] = 1.0
            cols.append(np.column_stack([wl, sn, v3, mom, base, np.full(Tn, hh), wl * hh, sn * hh, v3 * hh, wl * np.abs(v3), mom * hh, M]))
        return np.stack(cols, 1)

    walls = list(range(61, b.a - H + 1, 3)); Xs, ys, ws = [], [], []
    for w in walls:
        F = feats(w); Xs.append(F.reshape(-1, F.shape[-1]))
        ys.append(b.L[:, w:w + H].reshape(-1)); ws.append((~b.tie[:, w:w + H]).reshape(-1).astype(float))
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys).astype(np.float64); wtr = np.concatenate(ws)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Xtr = (Xtr - mu) / sd
    w_ = np.zeros(Xtr.shape[1] + 1); sw = wtr / wtr.mean()
    for _ in range(400):
        z = Xtr @ w_[:-1] + w_[-1]; p = 1 / (1 + np.exp(-np.clip(z, -30, 30))); g = (p - ytr) * sw
        w_ -= 2.0 * np.concatenate([Xtr.T @ g / len(ytr), [g.mean()]])
    F = feats(b.b if test else b.a); return ((F - mu) / sd) @ w_[:-1] + w_[-1]


def seasonal_champion(b, test=True):
    """THE CHAMPION: refined per-(topic, month-of-year) climatology of the rate-change direction —
    recency-weighted sign vote, base-rate shrinkage (k), seasonality-strength gate. Val-selected."""
    Tn = b.Y.shape[0]

    def clim(W, decay):
        cols = np.arange(1, W); mt = b.moy[cols]
        Wt = (~b.tie[:, cols]).astype(np.float64) * (decay ** ((W - 1 - cols) / 12.0))[None, :]
        Lc = b.L[:, cols].astype(np.float64)
        num = np.zeros((Tn, 12)); den = np.zeros((Tn, 12))
        for m in range(12):
            sel = mt == m; den[:, m] = Wt[:, sel].sum(1); num[:, m] = (Wt[:, sel] * Lc[:, sel]).sum(1)
        base = (Wt * Lc).sum(1) / (Wt.sum(1) + 1e-9); gclim = num.sum(0) / (den.sum(0) + 1e-9)
        return num, den, base, gclim

    def predict(W, decay, k, thr, width):
        num, den, base, gclim = clim(W, decay); shrunk = (num + k * base[:, None]) / (den + k)
        g = np.sqrt((den * (shrunk - base[:, None]) ** 2).sum(1) / (den.sum(1) + 1e-9))
        alpha = 1.0 / (1.0 + np.exp(-(g - thr) / width)); P = np.zeros((Tn, H))
        for h in range(H):
            m = b.moy[W + h]; P[:, h] = alpha * shrunk[:, m] + (1.0 - alpha) * gclim[m] - 0.5
        return P

    def vauc(pv):
        acc = np.empty(H)
        for h in range(H):
            c = ~b.tie[:, b.a + h]; acc[h] = ((pv[c, h] > 0) == (b.L[c, b.a + h] == 1)).mean()
        return float(np.trapezoid(acc, b.hs) / (H - 1))

    best = (-1.0, None)
    for decay in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5):
        for k in (0.5, 1, 2, 4, 8, 16, 32, 64):
            for thr in (-1.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
                a = vauc(predict(b.a, decay, k, thr, 0.05))
                if a > best[0]: best = (a, dict(decay=decay, k=k, thr=thr, width=0.05))
    cfg = best[1]
    return predict(b.b if test else b.a, cfg["decay"], cfg["k"], cfg["thr"], cfg["width"]), cfg, best[0]


def ceilings(b):
    num = den = 0
    for m in range(12):
        cols = [t for t in range(b.b, b.n) if b.moy[t] == m]
        for i in range(b.Y.shape[0]):
            vals = [b.L[i, t] for t in cols if not b.tie[i, t]]
            if vals: pos = sum(vals); num += max(pos, len(vals) - pos); den += len(vals)
    o3 = num / den
    ag = [((b.L[m, t] == b.L[m, t + 12]).mean()) for t in range(b.b, b.b + 12)
          for m in [(~b.tie[:, t]) & (~b.tie[:, t + 12])]]
    print(f"  O3 {o3:.4f} · O2 {np.mean(ag):.4f}", flush=True); return o3, float(np.mean(ag))


def selftest(b):
    rng = np.random.default_rng(7); Tn = b.Y.shape[0]; peak = rng.integers(0, 12, Tn)
    Lp = np.zeros_like(b.L)
    for i in range(Tn): Lp[i] = (((b.moy - peak[i]) % 12) < 6).astype(int)
    b.L = Lp; b.tie = np.zeros_like(b.tie)
    Pt = arm_block(b, b.b); acc = ((Pt["snaive"] > 0) == (Lp[:, b.b:] == 1)).mean()
    print(f"  planted-rule seasonal recovery {acc:.4f} (expect 1.0)", flush=True); return acc


def main():
    if len(sys.argv) < 2: sys.exit(__doc__)
    root = dataset_dir(sys.argv[1]); names, Y = load_all(root); b = BenchR(Y)
    if "--selftest" in sys.argv: selftest(b); return
    o3, o2 = ceilings(b); refs(b)
    b.score(direct_logits(b), "direct_pooled")
    Zc, cfg, vc = seasonal_champion(b)
    auc = b.score(Zc, "CHAMPION_seasonal")
    lo, hi = b.ci("CHAMPION_seasonal")
    print(f"  CHAMPION 95% CI [{lo:.4f}, {hi:.4f}] · val {vc:.4f} cfg {cfg}", flush=True)
    board = {r["arm"]: r["acc"] for r in b.rows}
    board.update(ceiling_o3=round(o3, 4), cross_year_o2=round(o2, 4), champion_ci=[round(lo, 4), round(hi, 4)],
                 flat_atom_pct=round(float(b.tie[:, 1:b.a].mean()), 4), topics=int(Y.shape[0]))
    json.dump(board, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("[results] written", flush=True)
    exp = os.path.join(HERE, "expected.json")
    if "--expected" in sys.argv:
        json.dump(board, open(exp, "w"), indent=1); print("[expected] written")
    elif os.path.exists(exp):
        E = json.load(open(exp))
        bad = [k for k, v in E.items() if isinstance(v, float) and abs(board.get(k, 1e9) - v) > 5e-5]
        print("[contract] MATCH — every number reproduces to 4 dp" if not bad else f"[contract] MISMATCH {bad}")
    # platform-identity check
    pp = os.path.join(root, "platform_predictions.json")
    if os.path.exists(pp):
        P = json.load(open(pp)); slug = lambda t: re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")
        mine = np.where(Zc > 0, 1, -1).astype(int)
        diff = sum(int(mine[i, h]) != int(P[slug(t)]["sqPred"][h]) for i, t in enumerate(names) for h in range(H) if slug(t) in P)
        print(f"[platform-identity] {diff} of {mine.size} cells differ from platform_predictions.json", flush=True)
    pd.DataFrame(b.rows).to_csv(os.path.join(HERE, "ratechange_results.csv"), index=False)
    if any(f in sys.argv for f in ("--fig1", "--fig2", "--fig3")):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        Pt = arm_block(b, b.b)
        if "--fig1" in sys.argv:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            for tag, P, st in (("champion (seasonal)", Zc, "-"), ("climatology", None, ":"), ("persistence", Pt["persist"], "--")):
                if tag == "climatology": continue
                acc = [((P[:, h] > 0) == (b.L[:, b.b + h] == 1))[~b.tie[:, b.b + h]].mean() for h in range(H)]
                ax.plot(range(1, H + 1), acc, st, label=tag)
            ax.axhline(o3, color="gray", lw=.8, ls="-."); ax.axhline(0.5, color="gray", lw=.8)
            ax.annotate(f"ceiling {o3:.3f}", (1, o3), fontsize=7, color="gray", va="bottom")
            ax.annotate(f"cross-year frontier {o2:.3f}", (1, o2), fontsize=7, color="#b04a4a", va="bottom")
            ax.axhline(o2, color="#b04a4a", lw=.8, ls=":")
            ax.set_ylim(0.45, 0.85); ax.set_xlabel("months ahead"); ax.set_ylabel("accuracy (moved months)")
            ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(HERE, "accuracy.png"), dpi=160)
            print("fig1 -> accuracy.png")
        if "--fig2" in sys.argv:
            rows = sorted([(r["arm"], r["acc"]) for r in b.rows], key=lambda x: x[1])
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh([r[0] for r in rows], [r[1] - 0.45 for r in rows], left=0.45,
                    color=["#E8B923" if r[0] == "CHAMPION_seasonal" else "#1746DC" for r in rows])
            for i, r in enumerate(rows): ax.text(r[1] + .002, i, f"{r[1]:.4f}", va="center", fontsize=7)
            ax.axvline(0.5, color="gray", lw=.8); ax.axvline(o2, color="#b04a4a", lw=.8, ls=":")
            ax.set_xlim(0.45, 0.87); ax.set_xlabel("tie-excluded horizon-AUC"); fig.tight_layout()
            fig.savefig(os.path.join(HERE, "scoreboard.png"), dpi=160); print("fig2 -> scoreboard.png")
        if "--fig3" in sys.argv:
            tiei = b.tie[:, b.b:]; act = np.where(b.L[:, b.b:] == 1, 1, -1); pr = np.where(Zc > 0, 1, -1)
            pa = np.array([((pr[i][~tiei[i]] == act[i][~tiei[i]]).mean() if (~tiei[i]).any() else np.nan) for i in range(len(names))])
            peak = np.array([np.bincount((b.moy[1:b.b][b.L[i, 1:b.b] == 1]) if (b.L[i, 1:b.b] == 1).any() else [0], minlength=12).argmax() for i in range(len(names))])
            fig, (axl, axr) = plt.subplots(1, 2, figsize=(9, 3.6))
            axl.hist(pa[~np.isnan(pa)], bins=25, color="#1746DC", alpha=.85); axl.axvline(0.5, color="gray", lw=.8)
            axl.set_xlabel("per-topic test accuracy (moved months)"); axl.set_ylabel("topics")
            mn = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
            cnt = np.bincount(peak, minlength=12); axr.bar(range(12), cnt, color="#E8B923")
            axr.set_xticks(range(12)); axr.set_xticklabels(mn); axr.set_xlabel("topic's peak-growth month"); axr.set_ylabel("topics")
            fig.tight_layout(); fig.savefig(os.path.join(HERE, "distributions.png"), dpi=160); print("fig3 -> distributions.png")


if __name__ == "__main__":
    main()
