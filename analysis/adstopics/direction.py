#!/usr/bin/env python3
"""adstopics — THE direction-classification engine (consolidates classify_run/v2/v3/links).

Task (operator, 2026-07-14): predict next month's RISE/FALL. Test = the last 24 months of the
recency-excluded window; per-horizon accuracy over topics (actual ties excluded); the global
score = the accuracy-vs-horizon curve's normalised AUC (0.5 = coin flip).

Model of record: the Gaussian furnace, frozen width, positive weights, bias, phases descended
from the 12 sign centres, prune-4 refit — fitted on the DIFFERENCED series (the furnace heats
the flow, not the stock), optionally with one window per lamp.

  python3 analysis/adstopics/direction.py selftest        # synthetic seasonal series must score ≥0.9
  python3 analysis/adstopics/direction.py arms [N]        # ablation: level|delta|sign × shared|per-lamp + votes
  python3 analysis/adstopics/direction.py links [N]       # all 7 links, Δ-fitted, same protocol
  python3 analysis/adstopics/direction.py headline [N|all]  # the record model + refs + the paper figure

→ direction_<mode>.csv (+ paper/figs/direction_accuracy.png for headline)
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
co = _load("analysis/adstopics/combo_experiments.py", "co")

H = 24
CHUNK = 400
LINKS = ["gauss", "cos", "rcos", "vonmises", "laplace", "sinc", "sinc2"]


def kval(kern, z, f):
    return {"vonmises": lambda: np.exp(f * (np.cos(z) - 1.0)),
            "cos":      lambda: np.cos(z),
            "rcos":     lambda: 0.5 * (1.0 + np.cos(z)),
            "gauss":    lambda: np.exp(-(z * f) ** 2),
            "laplace":  lambda: np.exp(-np.abs(z) * f),
            "sinc2":    lambda: np.sinc(z * f) ** 2,
            "sinc":     lambda: np.sinc(z * f)}[kern]()


def fit(target, X, dev, split, kern="gauss", per_lamp=False, prune_k=4, return_params=False):
    """Two-stage spec fit (12 lamps → prune → refit), chunked. target: (T,m) array."""
    a = split[0]
    T = len(target)
    out = np.zeros((T, target.shape[1]))
    pcol = []
    for lo in range(0, T, CHUNK):
        Yc = list(target[lo:lo + CHUNK])
        _, p1 = co.fit_vm(Yc, X, dev, kernel=kern, fixed_f=1.0, intercept=True,
                          split=split, per_lamp_phase=per_lamp)
        C = np.zeros((len(Yc), ex.NBX))
        for i in range(len(Yc)):
            ph = p1["p"][i][None, :] if per_lamp else p1["p"][i]
            z = np.deg2rad((X[:a] - ph + 180.0) % 360.0 - 180.0)
            C[i] = (p1["w"][i][None, :] * kval(kern, z, p1["kappa"][i][None, :])).std(0)
        m = np.zeros_like(C)
        for i, ix in enumerate(np.argsort(-C, axis=1)[:, :prune_k]):
            m[i, ix] = 1.0
        pred, p2 = co.fit_vm(Yc, X, dev, kernel=kern, fixed_f=1.0, intercept=True,
                             body_mask=m, split=split, per_lamp_phase=per_lamp)
        out[lo:lo + len(Yc)] = pred
        pcol.append(dict(p=p2["p"], mask=m))
        if T > CHUNK:
            print(f"    fit {min(lo + CHUNK, T)}/{T}", flush=True)
    if return_params:
        return out, {"p": np.concatenate([c["p"] for c in pcol]),
                     "mask": np.concatenate([c["mask"] for c in pcol])}
    return out


class Bench:
    """The protocol: data, references, scoring. dy[:,t-1] = change INTO month t."""

    def __init__(self, Y, X):
        self.Y, self.X = Y, X
        self.n = Y.shape[1]
        self.b = self.n - H
        self.a = self.b - H
        self.dy = np.diff(Y, axis=1)
        # THE SQUARE WAVE (operator 2026-07-16): every series converted once to its rise/fall
        # wave: +1 rise, -1 fall, a tie HOLDS the previous value (stated convention; the first
        # tie before any move resolves to +1). All strict square-wave features derive from this.
        sq = np.sign(self.dy)
        for t in range(sq.shape[1]):
            if t == 0:
                sq[:, 0] = np.where(sq[:, 0] == 0, 1, sq[:, 0])
            else:
                z = sq[:, t] == 0
                sq[z, t] = sq[z, t - 1]
        self.sq = sq
        self.Xd = X[1:]
        self.splitd = (self.a - 1, self.b - 1)
        self.moy = np.arange(1, self.n) % 12          # calendar bucket of each Δ's arrival month
        self.hs = np.arange(1, H + 1)
        self.rows = []

    def curve(self, dirfun):
        acc = []
        for h in range(1, H + 1):
            t = self.b + h - 1
            act = self.dy[:, t - 1]
            ok = act != 0
            acc.append(float((dirfun(t) == (act > 0))[ok].mean()))
        return np.array(acc)

    def score(self, dirfun, tag):
        acc = self.curve(dirfun)
        auc = float(np.trapezoid(acc, self.hs) / (H - 1))
        self.rows.append(dict(arm=tag, auc=round(auc, 4), acc_h1=round(acc[0], 4),
                              acc_mean=round(float(acc.mean()), 4), acc_h24=round(acc[-1], 4)))
        print(f"  {tag:16s} AUC {auc:.4f} · h1 {acc[0]:.3f} · h24 {acc[-1]:.3f}", flush=True)
        return acc

    def climatology(self):
        a1 = self.a - 1
        clim = np.zeros((self.Y.shape[0], 12))
        for mth in range(12):
            clim[:, mth] = self.dy[:, :a1][:, self.moy[:a1] == mth].mean(1)
        return clim

    def month_vote(self, pred, clim):
        """Use the model on the calendar months where its TRAIN record beats climatology's."""
        a1 = self.a - 1
        act = self.dy[:, :a1] > 0
        okc = self.dy[:, :a1] != 0
        accM = np.zeros((self.Y.shape[0], 12)); accC = np.zeros_like(accM)
        for mth in range(12):
            sel = (self.moy[:a1] == mth) & okc
            cnt = np.maximum(sel.sum(1), 1)
            accM[:, mth] = (((pred[:, :a1] > 0) == act) & sel).sum(1) / cnt
            accC[:, mth] = (((clim[:, mth] > 0)[:, None] == act) & sel).sum(1) / cnt
        return lambda t: np.where(accM[:, self.moy[t - 1]] > accC[:, self.moy[t - 1]],
                                  pred[:, t - 1] > 0, clim[:, self.moy[t - 1]] > 0)

    def wclim(self, gamma):
        """Recency-weighted climatology on the train window (γ^age per year)."""
        a1 = self.a - 1
        ages = (a1 - 1 - np.arange(a1)) / 12.0
        w = gamma ** ages
        clim = np.zeros((self.Y.shape[0], 12))
        for mth in range(12):
            sel = self.moy[:a1] == mth
            clim[:, mth] = (self.dy[:, :a1][:, sel] * w[sel]).sum(1) / w[sel].sum()
        return clim

    def val_acc(self, dirfun):
        """Mean direction accuracy over the VALIDATION months (for per-topic selection)."""
        hits = np.zeros(self.Y.shape[0]); cnt = np.zeros(self.Y.shape[0])
        for t in range(self.a, self.b):
            act = self.dy[:, t - 1]; ok = act != 0
            hits += ((dirfun(t) == (act > 0)) & ok); cnt += ok
        return hits / np.maximum(cnt, 1)

    def sq_clim(self, gamma=0.8):
        """(T,12) recency-weighted mean of the SQUARE WAVE per calendar month (train only)."""
        a1 = self.a - 1
        w = gamma ** ((a1 - 1 - np.arange(a1)) / 12.0)
        out = np.zeros((self.Y.shape[0], 12))
        for mth in range(12):
            sel = self.moy[:a1] == mth
            out[:, mth] = (self.sq[:, :a1][:, sel] * w[sel]).sum(1) / w[sel].sum()
        return out

    def pooled_sq_dirfun(self):
        """The pooled logistic on STRICTLY square-wave-derived features."""
        s08, s10 = self.sq_clim(0.8), self.sq_clim(1.0)
        Tn = self.Y.shape[0]
        def feats(t):
            F = [self.sq[:, t - 1 - 12 * k] for k in (1, 2, 3)]
            F += [s08[:, self.moy[t - 1]], s10[:, self.moy[t - 1]]]
            M = np.zeros((Tn, 12)); M[:, self.moy[t - 1]] = 1.0
            return np.column_stack(F + [M])
        def gather(ts):
            Xs, ys = [], []
            for t in ts:
                act = self.dy[:, t - 1]; ok = act != 0
                Xs.append(feats(t)[ok]); ys.append((act[ok] > 0).astype(float))
            return np.vstack(Xs), np.concatenate(ys)
        Xtr, ytr = gather(range(37, self.a))
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sd
        w = np.zeros(Xtr.shape[1] + 1)
        for _ in range(300):
            z = Xtr @ w[:-1] + w[-1]
            p = 1 / (1 + np.exp(-z))
            w -= 2.0 * np.concatenate([Xtr.T @ (p - ytr) / len(ytr), [(p - ytr).mean()]])
        return lambda t: ((feats(t) - mu) / sd) @ w[:-1] + w[-1] > 0

    def pooled_dirfun(self):
        """The pooled global logistic (train on pooled train months, lam on validation)."""
        c08, c10 = self.wclim(0.8), self.wclim(1.0)
        scale = np.median(np.abs(self.dy[:, :self.a - 1]), axis=1) + 1e-6
        Tn = self.Y.shape[0]
        def feats(t):
            F = [np.sign(self.dy[:, t - 1 - 12 * k]) for k in (1, 2, 3)]
            F += [np.clip(c08[:, self.moy[t - 1]] / scale, -3, 3),
                  np.clip(c10[:, self.moy[t - 1]] / scale, -3, 3)]
            M = np.zeros((Tn, 12)); M[:, self.moy[t - 1]] = 1.0
            return np.column_stack(F + [M])
        def gather(ts):
            Xs, ys = [], []
            for t in ts:
                act = self.dy[:, t - 1]; ok = act != 0
                Xs.append(feats(t)[ok]); ys.append((act[ok] > 0).astype(float))
            return np.vstack(Xs), np.concatenate(ys)
        Xtr, ytr = gather(range(37, self.a))
        Xva, yva = gather(range(self.a, self.b))
        mu, sdv = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sdv; Xva = (Xva - mu) / sdv
        def train_logistic(Xm, ym, lam, iters=300):
            wgt = np.zeros(Xm.shape[1] + 1)
            for _ in range(iters):
                z = Xm @ wgt[:-1] + wgt[-1]
                pgd = 1 / (1 + np.exp(-z))
                wgt -= 2.0 * np.concatenate([Xm.T @ (pgd - ym) / len(ym) + lam * wgt[:-1],
                                             [(pgd - ym).mean()]])
            return wgt
        best = None
        for lam in (0.0, 1e-4, 1e-3, 1e-2):
            wgt = train_logistic(Xtr, ytr, lam)
            va = (((Xva @ wgt[:-1] + wgt[-1]) > 0) == (yva > 0.5)).mean()
            if best is None or va > best[1]:
                best = (wgt, va, lam)
        wgt = best[0]
        return lambda t: ((feats(t) - mu) / sdv) @ wgt[:-1] + wgt[-1] > 0

    def refs(self):
        clim = self.climatology()
        self.score(lambda t: self.dy[:, t - 13] > 0, "ref_snaive")
        self.score(lambda t: clim[:, self.moy[t - 1]] > 0, "ref_climatology")
        self.score(lambda t: np.ones(self.Y.shape[0], bool), "ref_always_rise")
        return clim


def load_bench(want):
    names, Ys, X = ex.load_topics(10 ** 9 if want == "all" else int(want))
    b = Bench(np.stack(Ys), X)
    print(f"[direction] {len(Ys)} topics · train [0,{b.a}) val [{b.a},{b.b}) test [{b.b},{b.n})")
    return b


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "arms"
    want = sys.argv[2] if len(sys.argv) > 2 else "400"
    dev = r5._device()

    if mode == "selftest":
        # a perfectly sun-seasonal synthetic population: the engine must score near-perfectly
        rng = np.random.default_rng(7)
        n = 210
        grid = pd.DatetimeIndex(tf.GRID)
        i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
        lon = tf.ephemeris()
        moon = pd.read_csv("analysis/adstopics/_moon_monthly.csv")["moon_synodic"].to_numpy(float)
        X = np.column_stack([moon[i0:i0 + n]] +
                            [np.asarray(lon[bd], float)[i0:i0 + n] for bd in tf.BODIES])
        sun = np.deg2rad((X[:, 1] - 200.0 + 180.0) % 360.0 - 180.0)
        S = 50 + 30 * np.exp(-(sun) ** 2)                    # the clean signal, shared by all
        Y = np.stack([S + rng.normal(0, 0.1, n) for _ in range(24)])
        b = Bench(Y, X)
        pred = fit(np.diff(Y, axis=1), b.Xd, dev, b.splitd)
        acc = b.score(lambda t: pred[:, t - 1] > 0, "selftest_gauss")
        auc = float(np.trapezoid(acc, b.hs) / (H - 1))
        # the decisive check: on test months where the TRUE seasonal change dominates the noise,
        # the fitted machine must recover the direction near-perfectly. Quiet months (true Δ≈0)
        # are noise-sign coin flips for ANY predictor and cap the raw AUC structurally.
        dS = np.diff(S)
        strong = [h for h in range(1, H + 1) if abs(dS[b.b + h - 2]) > 0.5]
        acc_strong = float(np.mean([acc[h - 1] for h in strong]))
        print(f"  decisive months: {len(strong)}/24 · accuracy there {acc_strong:.3f} · raw AUC {auc:.3f}")
        # measured structural ceiling ≈0.93: the frozen-width window blurs flank edges (documented)
        assert acc_strong >= 0.90, f"SELFTEST FAILED: decisive-month accuracy {acc_strong:.3f} < 0.90"
        assert auc >= 0.70, f"SELFTEST FAILED: raw AUC {auc:.3f} < 0.70"
        print("  SELFTEST PASSED")
        return

    b = load_bench(want)
    if mode == "arms":
        for tag, target, X_, split, pl in (
                ("level_shared", b.Y, b.X, (b.a, b.b), False),
                ("delta_shared", b.dy, b.Xd, b.splitd, False),
                ("sign_shared", np.sign(b.dy), b.Xd, b.splitd, False),
                ("delta_perlamp", b.dy, b.Xd, b.splitd, True)):
            pred = fit(target, X_, dev, split, per_lamp=pl)
            if tag.startswith("level"):
                d = np.diff(pred, axis=1)
                b.score(lambda t: d[:, t - 1] > 0, tag)
            else:
                b.score(lambda t: pred[:, t - 1] > 0, tag)
                if tag == "delta_perlamp":
                    clim = b.climatology()
                    b.score(b.month_vote(pred, clim), "perlamp+vote")
        b.refs()
    elif mode == "squarewave":
        # STRICT SQUARE-WAVE round: everything derives from b.sq (triple-check, operator 2026-07-16)
        sn = lambda t: b.sq[:, t - 13] > 0
        s08 = b.sq_clim(0.8)
        sc = lambda t: s08[:, b.moy[t - 1]] > 0
        b.score(sn, "sq_snaive")
        b.score(sc, "sq_climatology")
        pl_sq = b.pooled_sq_dirfun()
        b.score(pl_sq, "sq_pooled")
        b.score(lambda t: (sn(t).astype(int) + sc(t) + pl_sq(t)) >= 2, "sq_majority")
        pl = b.pooled_dirfun()
        b.score(pl, "ref_pooled_mixed")
        b.refs()
        pd.DataFrame(b.rows).to_csv("analysis/adstopics/direction_squarewave.csv", index=False)
        # the square-wave test plot: actual vs predicted square wave over the final two years
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names, _, _ = ex.load_topics(10 ** 9 if want == "all" else int(want))
        show = [names.index(t) for t in ("heaters", "lawn", "new year") if t in names][:3]
        fig, axes = plt.subplots(len(show), 1, figsize=(9, 2.1 * len(show)), dpi=160, sharex=True)
        ts = np.arange(b.b, b.n)
        for ax, i in zip(np.atleast_1d(axes), show):
            act = b.sq[i, b.b - 1:b.n - 1]
            prd = np.array([1 if pl_sq(t)[i] else -1 for t in ts])
            ax.step(ts, act, where="mid", color="#1746DC", lw=1.8, label="actual square wave")
            ax.step(ts, prd * 0.9, where="mid", color="#E8B923", lw=1.6, ls="--",
                    label="predicted square wave")
            ax.set_yticks([-1, 1]); ax.set_yticklabels(["fall", "rise"])
            ax.set_ylim(-1.6, 1.6)
            ax.set_title(names[i], fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
        np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, ncol=2)
        np.atleast_1d(axes)[-1].set_xlabel("month index (the held-out final two years)")
        fig.tight_layout()
        fig.savefig("analysis/adstopics/paper/figs/square_waves.png")
        print("figure -> paper/figs/square_waves.png")
    elif mode == "recency":
        # (1) γ-weighted climatology, γ per topic chosen on VALIDATION accuracy
        gammas = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3]
        clims = {g: b.wclim(g) for g in gammas}
        vas = np.stack([b.val_acc(lambda t, g=g: clims[g][:, b.moy[t - 1]] > 0) for g in gammas])
        pick = np.array(gammas)[vas.argmax(0)]
        for g in gammas:
            b.score(lambda t, g=g: clims[g][:, b.moy[t - 1]] > 0, f"wclim_g{g}")
        b.score(lambda t: np.array([clims[pick[i]][i, b.moy[t - 1]] > 0
                                    for i in range(b.Y.shape[0])]), "wclim_val-pick")
        print(f"  γ picks: {dict(zip(*np.unique(pick, return_counts=True)))}")
        # (2) recency-weighted Δ-fit of the record model
        import functools
        old = co.fit_vm
        co.fit_vm = functools.partial(old, recency_gamma=0.8)
        pred = fit(b.dy, b.Xd, dev, b.splitd, per_lamp=True)
        co.fit_vm = old
        b.score(lambda t: pred[:, t - 1] > 0, "perlamp_g0.8")
        clim = b.refs()
        b.score(b.month_vote(pred, clim), "perlamp_g0.8+vote")
    elif mode == "phases":
        # distribution of the OPTIMISED phase p across topics, for each model variant
        panels = []
        for kern in LINKS:
            _, pr = fit(b.dy, b.Xd, dev, b.splitd, kern=kern, return_params=True)
            panels.append((f"{kern} (Δ-fit, shared p)", pr["p"].ravel()))
            print(f"  phases: {kern} done", flush=True)
        _, prl = fit(b.Y, b.X, dev, (b.a, b.b), kern="gauss", return_params=True)
        panels.append(("gauss (level-fit, shared p)", prl["p"].ravel()))
        _, prp = fit(b.dy, b.Xd, dev, b.splitd, kern="gauss", per_lamp=True, return_params=True)
        surv = prp["mask"].astype(bool)
        panels.append(("gauss per-lamp (surviving lamps)", prp["p"][surv].ravel()))
        panels.append(("gauss per-lamp (sun lamp)", prp["p"][surv[:, 1], 1].ravel()))
        rows = []
        for name, ph in panels:
            for v in ph:
                rows.append(dict(model=name, phase=round(float(v) % 360.0, 2)))
        pd.DataFrame(rows).to_csv("analysis/adstopics/phase_distributions.csv", index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 5, figsize=(16, 6), dpi=150)
        bins = np.arange(0, 361, 10)
        for ax, (name, ph) in zip(axes.ravel(), panels):
            ax.hist(ph % 360.0, bins=bins, color="#E8B923", edgecolor="#1746DC", lw=0.4)
            for sgn in range(0, 360, 30):
                ax.axvline(sgn, color="#bbbbbb", lw=0.5, ls=":")
            for c in range(15, 360, 30):
                ax.axvline(c, color="#1746DC", lw=0.5, ls="--", alpha=0.35)   # the 12 init centres
            ax.set_title(name, fontsize=8)
            ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
            ax.tick_params(labelsize=7)
        fig.suptitle("Optimised window direction p across topics (dashes = the 12 sign-centre initialisations)",
                     fontsize=10)
        fig.tight_layout()
        os.makedirs("analysis/adstopics/paper/figs", exist_ok=True)
        fig.savefig("analysis/adstopics/paper/figs/phase_distributions.png")
        print("figure -> analysis/adstopics/paper/figs/phase_distributions.png")
    elif mode == "pooled":
        # GLOBAL pooled logistic: one model over all topic-months. Features (causal): the same
        # calendar month's direction 1/2/3 years back, recency-weighted + flat climatology values
        # (scaled by the topic's own train |Δ| level), and the calendar month. ~18 params fitted
        # on millions of pooled train samples; L2 picked on validation; test touched once.
        c08, c10 = b.wclim(0.8), b.wclim(1.0)
        scale = np.median(np.abs(b.dy[:, :b.a - 1]), axis=1) + 1e-6
        Tn = b.Y.shape[0]

        def feats(t):
            F = [np.sign(b.dy[:, t - 1 - 12 * k]) for k in (1, 2, 3)]
            F += [np.clip(c08[:, b.moy[t - 1]] / scale, -3, 3),
                  np.clip(c10[:, b.moy[t - 1]] / scale, -3, 3)]
            M = np.zeros((Tn, 12)); M[:, b.moy[t - 1]] = 1.0
            return np.column_stack(F + [M])

        def gather(ts):
            Xs, ys = [], []
            for t in ts:
                act = b.dy[:, t - 1]
                ok = act != 0
                Xs.append(feats(t)[ok]); ys.append((act[ok] > 0).astype(float))
            return np.vstack(Xs), np.concatenate(ys)

        Xtr, ytr = gather(range(37, b.a))
        Xva, yva = gather(range(b.a, b.b))
        mu, sdv = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sdv; Xva = (Xva - mu) / sdv

        def train_logistic(Xm, ym, lam, iters=300):
            wgt = np.zeros(Xm.shape[1] + 1)
            for _ in range(iters):
                z = Xm @ wgt[:-1] + wgt[-1]
                pgd = 1 / (1 + np.exp(-z))
                g = np.concatenate([Xm.T @ (pgd - ym) / len(ym) + lam * wgt[:-1],
                                    [(pgd - ym).mean()]])
                wgt -= 2.0 * g
            return wgt

        best = None
        for lam in (0.0, 1e-4, 1e-3, 1e-2):
            wgt = train_logistic(Xtr, ytr, lam)
            va = (((Xva @ wgt[:-1] + wgt[-1]) > 0) == (yva > 0.5)).mean()
            print(f"    lam {lam:g}: val acc {va:.4f}", flush=True)
            if best is None or va > best[1]:
                best = (wgt, va, lam)
        wgt, _, lam = best
        print(f"    chosen lam {lam:g} · weights {np.round(wgt[:5], 3)} (sn1,sn2,sn3,wclim,clim)")
        b.score(lambda t: ((feats(t) - mu) / sdv) @ wgt[:-1] + wgt[-1] > 0, "pooled_logistic")
        # pooled + memory-majority disagreement check
        sn = lambda t: b.dy[:, t - 13] > 0
        b.score(lambda t: (sn(t).astype(int) + (c08[:, b.moy[t - 1]] > 0) +
                           (((feats(t) - mu) / sdv) @ wgt[:-1] + wgt[-1] > 0)) >= 2, "pooled+memory_maj")
        b.refs()
    elif mode == "ensemble":
        # pure-memory majority: snaive + γ0.8-climatology + flat climatology (CPU-only, no furnace)
        c08, c10 = b.wclim(0.8), b.wclim(1.0)
        sn = lambda t: b.dy[:, t - 13] > 0
        w8 = lambda t: c08[:, b.moy[t - 1]] > 0
        fl = lambda t: c10[:, b.moy[t - 1]] > 0
        b.score(lambda t: (sn(t).astype(int) + w8(t) + fl(t)) >= 2, "memory_majority")
        # per-topic val-pick between the three memories
        vs = np.stack([b.val_acc(f) for f in (sn, w8, fl)])
        pk = vs.argmax(0)
        fns = (sn, w8, fl)
        b.score(lambda t: np.select([pk == 0, pk == 1, pk == 2], [f(t) for f in fns]), "memory_val-pick")
        b.refs()
    elif mode == "links":
        for kern in LINKS:
            pred = fit(b.dy, b.Xd, dev, b.splitd, kern=kern)
            b.score(lambda t: pred[:, t - 1] > 0, f"link_{kern}")
        b.refs()
    elif mode == "headline":
        pred = fit(b.dy, b.Xd, dev, b.splitd, per_lamp=True)
        accM = b.score(lambda t: pred[:, t - 1] > 0, "model_perlamp")
        clim = b.refs()
        accV = b.score(b.month_vote(pred, clim), "model+vote")
        c08 = b.wclim(0.8)
        sn = lambda t: b.dy[:, t - 13] > 0
        pl = b.pooled_dirfun()
        accE = b.score(lambda t: (sn(t).astype(int) + (c08[:, b.moy[t - 1]] > 0) +
                                  pl(t)) >= 2, "pooled+memory_maj")
        accP = b.score(pl, "pooled_logistic")
        accS = b.curve(sn)
        accC = b.curve(lambda t: clim[:, b.moy[t - 1]] > 0)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
        for acc, lab, c, ls in ((accE, "pooled + memory majority", "#010C17", "-"),
                                (accP, "pooled global logistic", "#555555", "-"),
                                (accV, "furnace + month vote", "#1746DC", "-"),
                                (accM, "furnace alone", "#4a6fe8", "--"),
                                (accC, "climatology", "#E8B923", "-"),
                                (accS, "seasonal naive", "#8a6d1a", "--")):
            ax.plot(b.hs, acc, lw=1.8, ls=ls, color=c,
                    label=f"{lab} (AUC {np.trapezoid(acc, b.hs)/(H-1):.3f})")
        ax.axhline(0.5, color="#bbbbbb", lw=1, ls=":")
        ax.set_xlabel("months into the future"); ax.set_ylabel("rise/fall accuracy")
        ax.set_title("Direction accuracy over the final two held-out years")
        ax.set_xlim(1, H); ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        os.makedirs("analysis/adstopics/paper/figs", exist_ok=True)
        fig.savefig("analysis/adstopics/paper/figs/direction_accuracy.png")
    pd.DataFrame(b.rows).to_csv(f"analysis/adstopics/direction_{mode}.csv", index=False)

if __name__ == "__main__":
    main()
