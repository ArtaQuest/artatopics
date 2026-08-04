#!/usr/bin/env python3
"""AstroAttention (operator directive 2026-07-15) — attention that PREDICTS each series' phase,
then CENTRES every input by subtracting it, so one shared head learns in an aligned frame.

    tokens   : the 12 calendar months; token m carries the topic's causal seasonal evidence
               (recency-weighted + flat mean Δ at month m, magnitude-scaled) + [sin θ_m, cos θ_m]
    phase    : a learned probe attends over the tokens; a signed gate aims at the rising flank;
               p̂ = atan2( Σ α·g·sin θ , Σ α·g·cos θ )      (differentiable, circular)
    centring : every angular input is rotated by −p̂ — the target month angle (calendar arm)
               or all 12 celestial body phases (sky arm): z = x − p̂
    head     : ONE shared MLP over the centred features, trained with BCE on ALL topic-months
               pooled — per-topic adaptation with zero per-topic parameters.

Protocol: identical walls to direction.py (train [37,a) · val [a,b) selection/early-stop ·
test = last 24 months touched once); rise/fall accuracy per horizon; normalised AUC global score.

  python3 analysis/adstopics/astro_attention.py selftest      # must recover planted phases
  python3 analysis/adstopics/astro_attention.py arms [N|all]  # A calendar · B sky · ablations + refs
→ analysis/adstopics/astroattention_results.csv
"""
import importlib.util as u, math, os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

tf = _load("analysis/trends_fit.py", "tf")
r5 = _load("analysis/topic500_reference_solution.py", "r5")
dr = _load("analysis/adstopics/direction.py", "dr")

H = dr.H
D_EMB = 16
EPOCHS = int(os.environ.get("AQ_AA_EPOCHS", 400))
SEED = 7


def sign_consistency(b, gamma=0.8):
    """(T,12) recency-weighted mean SIGN of the month's change on train — pure direction evidence."""
    a1 = b.a - 1
    ages = (a1 - 1 - np.arange(a1)) / 12.0
    w = gamma ** ages
    out = np.zeros((b.Y.shape[0], 12))
    for mth in range(12):
        sel = b.moy[:a1] == mth
        out[:, mth] = (np.sign(b.dy[:, :a1][:, sel]) * w[sel]).sum(1) / w[sel].sum()
    return out


def build_tokens(b):
    """(T,12,5) causal token features + (12,) month angles. Profile from the TRAIN window only."""
    c08, c10 = b.wclim(0.8), b.wclim(1.0)
    sc = sign_consistency(b)
    scale = (np.median(np.abs(b.dy[:, :b.a - 1]), axis=1) + 1e-6)[:, None]
    theta = np.deg2rad(np.arange(12) * 30.0 + 15.0)
    F = np.stack([np.clip(c08 / scale, -3, 3), np.clip(c10 / scale, -3, 3), sc,
                  np.tile(np.sin(theta), (b.Y.shape[0], 1)), np.tile(np.cos(theta), (b.Y.shape[0], 1))],
                 axis=2).astype(np.float32)
    return F, theta


class AstroAttention:
    """arm='calendar' centres the target-month angle; arm='sky' centres all 12 body phases."""

    def __init__(self, b, dev, arm="calendar", centre="attn", memory=False,
                 decisive_w=False, recency_g=None, smooth=0.0, harm2=False, anchor=0.0):
        import torch
        self.t = torch
        self.b = b
        self.dev = dev
        self.arm = arm
        self.centre = centre                      # 'attn' | 'none' | 'gauss' (ablations)
        self.memory = memory
        self.decisive_w = decisive_w
        self.recency_g = recency_g
        self.smooth = smooth
        self.harm2 = harm2
        self.anchor = anchor
        torch.manual_seed(SEED)
        F, self.theta = build_tokens(b)
        self.F = torch.tensor(F, device=dev)
        self.th = torch.tensor(self.theta, dtype=torch.float32, device=dev)
        g = torch.Generator().manual_seed(SEED)
        def P(*shape):
            return torch.nn.Parameter(torch.randn(*shape, generator=g).to(dev) * 0.2)
        self.We, self.be = P(5, D_EMB), P(D_EMB)
        self.q, self.u = P(D_EMB), P(D_EMB)
        self.q2, self.u2 = P(D_EMB), P(D_EMB)                    # E1: second-harmonic probe
        nin = self._nin()
        self.W1, self.b1 = P(nin, 32), P(32)
        self.W2, self.b2 = P(32, 1), P(1)
        self.params = [self.We, self.be, self.q, self.u, self.W1, self.b1, self.W2, self.b2]
        if self.harm2:
            self.params += [self.q2, self.u2]
        self.gauss_p = None
        if centre == "gauss":                     # ablation: the furnace's fitted phase instead
            _, pr = dr.fit(b.dy, b.Xd, dev, b.splitd, kern="gauss", return_params=True)
            self.gauss_p = torch.tensor(np.deg2rad(pr["p"].astype(np.float32)), device=dev)

    def _nin(self):
        nb = self.b.X.shape[1]
        base = (6 + 2) if self.arm == "calendar" else (2 * nb + 2)
        return base + (2 if self.harm2 else 0) + (3 if self.memory else 0)

    def phase(self, q=None, u=None, mult=1.0):
        t = self.t
        e = t.tanh(self.F @ self.We + self.be)                   # (T,12,d)
        alpha = t.softmax((e @ (self.q if q is None else q)) / math.sqrt(D_EMB), dim=1)
        gate = t.tanh(e @ (self.u if u is None else u))          # signed aim
        w = alpha * gate
        vs = (w * t.sin(mult * self.th)[None, :]).sum(1)
        vc = (w * t.cos(mult * self.th)[None, :]).sum(1)
        conf = t.sqrt(vs ** 2 + vc ** 2 + 1e-8)
        return t.atan2(vs, vc), conf                             # (T,), (T,)

    def feats(self, ts):
        """Centred features for months ts: (T,len(ts),nin) + labels/masks from the bench."""
        t, b = self.t, self.b
        if self.centre == "attn":
            ph, conf = self.phase()
        elif self.centre == "gauss":
            ph = self.gauss_p; conf = t.ones_like(ph)
        else:
            ph = t.zeros(b.Y.shape[0], device=self.dev); conf = t.ones_like(ph)
        cols = []
        for mth in ts:
            if self.arm == "calendar":
                phi = t.tensor(float(np.deg2rad(b.moy[mth - 1] * 30.0 + 15.0)), device=self.dev) - ph
                ang = [f(k * phi) for k in (1, 2, 3) for f in (t.sin, t.cos)]
            else:
                xb = t.tensor(np.deg2rad(b.X[mth].astype(np.float32)), device=self.dev)
                z = xb[None, :] - ph[:, None]
                nb = b.X.shape[1]
                ang = [t.sin(z)[:, i] for i in range(nb)] + [t.cos(z)[:, i] for i in range(nb)]
            amp = t.tensor((np.abs(b.dy[:, :b.a - 1]).mean(1) / 10.0).astype(np.float32),
                           device=self.dev)
            fs = ang + [conf, amp]
            if self.harm2:
                ph2, _ = self.phase(self.q2, self.u2, mult=2.0)
                th_t = float(np.deg2rad(b.moy[mth - 1] * 30.0 + 15.0))
                psi = 2.0 * th_t - ph2
                fs += [t.sin(psi), t.cos(psi)]
            if self.memory:
                for k in (1, 2, 3):
                    fs.append(t.tensor(np.sign(b.dy[:, mth - 1 - 12 * k]).astype(np.float32),
                                       device=self.dev))
            cols.append(t.stack(fs, dim=1))
        return t.stack(cols, dim=1)                              # (T,M,nin)

    def logits(self, ts):
        h = self.t.relu(self.feats(ts) @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).squeeze(-1)               # (T,M)

    def fit(self):
        t, b = self.t, self.b
        tr = list(range(37, b.a)); va = list(range(b.a, b.b))
        ytr = t.tensor((b.dy[:, [m - 1 for m in tr]] > 0).astype(np.float32), device=self.dev)
        mtr = t.tensor((b.dy[:, [m - 1 for m in tr]] != 0).astype(np.float32), device=self.dev)
        yva = t.tensor((b.dy[:, [m - 1 for m in va]] > 0).astype(np.float32), device=self.dev)
        mva = t.tensor((b.dy[:, [m - 1 for m in va]] != 0).astype(np.float32), device=self.dev)
        opt = t.optim.Adam(self.params, lr=3e-3)
        bce = t.nn.functional.binary_cross_entropy_with_logits
        if self.anchor > 0:                                      # E7: anchor p-hat to the topic's own
            c08 = b.wclim(0.8)                                   # seasonal-peak direction (identifiability)
            th12 = np.deg2rad(np.arange(12) * 30.0 + 15.0)
            pos = np.clip(c08, 0, None)
            avs = (pos * np.sin(th12)[None]).sum(1); avc = (pos * np.cos(th12)[None]).sum(1)
            atgt = t.tensor(np.arctan2(avs, avc).astype(np.float32), device=self.dev)
            awt = t.tensor((np.sqrt(avs ** 2 + avc ** 2) /
                            (np.abs(pos).sum(1) + 1e-6)).astype(np.float32), device=self.dev)
        if self.decisive_w:                                      # E2: weight by move size (clipped)
            sc = (np.median(np.abs(b.dy[:, :b.a - 1]), axis=1) + 1e-6)[:, None]
            mtr = mtr * t.tensor(np.clip(np.abs(b.dy[:, [m - 1 for m in tr]]) / sc, 0, 3)
                                 .astype(np.float32), device=self.dev)
        if self.recency_g:                                       # E3: recency-weighted loss
            ages = np.array([(b.a - m) / 12.0 for m in tr], dtype=np.float32)
            mtr = mtr * t.tensor(self.recency_g ** ages, device=self.dev)[None, :]
        if self.smooth > 0:                                      # E10: label smoothing
            ytr = ytr * (1 - 2 * self.smooth) + self.smooth
        best, best_state, stall = -1.0, None, 0
        for ep in range(EPOCHS):
            opt.zero_grad()
            lo = self.logits(tr)
            loss = (bce(lo, ytr, reduction="none") * mtr).sum() / mtr.sum()
            if self.anchor > 0:
                ph, _ = self.phase()
                loss = loss + self.anchor * (awt * (1.0 - t.cos(ph - atgt))).mean()
            loss.backward()
            opt.step()
            if ep % 5 == 4:
                with t.no_grad():
                    acc = ((((self.logits(va) > 0).float() == yva).float() * mva).sum() / mva.sum()).item()
                if acc > best + 1e-5:
                    best, stall = acc, 0
                    best_state = [p.detach().clone() for p in self.params]
                else:
                    stall += 1
                if stall >= 12:
                    break
        with t.no_grad():
            for p, s in zip(self.params, best_state):
                p.copy_(s)
        return best

    def dirfun(self):
        t = self.t
        with t.no_grad():
            lo = {mth: self.logits([mth])[:, 0].cpu().numpy() for mth in range(self.b.b, self.b.n)}
        return lambda mth: lo[mth] > 0


def selftest(dev):
    """Planted-phase recovery: same bump profile at random phases — p̂ must align with truth."""
    rng = np.random.default_rng(3)
    n, T = 210, 60
    grid = pd.DatetimeIndex(tf.GRID)
    i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
    lon = tf.ephemeris()
    moon = pd.read_csv("analysis/adstopics/_moon_monthly.csv")["moon_synodic"].to_numpy(float)
    X = np.column_stack([moon[i0:i0 + n]] + [np.asarray(lon[bd], float)[i0:i0 + n] for bd in tf.BODIES])
    sun = X[:, 1]
    rho = rng.uniform(0, 360, T)
    Y = np.stack([50 + 30 * np.exp(-np.deg2rad((sun - r + 180) % 360 - 180) ** 2)
                  + rng.normal(0, 0.5, n) for r in rho])
    b = dr.Bench(Y, X)
    m = AstroAttention(b, dev, arm="calendar", centre="attn")
    va = m.fit()
    ph, _ = m.phase()
    ph = np.rad2deg(ph.detach().cpu().numpy()) % 360
    # circular correlation of predicted phase with the planted phase (up to a global offset)
    d = np.deg2rad(ph - rho)
    R = abs(np.exp(1j * d).mean())
    acc = b.curve(m.dirfun()).mean()
    print(f"[selftest] planted-phase alignment R={R:.3f} · val acc {va:.3f} · test mean acc {acc:.3f}")
    assert R > 0.85, f"phase recovery failed: R={R:.3f}"
    print("  SELFTEST PASSED")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "arms"
    want = sys.argv[2] if len(sys.argv) > 2 else "400"
    dev = r5._device()
    if mode == "selftest":
        selftest(dev)
        return
    b = dr.load_bench(want)
    global SEED

    if mode == "atlas":
        # classify EVERY loaded topic by the ensemble-averaged attention phase
        names, _, _ = dr.ex.load_topics(10 ** 9 if want == "all" else int(want))
        vs_sum = np.zeros(b.Y.shape[0]); vc_sum = np.zeros(b.Y.shape[0])
        accs = []
        for sd in (7, 17, 27):
            SEED = sd
            m = AstroAttention(b, dev, arm="calendar", memory=True, anchor=0.3)
            m.fit()
            with m.t.no_grad():
                ph, conf = m.phase()
            ph = ph.detach().cpu().numpy(); cf = conf.detach().cpu().numpy()
            vs_sum += np.sin(ph) * cf; vc_sum += np.cos(ph) * cf
            with m.t.no_grad():
                accs.append({mth: m.logits([mth])[:, 0].cpu().numpy() for mth in range(b.b, b.n)})
        phase = np.rad2deg(np.arctan2(vs_sum, vc_sum)) % 360.0
        conf = np.sqrt(vs_sum ** 2 + vc_sum ** 2) / 3.0
        aa = lambda mth: sum(l[mth] for l in accs) / 3 > 0
        hit = np.zeros(b.Y.shape[0]); cnt = np.zeros(b.Y.shape[0])
        for mth in range(b.b, b.n):
            act = b.dy[:, mth - 1]; ok = act != 0
            hit += ((aa(mth) == (act > 0)) & ok); cnt += ok
        SIGNS = r5.SIGNS
        df = pd.DataFrame(dict(topic=names, phase=np.round(phase, 2),
                               sign=[SIGNS[int(p // 30) % 12] for p in phase],
                               conf=np.round(conf, 4),
                               dir_acc_test=np.round(hit / np.maximum(cnt, 1), 4)))
        df.to_csv("analysis/adstopics/attention_atlas.csv", index=False)
        print(f"[atlas] {len(df)} topics classified by attention phase")
        print(df["sign"].value_counts().to_string())
        return

    if mode == "v5arms":
        pl = b.pooled_dirfun()
        c08v, c10v = b.wclim(0.8), b.wclim(1.0)
        a1 = b.a - 1
        wgt8 = 0.8 ** ((a1 - 1 - np.arange(a1)) / 12.0)
        scv = np.zeros((b.Y.shape[0], 12))
        for mth in range(12):
            sel = b.moy[:a1] == mth
            scv[:, mth] = (np.sign(b.dy[:, :a1][:, sel]) * wgt8[sel]).sum(1) / wgt8[sel].sum()
        scalev = np.median(np.abs(b.dy[:, :a1]), axis=1) + 1e-6
        ampv = np.abs(b.dy[:, :a1]).mean(1) / 10.0

        def tabF(mth):
            m_ = b.moy[mth - 1]
            cols = [np.sign(b.dy[:, mth - 1 - 12 * k]) for k in (1, 2, 3)]
            cols += [np.clip(c08v[:, m_] / scalev, -3, 3), np.clip(c10v[:, m_] / scalev, -3, 3),
                     scv[:, m_], ampv]
            oh = np.zeros((b.Y.shape[0], 12)); oh[:, m_] = 1.0
            return np.column_stack(cols + [oh])

        def gather(ts):
            Xs, ys, ws = [], [], []
            for t_ in ts:
                act = b.dy[:, t_ - 1]; ok = act != 0
                Xs.append(tabF(t_)[ok]); ys.append(act[ok] > 0)
                ws.append(np.clip(np.abs(act[ok]) / scalev[ok], 0, 3))
            return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)

        from sklearn.ensemble import HistGradientBoostingClassifier
        Xtr, ytr, wtr = gather(range(37, b.a))
        Xva, yva, _ = gather(range(b.a, b.b))
        best = None
        for depth in (3, None):
            for it in (150, 400):
                gbm = HistGradientBoostingClassifier(max_depth=depth, max_iter=it,
                                                     early_stopping=False, random_state=7)
                gbm.fit(Xtr, ytr, sample_weight=wtr)
                va = (gbm.predict(Xva) == yva).mean()
                if best is None or va > best[0]:
                    best = (va, gbm)
        gbm = best[1]
        print(f"    S1_gbm: val acc {best[0]:.4f}", flush=True)
        b.score(lambda mth: gbm.predict(tabF(mth)) > 0.5, "S1_gbm")

        ens = {}
        for tag, kw in (("V3_control", dict(memory=True)),
                        ("E1_harm2", dict(memory=True, harm2=True)),
                        ("E2_decisive", dict(memory=True, decisive_w=True)),
                        ("E3_recency", dict(memory=True, recency_g=0.9)),
                        ("E23_both", dict(memory=True, decisive_w=True, recency_g=0.9)),
                        ("E10_smooth", dict(memory=True, smooth=0.1))):
            los = []
            for sd in (7, 17, 27):
                SEED = sd
                m = AstroAttention(b, dev, arm="calendar", **kw)
                m.fit()
                with m.t.no_grad():
                    los.append({mth: m.logits([mth])[:, 0].cpu().numpy()
                                for mth in range(b.b, b.n)})
            ens[tag] = los
            b.score(lambda mth, l=los: sum(x[mth] for x in l) / len(l) > 0, tag)

        b.score(pl, "ref_pooled")
        aa3 = ens["V3_control"]
        b.score(lambda mth: ((sum(x[mth] for x in aa3) / 3 > 0).astype(int) + pl(mth) +
                             (gbm.predict(tabF(mth)) > 0.5).astype(int)) >= 2, "META_aa+pool+gbm")
        b.score(lambda mth: ((sum(x[mth] for x in aa3) / 3 > 0).astype(int) + pl(mth) +
                             (b.dy[:, mth - 13] > 0)) >= 2, "META_aa+pool+snaive")
        b.refs()
        pd.DataFrame(b.rows).to_csv("analysis/adstopics/astroattention_v5.csv", index=False)
        return

    los_by_tag = {}
    for tag, kw in (("AA_cal+memory", dict(arm="calendar", memory=True)),
                    ("abl_no-centre+mem", dict(arm="calendar", centre="none", memory=True)),
                    ("AA_calendar", dict(arm="calendar")),
                    ("AA_sky+memory", dict(arm="sky", memory=True))):
        los = []
        for sd in (7, 17, 27):                                   # 3-seed logit ensemble
            SEED = sd
            m = AstroAttention(b, dev, **kw)
            va = m.fit()
            with m.t.no_grad():
                los.append({mth: m.logits([mth])[:, 0].cpu().numpy()
                            for mth in range(b.b, b.n)})
        print(f"    {tag}: last val acc {va:.4f}", flush=True)
        los_by_tag[tag] = los
        b.score(lambda mth: sum(l[mth] for l in los) / 3 > 0, tag)
    pl = b.pooled_dirfun()
    b.score(pl, "ref_pooled")
    aa = los_by_tag.get("AA_cal+memory")
    if aa is not None:
        b.score(lambda mth: ((sum(l[mth] for l in aa) / 3 > 0).astype(int) + pl(mth) +
                             (b.dy[:, mth - 13] > 0)) >= 2, "META_aa+pooled+snaive")
    b.refs()
    pd.DataFrame(b.rows).to_csv("analysis/adstopics/astroattention_results.csv", index=False)

if __name__ == "__main__":
    main()
