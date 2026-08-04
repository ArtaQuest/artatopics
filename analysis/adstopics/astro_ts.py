#!/usr/bin/env python3
"""AstroAttention-TS (operator spec 2026-07-16) — PURE time-series analysis. No NLP, no topic
identity, no engineered climatology features. The model sees ONLY the series itself:

    input    : the last L values of the topic's SQUARE WAVE (+1 rise / -1 fall, ties hold),
               each lag tagged with its calendar angle theta (sin/cos)
    stage 1  : attention over the time-series tokens predicts the PHASE
               p = atan2( sum a*g*sin(theta), sum a*g*cos(theta) )
    stage 2  : the features are ROTATED by the predicted phase: phi_i = theta_i - p
    stage 3  : a second attention pass over the rotated features, queried by the target month's
               rotated angle, predicts rise/fall.

Simple, intuitive, efficient. Trained from scratch on ALL topics x train months pooled (BCE,
ties masked), validated on the standard walls, tested once on the last 24 months. Benchmarked
against the references; improvements iterate FROM this baseline. (The planted-phase R score is
retired per the operator — task accuracy is the only judge.)

  python3 analysis/adstopics/astro_ts.py [N|all]
→ analysis/adstopics/astro_ts_results.csv
"""
import importlib.util as u, math, os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

r5 = _load("analysis/topic500_reference_solution.py", "r5")
dr = _load("analysis/adstopics/direction.py", "dr")

L = int(os.environ.get("AQ_TS_L", 60))       # lag window (five years default)
D = 16
EPOCHS = int(os.environ.get("AQ_TS_EPOCHS", 150))
SEEDS = (7, 17, 27)


class AstroTS:
    def __init__(self, b, dev, seed, rotate=True, stage3="attn"):
        import torch as T
        self.T, self.b, self.dev = T, b, dev
        self.rotate, self.stage3 = rotate, stage3
        T.manual_seed(seed)
        g = T.Generator().manual_seed(seed)
        P = lambda *s: T.nn.Parameter(T.randn(*s, generator=g).to(dev) * 0.2)
        self.We, self.be = P(4, D), P(D)         # token embed: [sq_i, sin/cos theta_i, cos(theta_i - theta_t)]
        self.q1, self.g1 = P(D), P(D)            # stage-1 phase probe + signed gate
        self.Wr, self.br = P(4, D), P(D)         # rotated-token embed: [sq_i, sin/cos phi_i, cos(theta_i - theta_t)]
        self.q2 = P(2, D)                        # stage-3 query from the target's rotated angle
        self.Wo, self.bo = P(D, 1), P(1)
        self.params = [self.We, self.be, self.q1, self.g1, self.Wr, self.br, self.q2,
                       self.Wo, self.bo]
        # precompute all windows once: for each target month t, lags t-L..t-1 of sq + angles
        n = b.n
        moy_ang = np.deg2rad((np.arange(1, n) % 12) * 30.0 + 15.0)   # angle of each change month
        self.win_sq, self.win_th, self.tgt_th = {}, {}, {}
        for t in range(L + 1, n):
            self.win_sq[t] = b.sq[:, t - 1 - L:t - 1].astype(np.float32)
            self.win_th[t] = moy_ang[t - 1 - L:t - 1].astype(np.float32)
            self.tgt_th[t] = np.float32(moy_ang[t - 1])

    def logits(self, ts):
        T, b = self.T, self.b
        S = T.tensor(np.stack([self.win_sq[t] for t in ts], 1), device=self.dev)   # (T,M,L)
        th = T.tensor(np.stack([self.win_th[t] for t in ts]), device=self.dev)     # (M,L)
        tt = T.tensor(np.array([self.tgt_th[t] for t in ts]), device=self.dev)     # (M,)
        align = T.cos(th[None] - tt[None, :, None]).expand_as(S)   # calendar alignment with the target
        tok = T.stack([S, T.sin(th)[None].expand_as(S), T.cos(th)[None].expand_as(S), align], -1)
        e = T.tanh(tok @ self.We + self.be)                                        # (T,M,L,D)
        a1 = T.softmax((e @ self.q1) / math.sqrt(D), dim=-1)                       # (T,M,L)
        w = a1 * T.tanh(e @ self.g1)
        vs = (w * T.sin(th)[None]).sum(-1); vc = (w * T.cos(th)[None]).sum(-1)     # (T,M)
        p = T.atan2(vs, vc)
        if not self.rotate:
            p = p * 0.0
        phi = th[None] - p[:, :, None]                                             # rotated lags
        rtok = T.stack([S, T.sin(phi), T.cos(phi), align], -1)
        r = T.tanh(rtok @ self.Wr + self.br)                                       # (T,M,L,D)
        tphi = tt[None] - p                                                        # target rotated
        if self.stage3 == "attn":
            q = T.stack([T.sin(tphi), T.cos(tphi)], -1) @ self.q2                  # (T,M,D)
            a2 = T.softmax((r @ q[:, :, None, :].transpose(-1, -2)).squeeze(-1) / math.sqrt(D),
                           dim=-1)                                                 # (T,M,L)
            ctx = (a2[..., None] * r).sum(-2)                                      # (T,M,D)
        else:
            ctx = r.mean(-2)
        return (ctx @ self.Wo).squeeze(-1) + self.bo                               # (T,M)

    def fit(self):
        T, b = self.T, self.b
        tr = list(range(L + 1, b.a)); va = list(range(b.a, b.b))
        ytr = T.tensor((b.dy[:, [m - 1 for m in tr]] > 0).astype(np.float32), device=self.dev)
        mtr = T.tensor((b.dy[:, [m - 1 for m in tr]] != 0).astype(np.float32), device=self.dev)
        yva = T.tensor((b.dy[:, [m - 1 for m in va]] > 0).astype(np.float32), device=self.dev)
        mva = T.tensor((b.dy[:, [m - 1 for m in va]] != 0).astype(np.float32), device=self.dev)
        opt = T.optim.Adam(self.params, lr=3e-3)
        bce = T.nn.functional.binary_cross_entropy_with_logits
        best, state, stall = -1.0, None, 0
        rng = np.random.default_rng(0)
        order = np.arange(len(tr))
        CH = 16                                              # month minibatches (memory-safe)
        for ep in range(EPOCHS):
            rng.shuffle(order)
            for lo_ in range(0, len(order), CH):
                idx = order[lo_:lo_ + CH]
                sub = [tr[i] for i in idx]
                opt.zero_grad()
                ((bce(self.logits(sub), ytr[:, idx], reduction="none") * mtr[:, idx]).sum()
                 / mtr[:, idx].sum().clamp(min=1.0)).backward()
                opt.step()
            with T.no_grad():
                accs = []
                for lo_ in range(0, len(va), 12):
                    sub = va[lo_:lo_ + 12]
                    accs.append((((self.logits(sub) > 0).float() == yva[:, lo_:lo_ + 12]).float()
                                 * mva[:, lo_:lo_ + 12]).sum().item())
                acc = sum(accs) / mva.sum().item()
            if acc > best + 1e-5:
                best, stall, state = acc, 0, [p.detach().clone() for p in self.params]
            else:
                stall += 1
            if stall >= getattr(self, '_patience', 12):
                break
        with T.no_grad():
            for p, s in zip(self.params, state):
                p.copy_(s)
        return best

    def dirfun(self):
        """AUTOREGRESSIVE rollout over the test window (operator 2026-07-16): from the origin,
        the model's own predicted square-wave values replace ground truth in the lag window."""
        T, b = self.T, self.b
        sq_ar = b.sq.copy()                                   # true history up to the origin
        lo = {}
        with T.no_grad():
            for t in range(b.b, b.n):
                self.win_sq[t] = sq_ar[:, t - 1 - L:t - 1].astype(np.float32)
                z = self.logits([t])[:, 0].cpu().numpy()
                lo[t] = z
                sq_ar[:, t - 1] = np.where(z > 0, 1.0, -1.0)  # feed the PREDICTION forward
        return lo


def load_bench_all(return_names=False):
    """ENTIRE topic population (audit-clean; no variance/max gates) — operator 2026-07-16."""
    import json
    tf = _load("analysis/trends_fit.py", "tf")
    grid = pd.DatetimeIndex(tf.GRID)
    i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
    iend = len(grid) - tf.DROP_LAST
    vocab = json.load(open("analysis/adstopics/vocabulary.json"))
    bl = set(json.load(open("analysis/adstopics/blacklist.json")).get("excluded_topics", []))
    Ys, names = [], []
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
        Ys.append(y); names.append(t)
    lon = tf.ephemeris()
    X = np.column_stack([np.asarray(lon[bb], float)[i0:iend] for bb in tf.BODIES])
    b = dr.Bench(np.stack(Ys), X)
    print(f"[astro-ts] ENTIRE population: {len(Ys)} topics · {b.n} months")
    return (b, names) if return_names else b


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else "400"
    dev = r5._device()
    if want == "atlas":
        atlas(dev)
        return
    if want == "ceiling":
        # THE PREDICTABILITY CEILING: oracle bounds no honest model can exceed.
        # O1 test-window climatology oracle: knows each topic's TEST-months seasonal profile.
        # O2 one-year-cross oracle: predicts each test month with the OTHER test year's value.
        # O3 the theoretical cap of any deterministic per-(topic,month) rule on the test window.
        b = load_bench_all()
        def acc_of(dirfun):
            hits = tots = 0
            for t in range(b.b, b.n):
                act = b.dy[:, t - 1]; ok = act != 0
                hits += (dirfun(t) == (act > 0))[ok].sum(); tots += ok.sum()
            return hits / tots
        moy = b.moy
        # O1: per-(topic, calendar month) majority computed ON THE TEST WINDOW (2 samples each)
        test_ts = list(range(b.b, b.n))
        prof = {}
        for t in test_ts:
            m = moy[t - 1]
            prof.setdefault(m, []).append(np.sign(b.dy[:, t - 1]))
        o1 = {m: (np.stack(v).sum(0) >= 0) for m, v in prof.items()}
        print(f"  O1 test-climatology oracle: {acc_of(lambda t: o1[moy[t - 1]]):.4f}")
        # O2: the other test year's same-month value
        def o2(t):
            other = t - 12 if t - 12 >= b.b else t + 12
            return b.dy[:, other - 1] > 0
        print(f"  O2 cross-year oracle:       {acc_of(o2):.4f}")
        # O3: theoretical cap of ANY per-(topic,month) deterministic rule on this window:
        # per (topic, month) the majority of its (up to 2) outcomes → accuracy = mean of
        # max(count(+), count(-)) / count(non-tie)
        num = den = 0
        for m, v in prof.items():
            V = np.stack(v)                      # (years, T)
            pos = (V > 0).sum(0); neg = (V < 0).sum(0)
            num += np.maximum(pos, neg).sum(); den += (pos + neg).sum()
        print(f"  O3 per-(topic,month) cap:   {num / den:.4f}")
        return
    if want == "encdec_v3":
        # TOWARD 65% (operator): the champion ENCDEC_mem + the untapped CROSS-SECTION —
        # neighbour votes from the k most-correlated topics (their pooled same-month vote and
        # last-year mean), a global month bias, and a seed ensemble. Val-selected, honest.
        import torch as T
        b = load_bench_all()
        a1 = b.a - 1
        Tn = b.Y.shape[0]
        W = 36
        s08v = b.sq_clim(0.8)
        g9 = 0.9 ** ((a1 - 1 - np.arange(a1)) / 12.0)
        prof = np.zeros((Tn, 12)); tot = np.zeros(12)
        for mth in range(12):
            sel = b.moy[:a1] == mth
            prof[:, mth] = (b.sq[:, :a1][:, sel] * g9[sel]).sum(1)
            tot[mth] = g9[sel].sum()
        vote = prof / tot[None, :]
        # neighbour graph from TRAIN-window wave correlation (chunked)
        Zw = b.sq[:, :a1].astype(np.float32)
        Zw = (Zw - Zw.mean(1, keepdims=True)) / (Zw.std(1, keepdims=True) + 1e-6)
        K = 20
        nb_idx = np.zeros((Tn, K), dtype=int)
        for lo_ in range(0, Tn, 512):
            Cc = (Zw[lo_:lo_ + 512] @ Zw.T) / a1
            for i in range(Cc.shape[0]):
                Cc[i, lo_ + i] = -2
            nb_idx[lo_:lo_ + 512] = np.argsort(-Cc, 1)[:, :K]
        nb_vote = vote[nb_idx].mean(1)                            # (T,12) neighbours' votes
        print("    neighbour graph built", flush=True)
        moy_ang_all = np.deg2rad((np.arange(1, b.n) % 12) * 30.0 + 15.0)
        dev_ = r5._device()
        tr = list(range(37, b.a)); va = list(range(b.a, b.b))
        ytr = T.tensor((b.dy[:, [m - 1 for m in tr]] > 0).astype(np.float32), device=dev_)
        mtr = T.tensor((b.dy[:, [m - 1 for m in tr]] != 0).astype(np.float32), device=dev_)
        yva = T.tensor((b.dy[:, [m - 1 for m in va]] > 0).astype(np.float32), device=dev_)
        mva = T.tensor((b.dy[:, [m - 1 for m in va]] != 0).astype(np.float32), device=dev_)
        bce = T.nn.functional.binary_cross_entropy_with_logits
        win = b.sq[:, a1 - W:a1].astype(np.float32)
        thw = moy_ang_all[a1 - W:a1].astype(np.float32)

        def run_arm(tag, use_nb, use_mbias, seeds):
            los_all = []
            va_best = 0
            for sd in seeds:
                T.manual_seed(sd)
                gseed = T.Generator().manual_seed(sd)
                P = lambda *sh: T.nn.Parameter(T.randn(*sh, generator=gseed).to(dev_) * 0.2)
                Dh = 16
                We, be, qv, uv = P(3, Dh), P(Dh), P(Dh), P(Dh)
                Wd, bd, Qd = P(3, Dh), P(Dh), P(2, Dh)
                nmem = 3 + (2 if use_nb else 0)
                Wo, bo = P(Dh + nmem, 1), P(1)
                mb = T.zeros(12, device=dev_, requires_grad=True) if use_mbias else None
                tok = T.tensor(np.stack([win, np.tile(np.sin(thw), (Tn, 1)),
                                         np.tile(np.cos(thw), (Tn, 1))], -1), device=dev_)
                thwT = T.tensor(thw, device=dev_)
                sqT = T.tensor(win, device=dev_)

                def encoder():
                    e = T.tanh(tok @ We + be)
                    al = T.softmax((e @ qv) / math.sqrt(Dh), dim=1)
                    gt = T.tanh(e @ uv)
                    ws = al * gt
                    return T.atan2((ws * T.sin(thwT)[None]).sum(1),
                                   (ws * T.cos(thwT)[None]).sum(1))

                def mem(ts_):
                    cols = [np.stack([vote[:, b.moy[t - 1]] for t in ts_], 1),
                            np.stack([np.sign(b.dy[:, t - 13]) for t in ts_], 1),
                            np.stack([s08v[:, b.moy[t - 1]] for t in ts_], 1)]
                    if use_nb:
                        cols.append(np.stack([nb_vote[:, b.moy[t - 1]] for t in ts_], 1))
                        cols.append(np.stack([np.sign(b.dy[nb_idx, t - 13]).mean(1)
                                              if t - 13 >= 0 else np.zeros(Tn) for t in ts_], 1))
                    return T.tensor(np.stack(cols, -1).astype(np.float32), device=dev_)

                def decoder(ts_, p_):
                    phi = thwT[None, :] - p_[:, None]
                    rtok = T.stack([sqT, T.sin(phi), T.cos(phi)], -1)
                    r = T.tanh(rtok @ Wd + bd)
                    th_t = T.tensor(np.deg2rad((np.array([b.moy[t - 1] for t in ts_]) * 30.0 + 15.0))
                                    .astype(np.float32), device=dev_)
                    tphi = th_t[None, :] - p_[:, None]
                    q = T.stack([T.sin(tphi), T.cos(tphi)], -1) @ Qd
                    a2 = T.softmax(T.einsum("twd,tmd->tmw", r, q) / math.sqrt(Dh), dim=-1)
                    ctx = T.einsum("tmw,twd->tmd", a2, r)
                    out = (T.cat([ctx, mem(ts_)], -1) @ Wo).squeeze(-1) + bo
                    if use_mbias:
                        mo = T.tensor(np.array([b.moy[t - 1] for t in ts_]), device=dev_)
                        out = out + mb[mo][None, :]
                    return out

                params = [We, be, qv, uv, Wd, bd, Qd, Wo, bo] + ([mb] if use_mbias else [])
                opt = T.optim.Adam(params, lr=3e-3)
                best, stall, state = -1.0, 0, None
                for ep in range(500):
                    opt.zero_grad()
                    ((bce(decoder(tr, encoder()), ytr, reduction="none") * mtr).sum()
                     / mtr.sum()).backward()
                    opt.step()
                    if ep % 5 == 4:
                        with T.no_grad():
                            acc = (((decoder(va, encoder()) > 0).float() == yva).float() * mva
                                   ).sum().item() / mva.sum().item()
                        if acc > best + 1e-5:
                            best, stall, state = acc, 0, [x.detach().clone() for x in params]
                        else:
                            stall += 1
                        if stall >= 15:
                            break
                with T.no_grad():
                    for x, st in zip(params, state):
                        x.copy_(st)
                    lo = {t: decoder([t], encoder())[:, 0].cpu().numpy()
                          for t in range(b.b, b.n)}
                los_all.append(lo)
                va_best = max(va_best, best)
            print(f"    {tag}: best val {va_best:.4f}", flush=True)
            b.score(lambda t, L=los_all: sum(x[t] for x in L) / len(L) > 0, tag)

        run_arm("V3_nb", True, False, (7,))
        run_arm("V3_nb_mbias", True, True, (7,))
        run_arm("V3_nb_mbias_ens3", True, True, (7, 17, 27))
        def snaive_ar(t):
            return b.sq[:, b.b - 13 + ((t - b.b) % 12)] > 0
        b.score(snaive_ar, "ref_sq_snaive_AR")
        b.score(lambda t: s08v[:, b.moy[t - 1]] > 0, "ref_sq_climatology")
        pd.DataFrame(b.rows).to_csv("analysis/adstopics/encdec_v3_results.csv", index=False)
        return
    if want == "encdec_tf":
        # TRANSFORMER ENCODER-DECODER (operator final architecture): a real transformer.
        # ENCODER: multi-head self-attention over the raw W=36 square-wave window (one sample
        # per series) -> circular phase readout p. ROTATION: the data re-embedded with rotated
        # angles (rotary-style). DECODER: cross-attention — the rotated target-month token
        # queries the rotated sequence -> rise/fall.
        import torch as T
        b = load_bench_all()
        a1 = b.a - 1
        Tn = b.Y.shape[0]
        W = 36
        s08v = b.sq_clim(0.8)
        g9 = 0.9 ** ((a1 - 1 - np.arange(a1)) / 12.0)
        prof = np.zeros((Tn, 12)); tot = np.zeros(12)
        for mth in range(12):
            sel = b.moy[:a1] == mth
            prof[:, mth] = (b.sq[:, :a1][:, sel] * g9[sel]).sum(1)
            tot[mth] = g9[sel].sum()
        vote = prof / tot[None, :]
        moy_ang_all = np.deg2rad((np.arange(1, b.n) % 12) * 30.0 + 15.0)
        dev_ = r5._device()
        tr = list(range(37, b.a)); va = list(range(b.a, b.b))
        ytr = T.tensor((b.dy[:, [m - 1 for m in tr]] > 0).astype(np.float32), device=dev_)
        mtr = T.tensor((b.dy[:, [m - 1 for m in tr]] != 0).astype(np.float32), device=dev_)
        yva = T.tensor((b.dy[:, [m - 1 for m in va]] > 0).astype(np.float32), device=dev_)
        mva = T.tensor((b.dy[:, [m - 1 for m in va]] != 0).astype(np.float32), device=dev_)
        bce = T.nn.functional.binary_cross_entropy_with_logits
        win = b.sq[:, a1 - W:a1].astype(np.float32)
        thw = moy_ang_all[a1 - W:a1].astype(np.float32)
        DM, NH, FF = 32, 4, 64

        class EncDecTF(T.nn.Module):
            def __init__(self, use_mem):
                super().__init__()
                self.use_mem = use_mem
                self.emb_e = T.nn.Linear(3, DM)
                enc_layer = T.nn.TransformerEncoderLayer(DM, NH, FF, batch_first=True,
                                                         dropout=0.1, norm_first=True)
                self.encoder = T.nn.TransformerEncoder(enc_layer, 2)
                self.pool_q = T.nn.Parameter(T.randn(DM) * 0.2)
                self.gate = T.nn.Linear(DM, 1)
                self.emb_d = T.nn.Linear(3, DM)
                dec_layer = T.nn.TransformerDecoderLayer(DM, NH, FF, batch_first=True,
                                                         dropout=0.1, norm_first=True)
                self.decoder = T.nn.TransformerDecoder(dec_layer, 2)
                self.q_emb = T.nn.Linear(2, DM)
                self.out = T.nn.Linear(DM + (3 if use_mem else 0), 1)

            def phase(self, tok, thT):
                h = self.encoder(self.emb_e(tok))                        # (T,W,DM)
                al = T.softmax((h @ self.pool_q) / math.sqrt(DM), dim=1)  # (T,W)
                g = T.tanh(self.gate(h)).squeeze(-1)
                w_ = al * g
                return T.atan2((w_ * T.sin(thT)[None]).sum(1),
                               (w_ * T.cos(thT)[None]).sum(1))

            def forward(self, tok, thT, sqT, th_t, mem=None):
                p = self.phase(tok, thT)                                 # (T,)
                phi = thT[None, :] - p[:, None]
                rtok = T.stack([sqT, T.sin(phi), T.cos(phi)], -1)
                memseq = self.emb_d(rtok)                                # (T,W,DM)
                tphi = th_t[None, :] - p[:, None]                        # (T,M)
                q = self.q_emb(T.stack([T.sin(tphi), T.cos(tphi)], -1))  # (T,M,DM)
                ctx = self.decoder(q, memseq)                            # (T,M,DM)
                if self.use_mem and mem is not None:
                    ctx = T.cat([ctx, mem], -1)
                return self.out(ctx).squeeze(-1), p

        tokT = T.tensor(np.stack([win, np.tile(np.sin(thw), (Tn, 1)),
                                  np.tile(np.cos(thw), (Tn, 1))], -1), device=dev_)
        thT = T.tensor(thw, device=dev_)
        sqT = T.tensor(win, device=dev_)

        def memfeat(ts_):
            f_v = T.tensor(np.stack([vote[:, b.moy[t - 1]] for t in ts_], 1).astype(np.float32), device=dev_)
            f_sn = T.tensor(np.stack([np.sign(b.dy[:, t - 13]) for t in ts_], 1).astype(np.float32), device=dev_)
            f_cl = T.tensor(np.stack([s08v[:, b.moy[t - 1]] for t in ts_], 1).astype(np.float32), device=dev_)
            return T.stack([f_v, f_sn, f_cl], -1)

        def th_of(ts_):
            return T.tensor(np.deg2rad((np.array([b.moy[t - 1] for t in ts_]) * 30.0 + 15.0))
                            .astype(np.float32), device=dev_)

        keep = {}
        for tag, use_mem in (("TF_pure", False), ("TF_mem", True)):
            T.manual_seed(7)
            net = EncDecTF(use_mem).to(dev_)
            opt = T.optim.Adam(net.parameters(), lr=1e-3)
            best, stall, state = -1.0, 0, None
            for ep in range(400):
                net.train()
                opt.zero_grad()
                lg, _ = net(tokT, thT, sqT, th_of(tr), memfeat(tr) if use_mem else None)
                ((bce(lg, ytr, reduction="none") * mtr).sum() / mtr.sum()).backward()
                opt.step()
                if ep % 5 == 4:
                    net.eval()
                    with T.no_grad():
                        lv, _ = net(tokT, thT, sqT, th_of(va), memfeat(va) if use_mem else None)
                        acc = (((lv > 0).float() == yva).float() * mva).sum().item() / mva.sum().item()
                    if acc > best + 1e-5:
                        best, stall = acc, 0
                        state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    else:
                        stall += 1
                    if stall >= 15:
                        break
            net.load_state_dict(state)
            net.eval()
            with T.no_grad():
                lo = {}
                for t in range(b.b, b.n):
                    lg, pfin = net(tokT, thT, sqT, th_of([t]), memfeat([t]) if use_mem else None)
                    lo[t] = lg[:, 0].cpu().numpy()
            print(f"    {tag}: val acc {best:.4f}", flush=True)
            b.score(lambda t, l=lo: l[t] > 0, tag)
            keep[tag] = (best, lo, pfin.cpu().numpy())
        def snaive_ar(t):
            return b.sq[:, b.b - 13 + ((t - b.b) % 12)] > 0
        b.score(snaive_ar, "ref_sq_snaive_AR")
        b.score(lambda t: s08v[:, b.moy[t - 1]] > 0, "ref_sq_climatology")
        pd.DataFrame(b.rows).to_csv("analysis/adstopics/encdec_tf_results.csv", index=False)
        print(f"    winner: {max(keep, key=lambda k: keep[k][0])}")
        return
    if want == "encdec":
        # ENCODER-DECODER (operator spec, final form): the ENCODER estimates the phase of the
        # data (attention over the raw W=36 square-wave window — one sample per series); the
        # data is ROTATED by that phase; the DECODER estimates next month's rise/fall from the
        # rotated data (attention over rotated tokens, queried by the rotated target angle).
        # Arms: the pure form, and +memory concat at the output (allowed experimentation).
        import torch as T
        b = load_bench_all()
        a1 = b.a - 1
        Tn = b.Y.shape[0]
        W = 36
        s08v = b.sq_clim(0.8)
        g9 = 0.9 ** ((a1 - 1 - np.arange(a1)) / 12.0)
        prof = np.zeros((Tn, 12)); tot = np.zeros(12)
        for mth in range(12):
            sel = b.moy[:a1] == mth
            prof[:, mth] = (b.sq[:, :a1][:, sel] * g9[sel]).sum(1)
            tot[mth] = g9[sel].sum()
        vote = prof / tot[None, :]
        moy_ang_all = np.deg2rad((np.arange(1, b.n) % 12) * 30.0 + 15.0)
        dev_ = r5._device()
        tr = list(range(37, b.a)); va = list(range(b.a, b.b))
        ytr = T.tensor((b.dy[:, [m - 1 for m in tr]] > 0).astype(np.float32), device=dev_)
        mtr = T.tensor((b.dy[:, [m - 1 for m in tr]] != 0).astype(np.float32), device=dev_)
        yva = T.tensor((b.dy[:, [m - 1 for m in va]] > 0).astype(np.float32), device=dev_)
        mva = T.tensor((b.dy[:, [m - 1 for m in va]] != 0).astype(np.float32), device=dev_)
        bce = T.nn.functional.binary_cross_entropy_with_logits
        win = b.sq[:, a1 - W:a1].astype(np.float32)
        thw = moy_ang_all[a1 - W:a1].astype(np.float32)
        ang12a = np.deg2rad(np.arange(12) * 30.0 + 15.0)
        posv = np.clip(vote, 0, None)
        p_tgt = np.arctan2((posv * np.sin(ang12a)[None]).sum(1), (posv * np.cos(ang12a)[None]).sum(1))
        p_wt = (np.sqrt(((posv * np.sin(ang12a)[None]).sum(1)) ** 2 +
                        ((posv * np.cos(ang12a)[None]).sum(1)) ** 2) / (np.abs(posv).sum(1) + 1e-6))
        p_tgtT = T.tensor(p_tgt.astype(np.float32), device=dev_)
        p_wtT = T.tensor(p_wt.astype(np.float32), device=dev_)
        keep = {}
        for tag, use_mem, anchor, lagw in (("ENCDEC_mem_anchor", True, 0.3, False),
                                           ("ENCDEC_lagw", True, 0.3, True)):
            T.manual_seed(7)
            gseed = T.Generator().manual_seed(7)
            P = lambda *sh: T.nn.Parameter(T.randn(*sh, generator=gseed).to(dev_) * 0.2)
            Dh = 16
            # encoder
            We, be, qv, uv = P(3, Dh), P(Dh), P(Dh), P(Dh)
            # decoder
            Wd, bd = P(3, Dh), P(Dh)
            Qd = P(2, Dh)
            Wo = P(Dh + (3 if use_mem else 0), 1)
            bo = P(1)
            tok = T.tensor(np.stack([win, np.tile(np.sin(thw), (Tn, 1)),
                                     np.tile(np.cos(thw), (Tn, 1))], -1), device=dev_)
            thwT = T.tensor(thw, device=dev_)
            sqT = T.tensor(win, device=dev_)

            def encoder():
                e = T.tanh(tok @ We + be)
                al = T.softmax((e @ qv) / math.sqrt(Dh), dim=1)
                gt = T.tanh(e @ uv)
                ws = al * gt
                return T.atan2((ws * T.sin(thwT)[None]).sum(1),
                               (ws * T.cos(thwT)[None]).sum(1))          # (T,)

            def decoder(ts_, p_):
                phi = thwT[None, :] - p_[:, None]                        # rotated data angles
                rtok = T.stack([sqT, T.sin(phi), T.cos(phi)], -1)        # (T,W,3)
                r = T.tanh(rtok @ Wd + bd)                               # (T,W,Dh)
                th_t = T.tensor(np.deg2rad((np.array([b.moy[t - 1] for t in ts_]) * 30.0 + 15.0))
                                .astype(np.float32), device=dev_)
                tphi = th_t[None, :] - p_[:, None]                       # (T,M) rotated targets
                q = T.stack([T.sin(tphi), T.cos(tphi)], -1) @ Qd         # (T,M,Dh)
                sc2 = T.einsum("twd,tmd->tmw", r, q) / math.sqrt(Dh)
                if lagw:
                    # WEIGHTS PER ROTATED LAG: learned bias per relative-phase bin + lag-age curve
                    rel = tphi[:, :, None] - phi[:, None, :]
                    binidx = ((T.remainder(rel, 2 * math.pi) / (2 * math.pi) * 12).long()) % 12
                    sc2 = sc2 + rp_bias[binidx] + age_bias[None, None, :]
                a2 = T.softmax(sc2, dim=-1)
                ctx = T.einsum("tmw,twd->tmd", a2, r)                    # (T,M,Dh)
                if use_mem:
                    f_v = T.tensor(np.stack([vote[:, b.moy[t - 1]] for t in ts_], 1).astype(np.float32), device=dev_)
                    f_sn = T.tensor(np.stack([np.sign(b.dy[:, t - 13]) for t in ts_], 1).astype(np.float32), device=dev_)
                    f_cl = T.tensor(np.stack([s08v[:, b.moy[t - 1]] for t in ts_], 1).astype(np.float32), device=dev_)
                    ctx = T.cat([ctx, T.stack([f_v, f_sn, f_cl], -1)], -1)
                return (ctx @ Wo).squeeze(-1) + bo                       # (T,M)

            rp_bias = T.zeros(12, device=dev_, requires_grad=lagw)   # relative-phase bias (rotated-lag weights)
            age_bias = T.zeros(W, device=dev_, requires_grad=lagw)    # learned lag-age curve
            params = [We, be, qv, uv, Wd, bd, Qd, Wo, bo] + ([rp_bias, age_bias] if lagw else [])
            opt = T.optim.Adam(params, lr=3e-3)
            best, stall, state = -1.0, 0, None
            for ep in range(500):
                opt.zero_grad()
                p_cur = encoder()
                loss = ((bce(decoder(tr, p_cur), ytr, reduction="none") * mtr).sum() / mtr.sum())
                if anchor > 0:
                    loss = loss + anchor * (p_wtT * (1.0 - T.cos(p_cur - p_tgtT))).mean()
                loss.backward()
                opt.step()
                if ep % 5 == 4:
                    with T.no_grad():
                        acc = (((decoder(va, encoder()) > 0).float() == yva).float() * mva
                               ).sum().item() / mva.sum().item()
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
                lo = {t: decoder([t], pfin)[:, 0].cpu().numpy() for t in range(b.b, b.n)}
            print(f"    {tag}: val acc {best:.4f}", flush=True)
            b.score(lambda t, l=lo: l[t] > 0, tag)
            keep[tag] = (best, lo, pfin.cpu().numpy())
            if anchor > 0:
                d_ = pfin.cpu().numpy() - p_tgt
                Rg = abs(np.exp(1j * d_[p_wt > np.quantile(p_wt, 0.75)]).mean())
                med = np.median(np.abs(np.degrees(np.angle(np.exp(1j * d_[p_wt > np.quantile(p_wt, 0.75)])))))
                print(f"    {tag}: strong-anchor R={Rg:.3f} · median err {med:.1f}°", flush=True)
        def snaive_ar(t):
            return b.sq[:, b.b - 13 + ((t - b.b) % 12)] > 0
        b.score(snaive_ar, "ref_sq_snaive_AR")
        b.score(lambda t: s08v[:, b.moy[t - 1]] > 0, "ref_sq_climatology")
        pd.DataFrame(b.rows).to_csv("analysis/adstopics/encdec_results.csv", index=False)
        best_tag = max(keep, key=lambda k: keep[k][0])
        np.save("/tmp/encdec_phases.npy", keep[best_tag][2])
        print(f"    winner: {best_tag}")
        if "--atlas" in sys.argv:
            import json as _json
            tag = "ENCDEC_mem_anchor"
            _, lo, phat = keep[tag]
            tf = _load("analysis/trends_fit.py", "tf")
            _, names = load_bench_all(return_names=True)
            SIGNS_ = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio",
                      "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
            ph_deg = np.degrees(phat) % 360.0
            sq_pred = np.zeros((Tn, 24), dtype=int)
            for h, t in enumerate(range(b.b, b.n)):
                sq_pred[:, h] = np.where(lo[t] > 0, 1, -1)
            sq_act = b.sq[:, b.b - 1:b.n - 1].astype(int)
            hits = (sq_pred == sq_act).astype(int)
            conf = np.abs(np.array([np.tanh(lo[t]) for t in range(b.b, b.n)])).mean(0)
            df = pd.DataFrame(dict(topic=names, phase=np.round(ph_deg, 2),
                                   sign=[SIGNS_[int(x // 30) % 12] for x in ph_deg],
                                   conf=np.round(conf, 4),
                                   dir_acc_test=np.round(hits.mean(1), 4)))
            df.to_csv("analysis/adstopics/astro_ts_atlas.csv", index=False)
            print(df["sign"].value_counts().to_string())
            payload = {tf.slug(t): {"sqPred": [int(v) for v in sq_pred[i]],
                                    "sqHit": [int(v) for v in hits[i]]}
                       for i, t in enumerate(names)}
            _json.dump(payload, open("analysis/adstopics/astro_ts_pred.json", "w"))
            print(f"[encdec-atlas {tag}] {len(df)} topics · mean dir-acc {hits.mean():.4f}")
        return
    if want == "phasenet2":
        # PHASENET v2 (operator 2026-07-16): EACH TIME SERIES IS ONE SAMPLE — the estimator
        # ingests the topic's raw square-wave window (the last W months before the train wall)
        # and emits its optimal phase; W is swept. Same shared harmonic head + memory features.
        import torch as T
        b = load_bench_all()
        a1 = b.a - 1
        Tn = b.Y.shape[0]
        s08v = b.sq_clim(0.8)
        g9 = 0.9 ** ((a1 - 1 - np.arange(a1)) / 12.0)
        prof = np.zeros((Tn, 12)); tot = np.zeros(12)
        for mth in range(12):
            sel = b.moy[:a1] == mth
            prof[:, mth] = (b.sq[:, :a1][:, sel] * g9[sel]).sum(1)
            tot[mth] = g9[sel].sum()
        vote = prof / tot[None, :]
        moy_ang_all = np.deg2rad((np.arange(1, b.n) % 12) * 30.0 + 15.0)
        dev_ = r5._device()
        tr = list(range(37, b.a)); va = list(range(b.a, b.b))
        ytr = T.tensor((b.dy[:, [m - 1 for m in tr]] > 0).astype(np.float32), device=dev_)
        mtr = T.tensor((b.dy[:, [m - 1 for m in tr]] != 0).astype(np.float32), device=dev_)
        yva = T.tensor((b.dy[:, [m - 1 for m in va]] > 0).astype(np.float32), device=dev_)
        mva = T.tensor((b.dy[:, [m - 1 for m in va]] != 0).astype(np.float32), device=dev_)
        bce = T.nn.functional.binary_cross_entropy_with_logits
        results = {}
        for W in (24, 36, 60, 96, 120, 156):
            T.manual_seed(7)
            gseed = T.Generator().manual_seed(7)
            P = lambda *sh: T.nn.Parameter(T.randn(*sh, generator=gseed).to(dev_) * 0.2)
            Dh = 16
            We, be, qv, uv = P(3, Dh), P(Dh), P(Dh), P(Dh)
            w = T.zeros(7, device=dev_, requires_grad=True)
            bias = T.zeros(1, device=dev_, requires_grad=True)
            win = b.sq[:, a1 - W:a1].astype(np.float32)               # (T,W) raw wave sample
            thw = moy_ang_all[a1 - W:a1].astype(np.float32)
            tok = T.tensor(np.stack([win, np.tile(np.sin(thw), (Tn, 1)),
                                     np.tile(np.cos(thw), (Tn, 1))], -1), device=dev_)
            thwT = T.tensor(thw, device=dev_)

            def phase_est():
                e = T.tanh(tok @ We + be)
                al = T.softmax((e @ qv) / math.sqrt(Dh), dim=1)
                gt = T.tanh(e @ uv)
                ws = al * gt
                return T.atan2((ws * T.sin(thwT)[None]).sum(1), (ws * T.cos(thwT)[None]).sum(1))

            def feats(ts_, p_):
                th = T.tensor(np.deg2rad((np.array([b.moy[t - 1] for t in ts_]) * 30.0 + 15.0))
                              .astype(np.float32), device=dev_)
                phi = th[None, :] - p_[:, None]
                f_v = T.tensor(np.stack([vote[:, b.moy[t - 1]] for t in ts_], 1).astype(np.float32), device=dev_)
                f_sn = T.tensor(np.stack([np.sign(b.dy[:, t - 13]) for t in ts_], 1).astype(np.float32), device=dev_)
                f_cl = T.tensor(np.stack([s08v[:, b.moy[t - 1]] for t in ts_], 1).astype(np.float32), device=dev_)
                return T.stack([f_v, f_sn, f_cl, T.sin(phi), T.cos(phi),
                                T.sin(2 * phi), T.cos(2 * phi)], -1)

            params = [We, be, qv, uv, w, bias]
            opt = T.optim.Adam(params, lr=3e-3)
            best, stall, state = -1.0, 0, None
            for ep in range(500):
                opt.zero_grad()
                pcur = phase_est()
                ((bce((feats(tr, pcur) @ w) + bias, ytr, reduction="none") * mtr).sum()
                 / mtr.sum()).backward()
                opt.step()
                if ep % 5 == 4:
                    with T.no_grad():
                        acc = (((((feats(va, phase_est()) @ w) + bias > 0).float() == yva).float()
                                * mva).sum() / mva.sum()).item()
                    if acc > best + 1e-5:
                        best, stall, state = acc, 0, [x.detach().clone() for x in params]
                    else:
                        stall += 1
                    if stall >= 15:
                        break
            with T.no_grad():
                for x, st in zip(params, state):
                    x.copy_(st)
                pfin = phase_est()
                lo = {t: ((feats([t], pfin) @ w) + bias)[:, 0].cpu().numpy()
                      for t in range(b.b, b.n)}
            acc_ = b.score(lambda t: lo[t] > 0, f"PN2_W{W}")
            results[W] = (best, lo, [x.cpu().numpy() for x in state], pfin.cpu().numpy())
        def snaive_ar(t):
            return b.sq[:, b.b - 13 + ((t - b.b) % 12)] > 0
        b.score(snaive_ar, "ref_sq_snaive_AR")
        b.score(lambda t: s08v[:, b.moy[t - 1]] > 0, "ref_sq_climatology")
        pd.DataFrame(b.rows).to_csv("analysis/adstopics/phasenet2_results.csv", index=False)
        if "--atlas" in sys.argv:
            import json as _json
            Wbest = max(results, key=lambda k: results[k][0])
            print(f"    atlas from the champion width W={Wbest}", flush=True)
            _, lo, st, phat = results[Wbest]
            wv = st[4]
            tf = _load("analysis/trends_fit.py", "tf")
            _, names = load_bench_all(return_names=True)
            SIGNS_ = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio",
                      "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
            ang12 = np.deg2rad(np.arange(12) * 30.0 + 15.0)
            resp = (wv[3] * np.sin(ang12[None] - phat[:, None]) +
                    wv[4] * np.cos(ang12[None] - phat[:, None]) +
                    wv[5] * np.sin(2 * (ang12[None] - phat[:, None])) +
                    wv[6] * np.cos(2 * (ang12[None] - phat[:, None])))
            pos_ = np.clip(resp, 0, None)
            vs_ = (pos_ * np.sin(ang12)[None]).sum(1); vc_ = (pos_ * np.cos(ang12)[None]).sum(1)
            ph_resp = np.rad2deg(np.arctan2(vs_, vc_)) % 360.0
            sq_pred = np.zeros((Tn, 24), dtype=int)
            for h, t in enumerate(range(b.b, b.n)):
                sq_pred[:, h] = np.where(lo[t] > 0, 1, -1)
            sq_act = b.sq[:, b.b - 1:b.n - 1].astype(int)
            hits = (sq_pred == sq_act).astype(int)
            conf = np.abs(np.array([np.tanh(lo[t]) for t in range(b.b, b.n)])).mean(0)
            df = pd.DataFrame(dict(topic=names, phase=np.round(ph_resp, 2),
                                   sign=[SIGNS_[int(x // 30) % 12] for x in ph_resp],
                                   conf=np.round(conf, 4),
                                   dir_acc_test=np.round(hits.mean(1), 4)))
            df.to_csv("analysis/adstopics/astro_ts_atlas.csv", index=False)
            print(df["sign"].value_counts().to_string())
            payload = {tf.slug(t): {"sqPred": [int(v) for v in sq_pred[i]],
                                    "sqHit": [int(v) for v in hits[i]]}
                       for i, t in enumerate(names)}
            _json.dump(payload, open("analysis/adstopics/astro_ts_pred.json", "w"))
            print(f"[phasenet2-atlas W={Wbest}] {len(df)} topics · mean dir-acc {hits.mean():.4f}")
        return
    if want == "phasenet":
        # THE PHASE ESTIMATOR (operator 2026-07-16): a model that, GIVEN the time series,
        # estimates the optimal phase — amortising the per-topic gradient descent. Encoder:
        # attention over the 12 month-tokens of the series' recency-weighted square-wave profile
        # → circular readout p̂. Trained END-TO-END through the same shared harmonic head, so the
        # estimated phase is optimal-for-prediction; judged on accuracy AND agreement with the
        # per-topic GD phases.
        import torch as T
        b = load_bench_all()
        a1 = b.a - 1
        Tn = b.Y.shape[0]
        g9 = 0.9 ** ((a1 - 1 - np.arange(a1)) / 12.0)
        prof = np.zeros((Tn, 12)); tot = np.zeros(12)
        for mth in range(12):
            sel = b.moy[:a1] == mth
            prof[:, mth] = (b.sq[:, :a1][:, sel] * g9[sel]).sum(1)
            tot[mth] = g9[sel].sum()
        vote = prof / tot[None, :]
        s08v = b.sq_clim(0.8)
        ang12 = np.deg2rad(np.arange(12) * 30.0 + 15.0)
        dev_ = r5._device()
        T.manual_seed(7)
        Dh = 16
        gseed = T.Generator().manual_seed(7)
        P = lambda *sh: T.nn.Parameter(T.randn(*sh, generator=gseed).to(dev_) * 0.2)
        We, be, qv, uv = P(3, Dh), P(Dh), P(Dh), P(Dh)
        w = T.zeros(7, device=dev_, requires_grad=True)
        bias = T.zeros(1, device=dev_, requires_grad=True)
        tok_np = np.stack([vote, np.tile(np.sin(ang12), (Tn, 1)),
                           np.tile(np.cos(ang12), (Tn, 1))], -1).astype(np.float32)
        tok = T.tensor(tok_np, device=dev_)
        thc = T.tensor(ang12.astype(np.float32), device=dev_)

        def phase_est():
            e = T.tanh(tok @ We + be)                          # (T,12,Dh)
            al = T.softmax((e @ qv) / math.sqrt(Dh), dim=1)
            gt = T.tanh(e @ uv)
            wsum = al * gt
            vs = (wsum * T.sin(thc)[None]).sum(1); vc = (wsum * T.cos(thc)[None]).sum(1)
            return T.atan2(vs, vc)

        def feats(ts_, p_):
            th = T.tensor(np.deg2rad((np.array([b.moy[t - 1] for t in ts_]) * 30.0 + 15.0))
                          .astype(np.float32), device=dev_)
            phi = th[None, :] - p_[:, None]
            f_v = T.tensor(np.stack([vote[:, b.moy[t - 1]] for t in ts_], 1).astype(np.float32), device=dev_)
            f_sn = T.tensor(np.stack([np.sign(b.dy[:, t - 13]) for t in ts_], 1).astype(np.float32), device=dev_)
            f_cl = T.tensor(np.stack([s08v[:, b.moy[t - 1]] for t in ts_], 1).astype(np.float32), device=dev_)
            return T.stack([f_v, f_sn, f_cl, T.sin(phi), T.cos(phi), T.sin(2 * phi), T.cos(2 * phi)], -1)

        def logits(ts_):
            return (feats(ts_, phase_est()) @ w) + bias

        tr = list(range(37, b.a)); va = list(range(b.a, b.b))
        ytr = T.tensor((b.dy[:, [m - 1 for m in tr]] > 0).astype(np.float32), device=dev_)
        mtr = T.tensor((b.dy[:, [m - 1 for m in tr]] != 0).astype(np.float32), device=dev_)
        yva = T.tensor((b.dy[:, [m - 1 for m in va]] > 0).astype(np.float32), device=dev_)
        mva = T.tensor((b.dy[:, [m - 1 for m in va]] != 0).astype(np.float32), device=dev_)
        params = [We, be, qv, uv, w, bias]
        opt = T.optim.Adam(params, lr=3e-3)
        bce = T.nn.functional.binary_cross_entropy_with_logits
        best, stall, state = -1.0, 0, None
        for ep in range(500):
            opt.zero_grad()
            ((bce(logits(tr), ytr, reduction="none") * mtr).sum() / mtr.sum()).backward()
            opt.step()
            if ep % 5 == 4:
                with T.no_grad():
                    acc = ((((logits(va) > 0).float() == yva).float() * mva).sum() / mva.sum()).item()
                if acc > best + 1e-5:
                    best, stall, state = acc, 0, [x.detach().clone() for x in params]
                else:
                    stall += 1
                if stall >= 15:
                    break
        with T.no_grad():
            for x, st in zip(params, state):
                x.copy_(st)
            phat = phase_est().cpu().numpy()
            lo = {t: logits([t])[:, 0].cpu().numpy() for t in range(b.b, b.n)}
        print(f"    phasenet: val acc {best:.4f}", flush=True)
        b.score(lambda t: lo[t] > 0, "PHASENET")
        def snaive_ar(t):
            return b.sq[:, b.b - 13 + ((t - b.b) % 12)] > 0
        b.score(lambda t: ((lo[t] > 0).astype(int) + snaive_ar(t) +
                           (s08v[:, b.moy[t - 1]] > 0)) >= 2, "PHASENET_majority")
        b.score(snaive_ar, "ref_sq_snaive_AR")
        b.score(lambda t: s08v[:, b.moy[t - 1]] > 0, "ref_sq_climatology")
        # agreement with the per-topic GD-optimal phases (the ground truth of "optimal")
        try:
            gd = pd.read_csv("analysis/adstopics/gdphase_phases.csv")["phase"].to_numpy()
            d = np.deg2rad(np.rad2deg(phat) % 360.0 - gd)
            R_abs = abs(np.exp(1j * d).mean())
            med = np.median(np.abs(np.angle(np.exp(1j * d)))) * 180 / np.pi
            print(f"    agreement with GD-optimal phases: |R|={R_abs:.3f} · median |Δ|={med:.1f}°", flush=True)
        except Exception as ex:
            print("    (gd phases unavailable:", ex, ")")
        pd.DataFrame(dict(phase=np.round(np.rad2deg(phat) % 360.0, 2))).to_csv(
            "analysis/adstopics/phasenet_phases.csv", index=False)
        pd.DataFrame(b.rows).to_csv("analysis/adstopics/phasenet_results.csv", index=False)
        if "--atlas" in sys.argv:
            import json as _json
            tf = _load("analysis/trends_fit.py", "tf")
            _, names = load_bench_all(return_names=True)
            SIGNS_ = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio",
                      "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
            # convention-free classification: where the model's predicted SEASONAL RESPONSE peaks
            wv = state[4].cpu().numpy()                        # shared head weights (7,)
            resp = (wv[3] * np.sin(ang12[None] - phat[:, None]) +
                    wv[4] * np.cos(ang12[None] - phat[:, None]) +
                    wv[5] * np.sin(2 * (ang12[None] - phat[:, None])) +
                    wv[6] * np.cos(2 * (ang12[None] - phat[:, None])))   # (T,12)
            pos_ = np.clip(resp, 0, None)
            vs_ = (pos_ * np.sin(ang12)[None]).sum(1); vc_ = (pos_ * np.cos(ang12)[None]).sum(1)
            ph_resp = np.rad2deg(np.arctan2(vs_, vc_)) % 360.0
            sq_pred = np.zeros((Tn, 24), dtype=int)
            for h, t in enumerate(range(b.b, b.n)):
                sq_pred[:, h] = np.where(lo[t] > 0, 1, -1)     # the CHAMPION: PHASENET solo
            sq_act = b.sq[:, b.b - 1:b.n - 1].astype(int)
            hits = (sq_pred == sq_act).astype(int)
            conf = np.abs(np.array([np.tanh(lo[t]) for t in range(b.b, b.n)])).mean(0)
            df = pd.DataFrame(dict(topic=names, phase=np.round(ph_resp, 2),
                                   sign=[SIGNS_[int(x // 30) % 12] for x in ph_resp],
                                   conf=np.round(conf, 4),
                                   dir_acc_test=np.round(hits.mean(1), 4)))
            df.to_csv("analysis/adstopics/astro_ts_atlas.csv", index=False)
            print(df["sign"].value_counts().to_string())
            payload = {tf.slug(t): {"sqPred": [int(v) for v in sq_pred[i]],
                                    "sqHit": [int(v) for v in hits[i]]}
                       for i, t in enumerate(names)}
            _json.dump(payload, open("analysis/adstopics/astro_ts_pred.json", "w"))
            print(f"[phasenet-atlas] {len(df)} topics · mean dir-acc {hits.mean():.4f}")
        return
    if want == "gdphase":
        # THE CORE (operator 2026-07-16): each topic's PHASE — its sign — is a parameter
        # optimised by gradient descent. One shared head reads calendar-anchored harmonics of
        # (theta_t - p_i) plus the study's best memory features; sharing the head identifies
        # every phase absolutely. Initialised from the seasonal vote's circular mean; sign =
        # the learned phase's bucket.
        import torch as T
        b = load_bench_all()
        a1 = b.a - 1
        Tn = b.Y.shape[0]
        # memory features (constants w.r.t. GD)
        g9 = 0.9 ** ((a1 - 1 - np.arange(a1)) / 12.0)
        prof = np.zeros((Tn, 12)); tot = np.zeros(12)
        for mth in range(12):
            sel = b.moy[:a1] == mth
            prof[:, mth] = (b.sq[:, :a1][:, sel] * g9[sel]).sum(1)
            tot[mth] = g9[sel].sum()
        vote = prof / tot[None, :]                            # (T,12) in [-1,1]
        s08v = b.sq_clim(0.8)
        # phase init: the vote profile's rise direction (circular mean of positive months)
        ang12 = np.deg2rad(np.arange(12) * 30.0 + 15.0)
        pos = np.clip(vote, 0, None)
        p0 = np.arctan2((pos * np.sin(ang12)[None]).sum(1), (pos * np.cos(ang12)[None]).sum(1))
        dev_ = r5._device()
        p = T.tensor(p0.astype(np.float32), device=dev_, requires_grad=True)
        T.manual_seed(7)
        w = T.zeros(7, device=dev_, requires_grad=True)
        bias = T.zeros(1, device=dev_, requires_grad=True)

        def feats(ts_):
            th = T.tensor(np.deg2rad((np.array([b.moy[t - 1] for t in ts_]) * 30.0 + 15.0))
                          .astype(np.float32), device=dev_)                       # (M,)
            phi = th[None, :] - p[:, None]                                        # (T,M)
            f_v = T.tensor(np.stack([vote[:, b.moy[t - 1]] for t in ts_], 1)
                           .astype(np.float32), device=dev_)
            f_sn = T.tensor(np.stack([np.sign(b.dy[:, t - 13]) for t in ts_], 1)
                            .astype(np.float32), device=dev_)
            f_cl = T.tensor(np.stack([s08v[:, b.moy[t - 1]] for t in ts_], 1)
                            .astype(np.float32), device=dev_)
            return T.stack([f_v, f_sn, f_cl,
                            T.sin(phi), T.cos(phi), T.sin(2 * phi), T.cos(2 * phi)], -1)

        def logits(ts_):
            return (feats(ts_) @ w) + bias

        tr = list(range(37, b.a)); va = list(range(b.a, b.b))
        ytr = T.tensor((b.dy[:, [m - 1 for m in tr]] > 0).astype(np.float32), device=dev_)
        mtr = T.tensor((b.dy[:, [m - 1 for m in tr]] != 0).astype(np.float32), device=dev_)
        yva = T.tensor((b.dy[:, [m - 1 for m in va]] > 0).astype(np.float32), device=dev_)
        mva = T.tensor((b.dy[:, [m - 1 for m in va]] != 0).astype(np.float32), device=dev_)
        opt = T.optim.Adam([{"params": [w, bias], "lr": 3e-3}, {"params": [p], "lr": 2e-2}])
        bce = T.nn.functional.binary_cross_entropy_with_logits
        best, stall, state = -1.0, 0, None
        for ep in range(400):
            opt.zero_grad()
            ((bce(logits(tr), ytr, reduction="none") * mtr).sum() / mtr.sum()).backward()
            opt.step()
            if ep % 5 == 4:
                with T.no_grad():
                    acc = ((((logits(va) > 0).float() == yva).float() * mva).sum() / mva.sum()).item()
                if acc > best + 1e-5:
                    best, stall, state = acc, 0, (p.detach().clone(), w.detach().clone(), bias.detach().clone())
                else:
                    stall += 1
                if stall >= 15:
                    break
        with T.no_grad():
            p.copy_(state[0]); w.copy_(state[1]); bias.copy_(state[2])
        print(f"    gdphase: val acc {best:.4f} · head weights {np.round(state[1].cpu().numpy(), 3)}", flush=True)
        with T.no_grad():
            lo = {t: logits([t])[:, 0].cpu().numpy() for t in range(b.b, b.n)}
        b.score(lambda t: lo[t] > 0, "GDPHASE")
        # ensemble with the vote majority
        def snaive_ar(t):
            return b.sq[:, b.b - 13 + ((t - b.b) % 12)] > 0
        b.score(lambda t: ((lo[t] > 0).astype(int) + snaive_ar(t) +
                           (s08v[:, b.moy[t - 1]] > 0)) >= 2, "GDPHASE_majority")
        b.score(snaive_ar, "ref_sq_snaive_AR")
        b.score(lambda t: s08v[:, b.moy[t - 1]] > 0, "ref_sq_climatology")
        ph_deg = np.rad2deg(state[0].cpu().numpy()) % 360.0
        drift = np.rad2deg(np.abs(np.angle(np.exp(1j * (state[0].cpu().numpy() - p0)))))
        print(f"    phase drift from init: median {np.median(drift):.1f}° · >15°: {(drift > 15).mean()*100:.0f}%")
        pd.DataFrame(dict(phase=np.round(ph_deg, 2))).to_csv("analysis/adstopics/gdphase_phases.csv", index=False)
        pd.DataFrame(b.rows).to_csv("analysis/adstopics/gdphase_results.csv", index=False)
        if "--atlas" in sys.argv:
            import json as _json
            tf = _load("analysis/trends_fit.py", "tf")
            _, names = load_bench_all(return_names=True)
            SIGNS_ = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio",
                      "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
            maj = lambda t: ((lo[t] > 0).astype(int) + snaive_ar(t) +
                             (s08v[:, b.moy[t - 1]] > 0)) >= 2
            sq_pred = np.zeros((Tn, 24), dtype=int)
            for h, t in enumerate(range(b.b, b.n)):
                sq_pred[:, h] = np.where(maj(t), 1, -1)
            sq_act = b.sq[:, b.b - 1:b.n - 1].astype(int)
            hits = (sq_pred == sq_act).astype(int)
            conf = np.abs(np.array([np.tanh(lo[t]) for t in range(b.b, b.n)])).mean(0)
            df = pd.DataFrame(dict(topic=names, phase=np.round(ph_deg, 2),
                                   sign=[SIGNS_[int(x // 30) % 12] for x in ph_deg],
                                   conf=np.round(conf, 4),
                                   dir_acc_test=np.round(hits.mean(1), 4)))
            df.to_csv("analysis/adstopics/astro_ts_atlas.csv", index=False)
            print(df["sign"].value_counts().to_string())
            payload = {tf.slug(t): {"sqPred": [int(v) for v in sq_pred[i]],
                                    "sqHit": [int(v) for v in hits[i]]}
                       for i, t in enumerate(names)}
            _json.dump(payload, open("analysis/adstopics/astro_ts_pred.json", "w"))
            print(f"[gdphase-atlas] {len(df)} topics · mean dir-acc {hits.mean():.4f}")
        return
    if want == "simple":
        # THE SIMPLE RECIPE (operator 2026-07-16): no network, no gradients. For each topic and
        # calendar month, the prediction is a recency-weighted VOTE of that month's past
        # square-wave values, shrunk toward the topic's own base rate when the evidence is thin;
        # each topic's forgetting speed (gamma) and the shrinkage strength (k) are picked on the
        # validation window. Rise if the weighted vote says rise; a dead tie repeats last year.
        b = load_bench_all()
        a1 = b.a - 1
        gammas = [1.0, 0.9, 0.8, 0.7, 0.5]
        ks = [0.0, 1.0, 2.0, 4.0]

        def profile(gamma):
            w = gamma ** ((a1 - 1 - np.arange(a1)) / 12.0)
            up = np.zeros((b.Y.shape[0], 12)); tot = np.zeros(12)
            for mth in range(12):
                sel = b.moy[:a1] == mth
                up[:, mth] = ((b.sq[:, :a1][:, sel] > 0) * w[sel]).sum(1)
                tot[mth] = w[sel].sum()
            return up, tot

        base = (b.sq[:, :a1] > 0).mean(1)                     # each topic's overall rise rate
        profs = {g: profile(g) for g in gammas}

        def dirfun_for(g, k):
            up, tot = profs[g]
            p = (up + k * base[:, None]) / (tot[None, :] + k)
            return lambda t: np.where(np.abs(p[:, b.moy[t - 1]] - 0.5) < 1e-9,
                                      b.sq[:, t - 13] > 0, p[:, b.moy[t - 1]] > 0.5)

        # global k + per-topic gamma, both on VALIDATION only
        def val_acc(f):
            hits = np.zeros(b.Y.shape[0]); cnt = np.zeros(b.Y.shape[0])
            for t in range(b.a, b.b):
                act = b.dy[:, t - 1]; ok = act != 0
                hits += ((f(t) == (act > 0)) & ok); cnt += ok
            return hits / np.maximum(cnt, 1)
        best_k, best_score = None, -1
        for k in ks:
            va = np.stack([val_acc(dirfun_for(g, k)) for g in gammas])
            sc = va.max(0).mean()
            print(f"    k={k}: val {sc:.4f}", flush=True)
            if sc > best_score:
                best_score, best_k, best_va = sc, k, va
        pick = np.array(gammas)[best_va.argmax(0)]
        fs = {g: dirfun_for(g, best_k) for g in gammas}
        def simple(t):
            out = np.zeros(b.Y.shape[0], bool)
            for g in gammas:
                m_ = pick == g
                if m_.any():
                    out[m_] = fs[g](t)[m_]
            return out

        def simple_future(mth):
            out = np.zeros(b.Y.shape[0], bool)
            for g in gammas:
                m_ = pick == g
                if m_.any():
                    up, tot = profs[g]
                    p = (up + best_k * base[:, None]) / (tot[None, :] + best_k)
                    out[m_] = (p[:, mth] > 0.5)[m_]
            return out
        b.score(simple, f"SIMPLE(k={best_k})")
        print(f"    gamma picks: {dict(zip(*np.unique(pick, return_counts=True)))}")
        def snaive_ar(t):
            return b.sq[:, b.b - 13 + ((t - b.b) % 12)] > 0
        b.score(snaive_ar, "ref_sq_snaive_AR")
        s08 = b.sq_clim(0.8)
        b.score(lambda t: s08[:, b.moy[t - 1]] > 0, "ref_sq_climatology")
        # majority of the simple recipe + AR-naive + climatology
        b.score(lambda t: (simple(t).astype(int) + snaive_ar(t) +
                           (s08[:, b.moy[t - 1]] > 0)) >= 2, "SIMPLE_majority")
        pd.DataFrame(b.rows).to_csv("analysis/adstopics/astro_ts_simple.csv", index=False)
        if "--atlas" in sys.argv:
            import json as _json
            tf = _load("analysis/trends_fit.py", "tf")
            _, names = load_bench_all(return_names=True)
            T_ = b.Y.shape[0]
            # per-topic test predictions + hits from the champion majority rule
            sq_pred = np.zeros((T_, 24), dtype=int)
            for h, t in enumerate(range(b.b, b.n)):
                z = (simple(t).astype(int) + snaive_ar(t) + (s08[:, b.moy[t - 1]] > 0)) >= 2
                sq_pred[:, h] = np.where(z, 1, -1)
            sq_act = b.sq[:, b.b - 1:b.n - 1].astype(int)
            hits = (sq_pred == sq_act).astype(int)
            # classification: the recipe's OWN predicted year beyond the data — the growth season
            # is the circular mean of its predicted RISE months (frozen profiles, AR-safe)
            moy_next = (np.arange(b.n, b.n + 12) % 12)
            ang = np.deg2rad(moy_next * 30.0 + 15.0)
            rise = np.zeros((T_, 12))
            for kk in range(12):
                tt_ = b.n + kk
                # emulate the majority at future months (snaive_ar tiles the final year)
                sv = b.sq[:, b.b - 13 + ((tt_ - b.b) % 12)] > 0
                rise[:, kk] = ((simple_future(moy_next[kk]).astype(int) + sv +
                                (s08[:, moy_next[kk]] > 0)) >= 2)
            vs = (rise * np.sin(ang)[None]).sum(1); vc = (rise * np.cos(ang)[None]).sum(1)
            phase = np.rad2deg(np.arctan2(vs, vc)) % 360.0
            conf = np.sqrt(vs ** 2 + vc ** 2) / 12.0
            SIGNS_ = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio",
                      "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
            df = pd.DataFrame(dict(topic=names, phase=np.round(phase, 2),
                                   sign=[SIGNS_[int(p // 30) % 12] for p in phase],
                                   conf=np.round(conf, 4),
                                   dir_acc_test=np.round(hits.mean(1), 4)))
            df.to_csv("analysis/adstopics/astro_ts_atlas.csv", index=False)
            print(df["sign"].value_counts().to_string())
            payload = {tf.slug(t): {"sqPred": [int(v) for v in sq_pred[i]],
                                    "sqHit": [int(v) for v in hits[i]]}
                       for i, t in enumerate(names)}
            _json.dump(payload, open("analysis/adstopics/astro_ts_pred.json", "w"))
            print(f"[simple-atlas] {len(df)} topics · mean dir-acc {hits.mean():.4f}")
        return
    if want == "v2arms":
        b = load_bench_all()
        arms = (("I8_long", dict(), dict(epochs=300, patience=20, seeds=(7, 17, 27))),
                ("I3_D24", dict(D=24), dict(seeds=(7, 17, 27))),
                ("I3b_seeds5", dict(), dict(seeds=(7, 17, 27, 37, 47))))
        preds = {}
        for tag, mk, tk in arms:
            global D, EPOCHS
            oldD, oldE = D, EPOCHS
            D = mk.get("D", D); EPOCHS = tk.get("epochs", EPOCHS)
            los = []
            for sd in tk.get("seeds", SEEDS):
                m = AstroTS(b, dev, sd)
                if "patience" in tk:
                    m._patience = tk["patience"]
                m.fit()
                los.append(m.dirfun())
            D, EPOCHS = oldD, oldE
            preds[tag] = los
            b.score(lambda t, l=los: sum(x[t] for x in l) / len(l) > 0, tag)
        # I7: pure-TS ensemble — model + AR-naive + sq-climatology majority
        s08 = b.sq_clim(0.8)
        best = preds["I3b_seeds5"]
        def snaive_ar(t):
            return b.sq[:, b.b - 13 + ((t - b.b) % 12)] > 0
        b.score(lambda t: ((sum(x[t] for x in best) / len(best) > 0).astype(int) +
                           snaive_ar(t) + (s08[:, b.moy[t - 1]] > 0)) >= 2, "I7_TS_majority")
        b.score(snaive_ar, "ref_sq_snaive_AR")
        b.score(lambda t: s08[:, b.moy[t - 1]] > 0, "ref_sq_climatology")
        pd.DataFrame(b.rows).to_csv("analysis/adstopics/astro_ts_v2.csv", index=False)
        return
    b = load_bench_all() if want == "all" else dr.load_bench(want)
    for tag, kw in (("TS_baseline", dict(rotate=True, stage3="attn")),
                    ("abl_no-rotate", dict(rotate=False, stage3="attn")),
                    ("abl_mean-pool", dict(rotate=True, stage3="mean"))):
        los = []
        for sd in SEEDS:
            m = AstroTS(b, dev, sd, **kw)
            va = m.fit()
            los.append(m.dirfun())
        print(f"    {tag}: last val acc {va:.4f}", flush=True)
        b.score(lambda t, l=los: sum(x[t] for x in l) / len(l) > 0, tag)
    # AR-FAIR references: nothing may read true test values. Seasonal naive under AR = its own
    # rollout, which collapses to tiling the final observed pre-test year. The climatology is
    # already static/causal. (The mixed pooled model is reported as teacher-forced context only.)
    def snaive_ar(t):
        h = t - b.b
        return b.sq[:, b.b - 13 + (h % 12)] > 0
    b.score(snaive_ar, "ref_sq_snaive_AR")
    s08 = b.sq_clim(0.8)
    b.score(lambda t: s08[:, b.moy[t - 1]] > 0, "ref_sq_climatology")
    b.score(b.pooled_dirfun(), "ref_pooled_TF_context")
    pd.DataFrame(b.rows).to_csv("analysis/adstopics/astro_ts_results.csv", index=False)

SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio",
         "Sagittarius", "Capricorn", "Aquarius", "Pisces"]


def atlas(dev):
    """Analyse EVERY topic with the best model (L=60 + alignment): per-topic AR-test predictions
    and hits (the topic-page square-wave chart) + the classification = the model's predicted
    growth season (circular mean of the predicted RISE months over the year beyond the data)."""
    import json
    b, names = load_bench_all(return_names=True)
    los, models = [], []
    for sd in SEEDS:
        m = AstroTS(b, dev, sd)
        va = m.fit()
        models.append(m)
        los.append(m.dirfun())                                # AR rollout over the 24 test months
    print(f"  atlas ensemble trained (last val acc {va:.4f})", flush=True)
    # per-topic AR-test predictions + hits
    T_ = b.Y.shape[0]
    sq_pred = np.zeros((T_, 24), dtype=int)
    for h, t in enumerate(range(b.b, b.n)):
        z = sum(l[t] for l in los) / len(los)
        sq_pred[:, h] = np.where(z > 0, 1, -1)
    sq_act = b.sq[:, b.b - 1:b.n - 1].astype(int)
    hits = (sq_pred == sq_act).astype(int)
    dir_acc = hits.mean(1)
    # classification: roll 12 months BEYOND the data end on each model's own wave
    moy_next = (np.arange(b.n, b.n + 12) % 12)
    votes_s = np.zeros(T_); votes_c = np.zeros(T_)
    for m in models:
        sq_ar = np.concatenate([b.sq.copy(), np.zeros((T_, 12))], 1)
        base_n = b.n
        import torch as Torch
        with Torch.no_grad():
            for k in range(12):
                t = base_n + k
                m.win_sq[t] = sq_ar[:, t - 1 - L:t - 1].astype(np.float32)
                ang = np.float32(np.deg2rad((t % 12) * 30.0 + 15.0))
                m.tgt_th[t] = ang
                m.win_th[t] = np.deg2rad((np.arange(t - L, t) % 12) * 30.0 + 15.0).astype(np.float32)
                z = m.logits([t])[:, 0].cpu().numpy()
                sq_ar[:, t - 1] = np.where(z > 0, 1.0, -1.0)
        pred_year = sq_ar[:, base_n - 1:base_n + 11]          # the coming year's predicted wave
        ang = np.deg2rad(moy_next * 30.0 + 15.0)
        rise = (pred_year > 0).astype(float)
        votes_s += (rise * np.sin(ang)[None]).sum(1)
        votes_c += (rise * np.cos(ang)[None]).sum(1)
    phase = np.rad2deg(np.arctan2(votes_s, votes_c)) % 360.0
    conf = np.sqrt(votes_s ** 2 + votes_c ** 2) / (12 * len(models))
    df = pd.DataFrame(dict(topic=names, phase=np.round(phase, 2),
                           sign=[SIGNS[int(p // 30) % 12] for p in phase],
                           conf=np.round(conf, 4), dir_acc_test=np.round(dir_acc, 4)))
    df.to_csv("analysis/adstopics/astro_ts_atlas.csv", index=False)
    print(df["sign"].value_counts().to_string())
    tf = _load("analysis/trends_fit.py", "tf")
    pred_payload = {tf.slug(t): {"sqPred": [int(v) for v in sq_pred[i]],
                                 "sqHit": [int(v) for v in hits[i]]}
                    for i, t in enumerate(names)}
    json.dump(pred_payload, open("analysis/adstopics/astro_ts_pred.json", "w"))
    print(f"[atlas] {len(df)} topics · mean AR dir-acc {dir_acc.mean():.4f} → astro_ts_atlas.csv + astro_ts_pred.json")


if __name__ == "__main__":
    main()
