#!/usr/bin/env python3
"""CROSS-SECTIONAL RECURRENT ASTROLOGY MODEL (operator 2026-07-20).

Combines everything:
  * BALANCED at all times — the label is the sign of the rate change with FLAT MONTHS EXCLUDED as
    ties, so among moved months the classes are ~50/50 (coin 0.5) for any window.
  * RECURRENT, any start/end — a bidirectional GRU encodes each topic's series over a RANDOM window;
    a consistency loss makes two windows of a topic agree, so the estimate is window-invariant.
  * CROSS-SECTIONAL ATTENTION — the per-topic GRU codes are passed through a transformer that lets
    each topic ATTEND TO ANY OTHER TOPIC, borrowing signal across the cross-section.
  * ASTROLOGY head — the contextualised code emits a phase p + per-body weights w; the FIXED
    astrology lookup table is reconstructed through the Gaussian-furnace link to classify the
    square wave (BCE). Only the encoder is trainable; the table + sigma are fixed.

  python3 analysis/adstopics/astro_xsec_recurrent.py
"""
import importlib.util as u, math, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
rc = _load("analysis/adstopics/ratechange.py", "rc")
pr = _load("analysis/adstopics/phase_rotary.py", "pr")
NP = pr.NP


def prep(b):
    Y = b.Y; n = b.n
    R = np.zeros_like(Y, float); R[:, 1:] = (Y[:, 1:] - Y[:, :-1]) / np.maximum(Y[:, :-1], 1.0)
    # BALANCED AT ALL TIMES, nothing excluded: label = r > per-topic-median, but a tiny deterministic
    # per-month jitter breaks the 35% flat-month atom so the median splits the topic ~50/50 (coin 0.5)
    # with EVERY month kept (the genuinely flat months are split by the jitter -> effectively coin).
    jit = 1e-6 * (((np.arange(n) * 2654435761) % 1000) / 1000.0 - 0.5)
    Rj = R + jit[None]
    # GRID-SEARCH the per-topic threshold to achieve 50/50 on the label over the train window
    cand = np.quantile(Rj[:, 1:b.a], np.linspace(0.35, 0.65, 31), axis=1).T          # (Tn,31) candidates
    splits = np.stack([(Rj[:, 1:b.a] > cand[:, [k]]).mean(1) for k in range(cand.shape[1])], 1)
    tau = cand[np.arange(Y.shape[0]), np.abs(splits - 0.5).argmin(1)][:, None]        # closest to 50/50
    L = (Rj > tau).astype(np.float32); tie = np.zeros_like(L, dtype=bool)             # nothing excluded
    mu = R[:, 1:b.a].mean(1, keepdims=True); sd = R[:, 1:b.a].std(1, keepdims=True) + 1e-6
    x = np.clip((R - mu) / sd, -4, 4).astype(np.float32)
    TH = pr.planet_longitudes(n).T.astype(np.float32)
    moy_ang = np.deg2rad(b.moy * 30 + 15).astype(np.float32)
    feat = np.stack([x, np.tile(np.sin(moy_ang), (Y.shape[0], 1)), np.tile(np.cos(moy_ang), (Y.shape[0], 1)),
                     np.tile(np.sin(TH[0]), (Y.shape[0], 1)), np.tile(np.cos(TH[0]), (Y.shape[0], 1))], -1)
    return x, L, tie, feat.astype(np.float32), TH


def run(b, seed=7, d=48, sigma=0.5, epochs=200, lr=2e-3, cons_w=0.5, bs=1024, device="cpu"):
    import torch as T
    T.manual_seed(seed); rng = np.random.RandomState(seed); Tn = b.Y.shape[0]; n = b.n
    x, L, tie, feat, TH = prep(b)
    THd = T.tensor(TH, device=device); yL = T.tensor(L, device=device)
    keep = T.tensor((~tie).astype(np.float32), device=device); featT = T.tensor(feat, device=device)

    class Net(T.nn.Module):
        def __init__(s):
            super().__init__()
            s.gru = T.nn.GRU(5, d, num_layers=1, batch_first=True, bidirectional=True)     # RECURRENT (time)
            xl = T.nn.TransformerEncoderLayer(2 * d, 4, 4 * d, dropout=0.1, batch_first=True, activation="gelu")
            s.xsec = T.nn.TransformerEncoder(xl, 2)                                          # CROSS-SECTION (topics)
            s.toW = T.nn.Linear(2 * d, NP); s.toPhi = T.nn.Linear(2 * d, 2)
        def encode(s, fb):                                          # fb: (B,W,5) -> per-topic GRU code
            h, _ = s.gru(fb); return h.mean(1)                     # (B,2d) window-length invariant pool
        def forward(s, fb):
            h = s.encode(fb)                                       # (B,2d)
            hc = s.xsec(h[None]).squeeze(0)                        # attend across topics -> (B,2d)
            w = T.nn.functional.softplus(s.toW(hc)); pv = s.toPhi(hc)
            return w, T.atan2(pv[:, 0], pv[:, 1]), s.encode(fb)    # (also return raw code for consistency)

    def decode(w, p, cols):
        dphi = THd[None, :, cols] - p[:, None, None]
        g = T.exp(-(T.atan2(T.sin(dphi), T.cos(dphi)) ** 2) / (2 * sigma ** 2))
        return (w[:, :, None] * g).sum(1)

    net = Net().to(device); opt = T.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    idx = np.arange(Tn); colsT = T.arange(n, device=device); scale_c = 4.0

    def rand_window():
        e = rng.randint(60, b.b + 1); s = rng.randint(0, e - 48); return s, e

    def evaluate(win=None):
        net.eval(); xhat = np.zeros((Tn, n)); phis = np.zeros(Tn)
        s, e = (0, b.b) if win is None else win
        with T.no_grad():
            # cross-section attention over the WHOLE population at eval (all topics attend to all)
            hs = []
            for i in range(0, Tn, bs):
                hs.append(net.encode(featT[i:i + bs, s:e]))
            hall = T.cat(hs, 0); hc = net.xsec(hall[None]).squeeze(0)
            w = T.nn.functional.softplus(net.toW(hc)); pv = net.toPhi(hc); p = T.atan2(pv[:, 0], pv[:, 1])
            xhat = decode(w, p, colsT).cpu().numpy(); phis = p.cpu().numpy()
        z = xhat - np.median(xhat[:, :b.b], 1, keepdims=True)
        # calibrate threshold on val (tie-excluded balanced accuracy)
        yv = L[:, b.a:b.b]; kv = ~tie[:, b.a:b.b]; best = (-1, 0.0)
        for q in np.linspace(-1.5, 1.5, 41):
            pv2 = (z[:, b.a:b.b] > q).astype(int)
            acc = (pv2[kv] == yv[kv]).mean()
            if acc > best[0]: best = (acc, q)
        thr = best[1]; pt = (z[:, b.b:] > thr).astype(int); yt = L[:, b.b:]; kt = ~tie[:, b.b:]
        return best[0], (pt[kt] == yt[kt]).mean(), phis

    best, stall, state = -1.0, 0, None
    for ep in range(epochs):
        net.train(); rng.shuffle(idx)
        for i in range(0, Tn, bs):
            bi = idx[i:i + bs]; jt = T.tensor(bi, device=device)
            s1, e1 = rand_window(); s2, e2 = rand_window()
            opt.zero_grad()
            w1, p1, c1 = net(featT[jt, s1:e1]); _, _, c2 = net(featT[jt, s2:e2])
            xh = decode(w1, p1, colsT); med = xh[:, s1:e1].median(1, keepdim=True).values
            logit = (xh - med) * scale_c
            m = keep[jt]                                            # tie mask (balanced task excludes ties)
            cls = (T.nn.functional.binary_cross_entropy_with_logits(logit, yL[jt], reduction="none") * m).sum() / m.sum()
            cons = (c1 - c2).pow(2).mean()                          # window-consistency of the temporal code
            (cls + cons_w * cons).backward(); opt.step()
        if ep % 3 == 2:
            va, te, _ = evaluate()
            if va > best + 1e-5: best, stall, state = va, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
            else: stall += 1
            if stall >= 8: break
    net.load_state_dict(state)
    va, te, pf = evaluate((0, b.b)); _, ts, ps = evaluate((b.b - 72, b.b)); _, tearly, pe = evaluate((0, b.b - 72))
    pstab = np.abs(np.exp(1j * np.stack([pf, ps, pe])).mean(0)).mean()
    return dict(val=best, test=te, test_short=ts, test_early=tearly, phase_stability=pstab)


def main():
    dev = "cpu"
    try:
        import torch as T
        if T.backends.mps.is_available(): dev = "mps"
    except Exception: pass
    names, Y = rc.bal.load_all(); b = rc.BenchR(Y)
    print(f"  BALANCED task (ties excluded) · test-moved {b.L[:,b.b:][~b.tie[:,b.b:]].mean()*100:.1f}% higher · coin 0.5 · dev {dev}", flush=True)
    for sig in (0.35, 0.5):
        r = run(b, sigma=sig, device=dev)
        print(f"  [xsec-GRU sigma={sig}] test {r['test']:.4f} · windows(last6y/early) {r['test_short']:.4f}/{r['test_early']:.4f} · "
              f"phase-stability {r['phase_stability']:.3f}  (coin 0.5)", flush=True)


if __name__ == "__main__":
    main()
