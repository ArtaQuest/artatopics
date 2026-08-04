#!/usr/bin/env python3
"""RECURRENT ASTROLOGY AUTOENCODER, time-window robust (operator 2026-07-20).

A GRU encoder (topic-independent — each topic a sample) reads a topic's series over ANY window
[start,end] and emits a single phase p + per-body weights w. The FIXED astrology lookup table is
reconstructed through the Gaussian-furnace link  xhat(t)=sum_i w_i exp(-(theta_i(t)-p)^2/2sig^2);
MSE reconstruction backprops into the ENCODER ONLY.

ROBUST TO ANY START/END: trained on RANDOM windows (random start, random length), and a CONSISTENCY
loss forces two different windows of the same topic to yield the SAME (p,w) — so the phase/weights
are intrinsic to the topic, not artefacts of the observation window. A recurrent encoder naturally
handles variable-length windows and any endpoints.

  python3 analysis/adstopics/astro_recurrent.py
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
    tau = np.median(R[:, 1:b.a], 1, keepdims=True); L = (R > tau).astype(int)
    mu = R[:, 1:b.a].mean(1, keepdims=True); sd = R[:, 1:b.a].std(1, keepdims=True) + 1e-6
    x = np.clip((R - mu) / sd, -4, 4).astype(np.float32)
    TH = pr.planet_longitudes(n).T.astype(np.float32)                 # (NP,n) fixed table (radians)
    moy_ang = np.deg2rad(b.moy * 30 + 15).astype(np.float32)
    # per-step encoder features: [x, sin/cos moy, sin/cos sun-longitude(anchors the sidereal frame)]
    feat = np.stack([x, np.tile(np.sin(moy_ang), (Y.shape[0], 1)), np.tile(np.cos(moy_ang), (Y.shape[0], 1)),
                     np.tile(np.sin(TH[0]), (Y.shape[0], 1)), np.tile(np.cos(TH[0]), (Y.shape[0], 1))], -1)
    return x, L, feat.astype(np.float32), TH


def run(b, seed=7, d=64, sigma=0.5, epochs=200, lr=2e-3, cons_w=0.5, cls_w=0.3, bs=512, device="cpu"):
    import torch as T
    T.manual_seed(seed); rng = np.random.RandomState(seed); Tn = b.Y.shape[0]; n = b.n
    x, L, feat, TH = prep(b)
    THd = T.tensor(TH, device=device); xT = T.tensor(x, device=device); yL = T.tensor(L.astype(np.float32), device=device)
    featT = T.tensor(feat, device=device)

    class Enc(T.nn.Module):
        def __init__(s):
            super().__init__()
            s.gru = T.nn.GRU(5, d, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
            s.toW = T.nn.Linear(2 * d, NP); s.toPhi = T.nn.Linear(2 * d, 2)
        def forward(s, fb):                                          # fb: (B,W,5) any window
            h, _ = s.gru(fb); hp = h.mean(1)                        # pool over time (window-length invariant)
            w = T.nn.functional.softplus(s.toW(hp))
            pv = s.toPhi(hp); p = T.atan2(pv[:, 0], pv[:, 1])
            return w, p

    def decode(w, p, cols):
        dphi = THd[None, :, cols] - p[:, None, None]                # (B,NP,|cols|)
        g = T.exp(-(T.atan2(T.sin(dphi), T.cos(dphi)) ** 2) / (2 * sigma ** 2))
        return (w[:, :, None] * g).sum(1)                           # (B,|cols|)

    net = Enc().to(device); opt = T.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    idx = np.arange(Tn)

    def rand_window():
        e = rng.randint(60, b.b + 1); s = rng.randint(0, e - 48)    # random start & end within pre-wall
        return s, e

    def evaluate(win=None):
        net.eval(); xhat = np.zeros((Tn, n)); phis = np.zeros(Tn)
        s, e = (0, b.b) if win is None else win
        cols = T.arange(n, device=device)
        with T.no_grad():
            for i in range(0, Tn, bs):
                j = slice(i, min(i + bs, Tn))
                w, p = net(featT[j, s:e]); xhat[j] = decode(w, p, cols).cpu().numpy(); phis[j] = p.cpu().numpy()
        z = xhat - np.median(xhat[:, :b.b], 1, keepdims=True)
        best = (-1, 0.0)                                             # calibrate threshold on BALANCED val accuracy
        yv = L[:, b.a:b.b]
        for q in np.linspace(-1.5, 1.5, 41):
            pv = (z[:, b.a:b.b] > q).astype(int)
            rec1 = (pv[yv == 1] == 1).mean() if (yv == 1).any() else 0
            rec0 = (pv[yv == 0] == 0).mean() if (yv == 0).any() else 0
            bal = 0.5 * (rec1 + rec0)
            if bal > best[0]: best = (bal, q)
        thr = best[1]
        pt = (z[:, b.b:] > thr).astype(int); yt = L[:, b.b:]
        raw = (pt == yt).mean(); balt = 0.5 * ((pt[yt == 1] == 1).mean() + (pt[yt == 0] == 0).mean())
        return best[0], raw, balt, phis        # (val-balanced for selection, raw test, balanced test, phases)

    best, stall, state = -1.0, 0, None
    colsT = T.arange(n, device=device); scale_c = 4.0        # fixed logit scale (only the encoder is trainable)
    ptr = float(L[:, 1:b.a].mean()); posw = T.tensor((1 - ptr) / (ptr + 1e-9), dtype=T.float32, device=device)   # class balance
    for ep in range(epochs):
        net.train(); rng.shuffle(idx)
        for i in range(0, Tn, bs):
            bi = idx[i:i + bs]; jt = T.tensor(bi, device=device)
            s1, e1 = rand_window(); s2, e2 = rand_window()
            opt.zero_grad()
            w1, p1 = net(featT[jt, s1:e1]); w2, p2 = net(featT[jt, s2:e2])
            # RECONSTRUCT THE SQUARE WAVE with a CLASSIFICATION loss (operator): the furnace output,
            # centred at its per-topic median, is the logit for the +/-1 direction label.
            xh = decode(w1, p1, colsT); med = xh[:, s1:e1].median(1, keepdim=True).values
            logit = (xh - med) * scale_c
            cls = T.nn.functional.binary_cross_entropy_with_logits(logit[:, s1:e1], yL[jt, s1:e1],
                                                                   pos_weight=posw)   # class-balanced
            # CONSISTENCY: same topic, two windows -> same phase & weights (window-robust)
            cons = (1 - T.cos(p1 - p2)).mean() + (w1 - w2).pow(2).mean()
            (cls + cons_w * cons).backward(); opt.step()
        if ep % 3 == 2:
            va, _, _, _ = evaluate()
            if va > best + 1e-5: best, stall, state = va, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
            else: stall += 1
            if stall >= 8: break
    net.load_state_dict(state)
    # robustness: evaluate + report the phase from several DIFFERENT pre-wall windows
    va, te_raw, te_bal, phi_full = evaluate((0, b.b))
    _, ts_raw, _, phi_short = evaluate((b.b - 72, b.b))              # last 6 years only
    _, te_early, _, phi_early = evaluate((0, b.b - 72))               # exclude the last 6 years
    pstab = np.abs(np.exp(1j * np.stack([phi_full, phi_short, phi_early])).mean(0)).mean()
    return dict(val=best, test=te_raw, test_bal=te_bal, test_short=ts_raw, test_early=te_early, phase_window_stability=pstab)


def main():
    dev = "cpu"
    try:
        import torch as T
        if T.backends.mps.is_available(): dev = "mps"
    except Exception: pass
    names, Y = rc.bal.load_all(); b = rc.BenchR(Y)
    x, L, feat, TH = prep(b)
    base = max((L[:, b.b:] == 1).mean(), (L[:, b.b:] == 0).mean())
    print(f"  no-ties median label · test {(L[:,b.b:]==1).mean()*100:.1f}% higher · base {base:.4f} · dev {dev}", flush=True)
    for sig in (0.35, 0.5):
        r = run(b, sigma=sig, device=dev)
        print(f"  [GRU sigma={sig}] test-raw {r['test']:.4f} · test-balanced {r['test_bal']:.4f} · "
              f"windows(last6y/early) {r['test_short']:.4f}/{r['test_early']:.4f} · "
              f"phase-stability {r['phase_window_stability']:.3f}  (base {base:.4f}, coin-balanced 0.5)", flush=True)


if __name__ == "__main__":
    main()
