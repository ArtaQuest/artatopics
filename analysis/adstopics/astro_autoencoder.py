#!/usr/bin/env python3
"""ASTROLOGY AUTOENCODER — attention enc/dec, Gaussian-furnace link (operator 2026-07-20).

Each topic's whole training timeseries goes in; a TOPIC-INDEPENDENT attention TRANSFORMER ENCODER
(each topic a sample, self-attention over its own 210 months) emits a single PHASE p and per-body
WEIGHTS w. The FIXED astrology lookup table (planetary longitudes theta_i(t)) is then reconstructed
through the Gaussian-furnace LINK FUNCTION:

    xhat(t) = sum_i  w_i * exp( - wrap(theta_i(t) - p)^2 / (2*sigma^2) )

MSE(xhat, x) over the visible months is the loss; it backpropagates into the ENCODER ONLY (the
astrology table is fixed, sigma is frozen). The reconstruction is thresholded at its median to
classify the direction — NOTHING is set aside (every month is labelled r > per-topic median).
Honest walls: the encoder sees only pre-wall months; the fixed table extends the reconstruction
into the held-out window as the forecast.

  python3 analysis/adstopics/astro_autoencoder.py
"""
import importlib.util as u, math, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
rc = _load("analysis/adstopics/ratechange.py", "rc")
pr = _load("analysis/adstopics/phase_rotary.py", "pr")
H = 24; NP = pr.NP


def prep(b):
    Y = b.Y; n = b.n
    R = np.zeros_like(Y, float); R[:, 1:] = (Y[:, 1:] - Y[:, :-1]) / np.maximum(Y[:, :-1], 1.0)
    tau = np.median(R[:, 1:b.a], 1, keepdims=True)               # per-topic TRAIN median (leak-free)
    L = (R > tau).astype(int)                                    # label for EVERY month (no ties)
    mu = R[:, 1:b.a].mean(1, keepdims=True); sd = R[:, 1:b.a].std(1, keepdims=True) + 1e-6
    x = np.clip((R - mu) / sd, -4, 4).astype(np.float32)
    TH = pr.planet_longitudes(n).T.astype(np.float32)            # (NP, n) fixed astrology lookup table (radians)
    moy_ang = np.deg2rad(b.moy * 30 + 15).astype(np.float32)
    vis = np.ones(n, np.float32); vis[b.b:] = 0.0
    tok = np.stack([x * vis[None], np.tile(np.sin(moy_ang), (Y.shape[0], 1)),
                    np.tile(np.cos(moy_ang), (Y.shape[0], 1)), np.tile(1 - vis, (Y.shape[0], 1)),
                    np.tile((np.arange(n) / n).astype(np.float32), (Y.shape[0], 1))], -1)   # (Tn,n,5)
    return x, L, tok.astype(np.float32), TH


def run(b, seed=7, d=48, layers=2, heads=4, sigma=0.5, epochs=200, lr=2e-3, cls_w=0.3, bs=400, device="cpu"):
    import torch as T
    T.manual_seed(seed); Tn = b.Y.shape[0]; n = b.n
    x, L, tok, TH = prep(b)
    THd = T.tensor(TH, device=device)                           # (NP,n)
    xT = T.tensor(x, device=device); yL = T.tensor(L.astype(np.float32), device=device)
    tokT = T.tensor(tok, device=device)
    tr_m = np.zeros(n, np.float32); tr_m[37:b.a] = 1.0; trm = T.tensor(tr_m, device=device)

    class Enc(T.nn.Module):
        def __init__(s):
            super().__init__(); s.proj = T.nn.Linear(5, d)
            enc = T.nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.1, batch_first=True, activation="gelu")
            s.tr = T.nn.TransformerEncoder(enc, layers)
            s.toW = T.nn.Linear(d, NP); s.toPhi = T.nn.Linear(d, 2)
        def forward(s, tb):                                     # tb: (B,n,5) — each topic INDEPENDENT
            h = s.tr(s.proj(tb)).mean(1)
            w = T.nn.functional.softplus(s.toW(h))             # (B,NP) body weights >=0
            pv = s.toPhi(h); p = T.atan2(pv[:, 0], pv[:, 1])   # (B,) single phase
            return w, p

    def decode(w, p):
        # Gaussian-furnace link over the FIXED table; sigma frozen
        dphi = THd[None] - p[:, None, None]                    # (B,NP,n)
        wrap = T.atan2(T.sin(dphi), T.cos(dphi))
        gauss = T.exp(-(wrap ** 2) / (2 * sigma ** 2))         # (B,NP,n)
        return (w[:, :, None] * gauss).sum(1)                  # (B,n) reconstruction

    net = Enc().to(device); opt = T.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    idx_all = np.arange(Tn)

    def evaluate():
        net.eval()
        xhat = np.zeros((Tn, n))
        with T.no_grad():
            for i in range(0, Tn, bs):
                j = slice(i, min(i + bs, Tn))
                w, p = net(tokT[j]); xhat[j] = decode(w, p).cpu().numpy()
        # the reconstruction is a different scale than r, so CALIBRATE the classification threshold
        # on the reconstruction's own pre-wall distribution + a global offset chosen on VALIDATION.
        z = xhat - np.median(xhat[:, :b.b], 1, keepdims=True)      # centre per topic (the "median" step)
        best = (-1, 0.0)
        for q in np.linspace(-1.5, 1.5, 31):
            va = ((z[:, b.a:b.b] > q).astype(int) == L[:, b.a:b.b]).mean()
            if va > best[0]: best = (va, q)
        thr = best[1]
        va = ((z[:, b.a:b.b] > thr).astype(int) == L[:, b.a:b.b]).mean()
        te = ((z[:, b.b:] > thr).astype(int) == L[:, b.b:]).mean()
        return va, te, (z > thr).astype(int)

    best, stall, state = -1.0, 0, None
    for ep in range(epochs):
        net.train(); np.random.RandomState(ep).shuffle(idx_all)
        for i in range(0, Tn, bs):
            bi = idx_all[i:i + bs]; jt = T.tensor(bi, device=device)
            opt.zero_grad(); w, p = net(tokT[jt]); xh = decode(w, p)
            rec = (((xh - xT[jt]) ** 2) * trm[None]).sum() / (trm.sum() * len(bi))     # AUTOENCODER MSE
            med = xh[:, 37:b.a].median(1, keepdim=True).values
            cls = (T.nn.functional.binary_cross_entropy_with_logits((xh - med) * 4, yL[jt], reduction="none")
                   * trm[None]).sum() / (trm.sum() * len(bi))
            (rec + cls_w * cls).backward(); opt.step()
        if ep % 3 == 2:
            va, te, _ = evaluate()
            if va > best + 1e-5: best, stall, state = va, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
            else: stall += 1
            if stall >= 8: break
    net.load_state_dict(state)
    va, te, preds = evaluate()
    return te, va


def main():
    dev = "cpu"
    try:
        import torch as T
        if T.backends.mps.is_available(): dev = "mps"
    except Exception: pass
    names, Y = rc.bal.load_all(); b = rc.BenchR(Y)
    x, L, tok, TH = prep(b)
    base = max((L[:, b.b:] == 1).mean(), (L[:, b.b:] == 0).mean())
    print(f"  label r>median (NO ties) · test {(L[:,b.b:]==1).mean()*100:.1f}% higher · majority base {base:.4f} · dev {dev}", flush=True)
    for sig in (0.35, 0.5, 0.8):
        te, va = run(b, sigma=sig, d=48, layers=2, device=dev)
        print(f"    [AE gaussian sigma={sig}] val {va:.4f} · TEST accuracy {te:.4f}  (vs base {base:.4f})", flush=True)


if __name__ == "__main__":
    main()
