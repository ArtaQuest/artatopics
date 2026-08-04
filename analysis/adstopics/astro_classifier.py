#!/usr/bin/env python3
"""PURE CLASSIFICATION — cross-sectional recurrent next-month direction classifier (operator 2026-07-20).

NO autoencoder, NO fixed decoder, NO reconstruction. A trainable head classifies next month's
rate-change direction DIRECTLY.

  input  (per topic, per month t): a CAUSAL feature stream — the last OBSERVED rate r_{t-1}, the
         calendar angle of t, and the deterministic SKY at t (planetary longitudes; known for the
         target month, so not leakage). r_t itself is NEVER fed before it is classified.
  model  unidirectional GRU (causal state h_t summarising months <=t)  ->  cross-sectional
         TransformerEncoder over topic codes (each topic attends to every other topic)  ->  a Linear
         head emitting ONE logit per (topic, month).
  output sigmoid(logit) = P(next month's rate change is in the "up" class).
  loss   binary cross-entropy of that logit against the balanced-at-all-times direction label.
  metric average monthly classification accuracy on the held-out test window [b.b, n).

  python3 analysis/adstopics/astro_classifier.py [lag]     (lag 1 = monthly; 12 = ANNUAL growth task)
"""
import importlib.util as u, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
rc = _load("analysis/adstopics/ratechange.py", "rc")
pr = _load("analysis/adstopics/phase_rotary.py", "pr")


def prep(b, lag=1):
    """lag=1: monthly rate change. lag=12: ANNUAL rate change r_t=(y_t-y_{t-12})/max(y_{t-12},1) —
    the long-term growth task (12-month differencing also cancels any exactly-annual cycle)."""
    Y = b.Y; n = b.n
    R = np.zeros_like(Y, float); R[:, lag:] = (Y[:, lag:] - Y[:, :-lag]) / np.maximum(Y[:, :-lag], 1.0)
    # BALANCED AT ALL TIMES: grid-searched per-topic threshold + tiny deterministic jitter splits the
    # flat-month atom so the label is ~50/50 EVERY month (coin 0.5), nothing excluded.
    jit = 1e-6 * (((np.arange(n) * 2654435761) % 1000) / 1000.0 - 0.5); Rj = R + jit[None]
    if lag > 1:
        # GROWTH task (operator): ONE GLOBAL threshold, the same for all topics, computed ONCE —
        # iterative grid search (100 steps per round, bracket refined) until the pooled train
        # label splits 50/50. The zero atom (12-month-flat cells) is identical across topics, so a
        # global tau needs a per-(topic,month) deterministic jitter to be able to split it 50/50.
        k_ = np.arange(Y.shape[0], dtype=np.int64)[:, None]; t_ = np.arange(n, dtype=np.int64)[None]
        jit2 = 1e-6 * (((k_ * 2654435761 + t_ * 40503) % 65536) / 65536.0 - 0.5)
        Rj = R + jit2
        pool = np.sort(Rj[:, lag:b.a].ravel()); N = pool.size
        lo, hi = pool[int(N * 0.10)], pool[int(N * 0.90)]; tau_g = 0.0
        for _ in range(8):                     # iterate until the 50/50 is achieved
            cand = np.linspace(lo, hi, 100)
            split = 1.0 - np.searchsorted(pool, cand, side="right") / N   # monotone decreasing
            j = int(np.clip(np.searchsorted(-split, -0.5), 1, 99))        # first split <= 0.5
            lo, hi = cand[j - 1], cand[j]                                 # keep the crossing bracketed
            tau_g = cand[j - 1] if abs(split[j - 1] - 0.5) <= abs(split[j] - 0.5) else cand[j]
            if min(abs(split[j - 1] - 0.5), abs(split[j] - 0.5)) < 1e-6: break
        L = (Rj > tau_g).astype(np.float32)
        ach = L[:, lag:b.a].mean()
        print(f"  [prep] GLOBAL threshold tau={tau_g:+.6f} (100-step grid, bracketed to 50/50) · train split {ach*100:.2f}/{(1-ach)*100:.2f}", flush=True)
    else:
        cand = np.quantile(Rj[:, lag:b.a], np.linspace(0.35, 0.65, 31), axis=1).T
        splits = np.stack([(Rj[:, lag:b.a] > cand[:, [k]]).mean(1) for k in range(cand.shape[1])], 1)
        tau = cand[np.arange(Y.shape[0]), np.abs(splits - 0.5).argmin(1)][:, None]
        L = (Rj > tau).astype(np.float32)
    # CAUSAL features at month t (to classify dir(r_t)): last observed rate r_{t-1}, never r_t.
    mu = R[:, lag:b.a].mean(1, keepdims=True); sd = R[:, lag:b.a].std(1, keepdims=True) + 1e-6
    rprev = np.zeros_like(R); rprev[:, 1:] = np.clip((R[:, :-1] - mu) / sd, -4, 4)
    TH = pr.planet_longitudes(n).T.astype(np.float32)                # sky at month t (deterministic)
    moy = np.deg2rad(b.moy * 30 + 15).astype(np.float32)
    tile = lambda v: np.tile(v, (Y.shape[0], 1))
    feat = np.stack([rprev.astype(np.float32), tile(np.sin(moy)), tile(np.cos(moy)),
                     tile(np.sin(TH[0])), tile(np.cos(TH[0]))], -1)   # (Tn,n,5)
    return L, feat.astype(np.float32)


def run(b, seed=7, d=64, epochs=200, lr=2e-3, bs=768, use_series=True, lag=1, device="cpu"):
    import torch as T
    T.manual_seed(seed); rng = np.random.RandomState(seed); Tn = b.Y.shape[0]; n = b.n
    L, feat = prep(b, lag)
    if not use_series: feat = feat.copy(); feat[:, :, 0] = 0.0        # ablate the series -> sky-only
    featT = T.tensor(feat, device=device); yL = T.tensor(L, device=device)

    class Net(T.nn.Module):
        def __init__(s):
            super().__init__()
            s.gru = T.nn.GRU(5, d, batch_first=True)                 # UNIDIRECTIONAL -> causal h_t
            xl = T.nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, activation="gelu")
            s.xsec = T.nn.TransformerEncoder(xl, 2)                   # attention across topics
            s.head = T.nn.Sequential(T.nn.Linear(2 * d, d), T.nn.GELU(), T.nn.Linear(d, 1))
        def codes(s, fb, wall):
            H, _ = s.gru(fb); return H, H[:, :wall].mean(1)          # causal states + visible-window summary
        def forward(s, fb, ctx_codes, wall):
            H, _ = s.codes(fb, wall)                                 # (B,T,d)
            ctx = s.xsec(ctx_codes[None]).squeeze(0)                 # (B,d) cross-sectional context
            hc = T.cat([H, ctx[:, None].expand(-1, H.shape[1], -1)], -1)
            return s.head(hc).squeeze(-1)                            # (B,T) logits

    net = Net().to(device); opt = T.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    bce = T.nn.BCEWithLogitsLoss()

    def all_codes(wall):
        cs = []
        for i in range(0, Tn, bs):
            _, c = net.codes(featT[i:i + bs], wall); cs.append(c)
        return T.cat(cs, 0)

    def evaluate():
        net.eval()
        with T.no_grad():
            cc = all_codes(b.b); logit = T.zeros((Tn, n), device=device)
            for i in range(0, Tn, bs):
                H, _ = net.codes(featT[i:i + bs], b.b)
                ctx = net.xsec(cc[i:i + bs][None]).squeeze(0)        # (still a valid cross-section subset at eval)
                hc = T.cat([H, ctx[:, None].expand(-1, n, -1)], -1)
                logit[i:i + bs] = net.head(hc).squeeze(-1)
            z = logit.cpu().numpy()
        # calibrate the decision threshold on validation, apply to test (report at 0.0 too)
        yv = L[:, b.a:b.b]; best = (-1, 0.0)
        for q in np.linspace(-1.5, 1.5, 41):
            acc = ((z[:, b.a:b.b] > q).astype(int) == yv).mean()
            if acc > best[0]: best = (acc, q)
        thr = best[1]; yt = L[:, b.b:]
        te_cal = ((z[:, b.b:] > thr).astype(int) == yt).mean()
        te_0 = ((z[:, b.b:] > 0).astype(int) == yt).mean()
        return best[0], te_cal, te_0

    best, stall, state = -1.0, 0, None
    for ep in range(epochs):
        net.train(); idx = rng.permutation(Tn)
        for i in range(0, Tn, bs):
            jt = T.tensor(idx[i:i + bs], device=device)
            fb = featT[jt]; H, code = net.codes(fb, b.a)             # visible = train window
            ctx = net.xsec(code[None]).squeeze(0)                    # cross-section over the minibatch
            hc = T.cat([H, ctx[:, None].expand(-1, n, -1)], -1)
            logit = net.head(hc).squeeze(-1)
            loss = bce(logit[:, lag:b.a], yL[jt, lag:b.a])           # classify TRAIN months only
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 3 == 2:
            va, _, _ = evaluate()
            if va > best + 1e-5: best, stall, state = va, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
            else: stall += 1
            if stall >= 8: break
    net.load_state_dict(state)
    va, te_cal, te_0 = evaluate()
    return dict(val=best, test_cal=te_cal, test_raw=te_0)


def main():
    dev = "cpu"
    try:
        import torch as T
        if T.backends.mps.is_available(): dev = "mps"
    except Exception: pass
    lag = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    names, Y = rc.bal.load_all(); b = rc.BenchR(Y)
    L, _ = prep(b, lag)
    up = L[:, b.b:].mean(); base = max(up, 1 - up)
    print(f"  lag={lag} ({'ANNUAL growth' if lag == 12 else 'monthly'}) balanced label · test-up {up*100:.1f}% "
          f"(majority base {base:.4f}) · coin 0.5 · dev {dev}", flush=True)
    for tag, us in (("series + sky (full)", True), ("sky only (astrology-pure)", False)):
        r = run(b, use_series=us, lag=lag, device=dev)
        print(f"  [classifier · {tag:26s}] TEST {r['test_cal']:.4f} (raw-thr {r['test_raw']:.4f}) · val {r['val']:.4f}  (base {base:.4f})", flush=True)


if __name__ == "__main__":
    main()
