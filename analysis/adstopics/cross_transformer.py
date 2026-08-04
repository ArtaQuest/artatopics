#!/usr/bin/env python3
"""CROSS-SECTIONAL TRANSFORMER (operator 2026-07-19): each topic is a TOKEN in the vocabulary
(a learned per-topic embedding, like a word vector), and a transformer attends ACROSS all 3,021
topics so each one borrows direction signal from correlated topics. Applied to the balanced
rate-change DIRECTION task. Runs locally (CPU/MPS) and on HF ZeroGPU (the train() entrypoint).

  Topic token   x_i = topic_embed[i]  +  Linear(pre-wall features_i)
  Cross-section : L transformer-encoder layers of SELF-ATTENTION over the 3,021 topic tokens
  Head          : h_i -> a 12-dim month-of-year DIRECTION profile; test month t -> profile_i[moy(t)]

Walls-first & tie-aware: features/fit < b.a for validation selection, < b.b for the test
prediction; flat months are ties, never fit or scored. No topic's label at index >= b.b is used.
"""
import importlib.util as u, json, math, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
rc = _load("analysis/adstopics/ratechange.py", "rc")
pr = _load("analysis/adstopics/phase_rotary.py", "pr")
H = 24


def build_features(b, wall):
    """Per-topic pre-wall token features (all from months < wall). Returns (feat[T,F], prof_target)."""
    Tn = b.Y.shape[0]; lw = np.where(b.tie, 0.0, b.L * 2.0 - 1.0)
    keep = (~b.tie[:, 1:wall]).astype(float); moy = b.moy[1:wall]
    clim = np.zeros((Tn, 12))
    for m in range(12):
        sel = moy == m; k = keep[:, sel]
        clim[:, m] = (lw[:, 1:wall][:, sel] * k).sum(1) / (k.sum(1) + 1e-9)   # per-month higher-fraction
    base = (lw[:, 1:wall] * keep).sum(1) / (keep.sum(1) + 1e-9)
    strength = clim.std(1)                                                    # seasonality strength
    recent = lw[:, wall - 6:wall]                                            # last 6 directions
    mag = np.clip(np.abs(b.R[:, wall - 6:wall]).mean(1), 0, 3)               # recent magnitude
    TH = pr.planet_longitudes(b.n)
    sunphi = np.arctan2((lw[:, 1:wall] * keep * np.sin(TH[1:wall, 0])[None]).sum(1),
                        (lw[:, 1:wall] * keep * np.cos(TH[1:wall, 0])[None]).sum(1))   # Sun resonance phase
    feat = np.column_stack([clim, base, strength, recent, mag, np.sin(sunphi), np.cos(sunphi)])
    return feat.astype(np.float32)


def make_model(Tn, F, d=64, layers=2, heads=4, device="cpu"):
    import torch as T
    class XSec(T.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = T.nn.Embedding(Tn, d)
            self.proj = T.nn.Linear(F, d)
            enc = T.nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.1, batch_first=True, activation="gelu")
            self.tr = T.nn.TransformerEncoder(enc, layers)
            self.head = T.nn.Sequential(T.nn.LayerNorm(d), T.nn.Linear(d, 12))
        def forward(self, idx, feat):
            x = self.emb(idx) + self.proj(feat)                # (1, T, d)
            h = self.tr(x)                                     # cross-section attention over topics
            return self.head(h).squeeze(0)                     # (T, 12) month-of-year direction profile
    return XSec().to(device)


def run(b, seed=7, d=64, layers=2, heads=4, epochs=300, lr=2e-3, device="cpu", log=print):
    import torch as T
    T.manual_seed(seed); Tn = b.Y.shape[0]
    fa = build_features(b, b.a); fb = build_features(b, b.b)                 # <a (val) and <b (test) features
    mu, sd = fa.mean(0), fa.std(0) + 1e-6
    fa = (fa - mu) / sd; fb = (fb - mu) / sd
    idx = T.arange(Tn, device=device)
    faT = T.tensor(fa, device=device); fbT = T.tensor(fb, device=device)
    moy = b.moy
    # training targets: pre-wall months' direction (tie-masked), grouped by month-of-year via the profile
    def month_batch(lo, hi):
        cols = np.arange(lo, hi); y = b.L[:, cols].astype(np.float32); m = (~b.tie[:, cols]).astype(np.float32)
        return T.tensor(y, device=device), T.tensor(m, device=device), T.tensor(moy[cols], device=device)
    ytr, mtr, mtr_moy = month_batch(37, b.a)                                # train window
    yva, mva, mva_moy = month_batch(b.a, b.b)
    net = make_model(Tn, fa.shape[1], d, layers, heads, device)
    opt = T.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    bce = T.nn.functional.binary_cross_entropy_with_logits
    best, stall, state = -1.0, 0, None
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        prof = net(idx, faT)                                                # (T,12)
        logit = prof[:, mtr_moy]                                            # (T, n_train) via month-of-year
        loss = (bce(logit, ytr, reduction="none") * mtr).sum() / mtr.sum()
        loss.backward(); opt.step()
        if ep % 5 == 4:
            net.eval()
            with T.no_grad():
                prof = net(idx, faT); lo = prof[:, mva_moy]
                acc = (((lo > 0).float() == yva).float() * mva).sum().item() / mva.sum().item()
            if acc > best + 1e-5: best, stall, state = acc, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
            else: stall += 1
            if stall >= 12: break
    net.load_state_dict(state); net.eval()
    with T.no_grad():
        prof = net(idx, fbT).cpu().numpy()                                  # TEST: <b features
    test_moy = moy[b.b:b.n]
    Z = prof[:, test_moy]                                                    # (T,24)
    return Z, best


def main():
    dev = "cpu"
    try:
        import torch as T
        if T.backends.mps.is_available(): dev = "mps"
    except Exception:
        pass
    names, Y = rc.bal.load_all(); b = rc.BenchR(Y)
    rc.ceilings(b)
    for (d, L, hd) in [(48, 2, 4), (64, 3, 4), (96, 3, 8)]:
        Z, val = run(b, d=d, layers=L, heads=hd, device=dev)
        auc = b.score(Z, f"XTF_d{d}_L{L}_h{hd}")
        print(f"    -> d{d} L{L} h{hd}: val {val:.4f} test {auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
