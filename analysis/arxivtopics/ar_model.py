#!/usr/bin/env python3
"""ARTATREND — the AUTOREGRESSIVE model (operator 2026-07-26: "takes any topic and autoregressively
predicts the next 30 years ... take the current value and rotated phases and predict the next year").

    (√y_t , the rotated sky at year t)  ──▶  shared MLP  ──▶  √y_{t+1}

One model for every field. The per-field part is only its seven tunings p_jᵢ (and seven arrow
lengths); the predictor itself is SHARED, so the same weights roll any field forward. Rolled 30 times
from the last observed value, it produces the thirty-year trend the page draws.

    features(j, t) = [ √y_t ,  cos(θᵢ(t) − p_jᵢ) · a_jᵢ ,  sin(θᵢ(t) − p_jᵢ) · a_jᵢ ]      (1 + 14)
    √y_{t+1}       = √y_t + Δ,   Δ = MLP(features)          ← residual: the step, not the level

WHY RESIDUAL: predicting the level from the level is trivially solved by the identity, which scores
well one step ahead and drifts badly over thirty. Predicting the STEP forces the sky terms to earn
their place, and the identity baseline (Δ=0) is exactly the persistence forecast we already measure.

NOTE ON INPUTS: unlike the deployed receiver, this model is *seeded with the current value* by design —
that is what "autoregressive" means and what the page needs. It still reads no future data: the roll
is fed by its OWN previous prediction, and the sky is known centuries ahead.

TRAINING: teacher-forced one-step-ahead on the training years, then SCHEDULED SAMPLING (the model is
progressively fed its own predictions) so that training matches the way it is used. Evaluated the only
way that matters here: a free 30-year roll from the wall, scored against the same carry-forward
baseline as everything else in this project.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/ar_model.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import torch as T
import torch.nn as nn

DEV = "mps" if T.backends.mps.is_available() else "cpu"
BODS = CHAMPION_BODIES
BI = [BODIES_ALL.index(b) for b in BODS]
NB = len(BODS)
HERE = os.path.dirname(os.path.abspath(__file__))


class ArtaTrend(nn.Module):
    """Per-field tunings + a SHARED one-step predictor."""

    def __init__(self, ntop, hidden=64, seed=7):
        super().__init__()
        g = np.random.RandomState(seed)
        self.U = nn.Parameter(T.tensor(np.tile([0.0, 1.0], (ntop, NB, 1)).astype(np.float32)
                                       + g.randn(ntop, NB, 2).astype(np.float32) * 0.01))
        self.Araw = nn.Parameter(T.full((ntop, NB), -2.0))
        self.net = nn.Sequential(nn.Linear(1 + 2 * NB, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        with T.no_grad():
            self.net[-1].weight.mul_(0.01); self.net[-1].bias.zero_()

    def sky(self, cth, sth):
        """Rotated sky per field-year, scaled by the field's arrows → (Tn, 2NB, ne)."""
        p = T.atan2(self.U[:, :, 0], self.U[:, :, 1])
        a = nn.functional.softplus(self.Araw)
        cp, sp = T.cos(p), T.sin(p)
        cphi = cp[:, :, None] * cth[None] + sp[:, :, None] * sth[None]
        sphi = cp[:, :, None] * sth[None] - sp[:, :, None] * cth[None]
        return T.cat([cphi * a[:, :, None], sphi * a[:, :, None]], 1)

    def step(self, y, sky_t):
        """One year: (√y_t, rotated sky at t) → √y_{t+1}. Residual in the STEP, damped."""
        z = T.cat([y[:, None], sky_t], 1)
        # tanh-bounded step: over thirty compounding years an unbounded residual is what explodes
        d = T.tanh(self.net(z).squeeze(-1)) * getattr(self, "max_step", 0.15)
        return T.clamp(y + d * (1.0 - getattr(self, "damp", 0.0)), min=0.0)

    def roll(self, y0, S, k):
        """Free autoregressive roll: k years fed by the model's own output."""
        out, y = [], y0
        for h in range(k):
            y = self.step(y, S[:, :, h])
            out.append(y)
        return T.stack(out, 1)


def fit(wall, rows=None, hidden=64, seed=7, steps=1500, lr=3e-3, k_roll=HORIZON,
        damp=0.05, device=DEV):
    """TRAIN ON THE ROLL ITSELF. One-step teacher forcing scored beautifully and drifted catastrophically
    over thirty free steps (AUC −0.43): a model tuned to predict next year is not a model tuned to
    survive being fed its own output thirty times. So training unrolls exactly like deployment —
    origins are sampled INSIDE the training era, the model is rolled k years from each on its own
    predictions, and the whole trajectory is scored. Backprop therefore sees the compounding.

    `damp` shrinks each step toward zero (a mean-reversion prior on the STEP, not the level) — the same
    lesson as the deployed model's horizon anchor: what breaks a long forecast is unchecked drift."""
    rows = np.arange(Tn) if rows is None else rows
    T.manual_seed(seed); np.random.seed(seed)
    TH = TH_ALL[:, BI]
    tb = lambda a: T.tensor(np.asarray(a, np.float32), device=device)
    cth, sth = tb(np.cos(TH).T), tb(np.sin(TH).T)
    Ysq = tb(np.sqrt(Y[rows]))
    tvm = train_mask(wall)[rows]
    wy = tb(np.clip(N[:wall], 0, None) ** 0.75)
    m = ArtaTrend(len(rows), hidden, seed).to(device)
    m.damp = damp; m.max_step = 0.15
    opt = T.optim.Adam(m.parameters(), lr=lr)
    # origins with a full k-year window inside the TRAIN era only
    origins = [o for o in range(20, wall - k_roll) if tvm[:, o - 1].mean() > 0.5]
    rs = np.random.RandomState(seed)
    # start from the initial weights so a run that never improves still returns a usable model,
    # and never let a non-finite loss (the 30-step unroll can explode) poison the parameters
    best, stall = np.inf, 0
    state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    for it in range(steps):
        S = m.sky(cth, sth)
        og = rs.choice(origins, size=min(6, len(origins)), replace=False)
        loss = 0.0
        for o in og:
            r = m.roll(Ysq[:, o - 1], S[:, :, o - 1:o - 1 + k_roll], k_roll)      # free roll, as deployed
            tgt = Ysq[:, o:o + k_roll]
            w = tb(tvm[:, o:o + k_roll].astype(np.float32)) * wy[None, o:o + k_roll]
            loss = loss + ((r - tgt).abs() * w).sum(1).div(w.sum(1).clamp(min=1e-9)).mean()
        loss = loss / len(og)
        if not T.isfinite(loss):                       # exploded roll → discard this step entirely
            opt.zero_grad(); continue
        opt.zero_grad(); loss.backward()
        if not all(T.isfinite(q.grad).all() for q in m.parameters() if q.grad is not None):
            opt.zero_grad(); continue
        T.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if it % 50 == 49:
            lv = loss.item()
            if np.isfinite(lv) and lv < best - 1e-7: best, stall, state = lv, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                stall += 1
                if stall >= 10: break
    m.load_state_dict(state)
    return m, cth, sth


def forecast(m, cth, sth, wall, rows, k=HORIZON):
    """Free roll of k years from the last observed value at `wall`."""
    with T.no_grad():
        S = m.sky(cth, sth)
        y0 = T.tensor(np.sqrt(Y[rows, wall - 1]).astype(np.float32), device=S.device)
        r = m.roll(y0, S[:, :, wall - 1:wall - 1 + k], k)
        return (r ** 2).cpu().numpy()


def score(yh, wall, rows):
    tvw = TV[rows, :wall].astype(float)
    mu = (Y[rows, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    hi = min(wall + HORIZON, n)
    yt = Y[rows, wall:hi]; yp = yh[:, :hi - wall]
    curve = [1 - ((yt[:, h] - yp[:, h]) ** 2).sum() / max(((yt[:, h] - mu) ** 2).sum(), 1e-9)
             for h in range(hi - wall)]
    sk = 1 - ((yt - yp) ** 2).sum(1) / np.maximum(((yt - mu[:, None]) ** 2).sum(1), 1e-9)
    return float(np.mean(curve)), float(np.median(sk)), float((sk > 0).mean() * 100)


if __name__ == "__main__":
    wall = WALL_OUTER
    rows = np.arange(Tn)
    print(f"═══ ArtaTrend autoregressive model · wall {YEARS[wall]} → roll {HORIZON} years ═══", flush=True)
    res = {}
    for seed in (7, 11, 23):
        m, cth, sth = fit(wall, rows, seed=seed)
        a, s, p = score(forecast(m, cth, sth, wall, rows), wall, rows)
        res[seed] = a
        print(f"  seed {seed}: 30-yr AUC {a:+.4f} · median skill {s:+.4f} · {p:.1f}%>0", flush=True)
    med = float(np.median(list(res.values())))
    npar = sum(q.numel() for q in ArtaTrend(Tn).parameters())
    print(f"\n  AUTOREGRESSIVE  {med:+.4f}  ({npar:,} params, shared predictor + per-field tunings)", flush=True)
    print(f"  deployed receiver (direct, not autoregressive)  +0.8193", flush=True)
    print(f"  carry-forward persistence                       +0.7344", flush=True)
    json.dump({"auc": med, "per_seed": res, "params": npar},
              open(os.path.join(HERE, "ar_model_result.json"), "w"), indent=1)
    T.save(fit(wall, rows, seed=7)[0].state_dict(), os.path.join(HERE, "artatrend.pt"))
    print("ARDONE", flush=True)
