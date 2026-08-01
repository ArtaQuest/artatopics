#!/usr/bin/env python3
"""THE NEW PROTOCOL (operator 2026-07-26): the model must GENERALISE TO UNSEEN FIELDS.

    * 10% of the fields (25 of 251) are held out ENTIRELY — their whole series, every year, is removed
      from training. The shared model never sees them at all.
    * 30 ROLLING ORIGINS. The first wall sits 60 years before the end of the data and the last sits 31
      years before it, so every roll has a full 30-year forecast window inside the record:
          roll r: train ≤ YEARS[n-60+r]  →  score the next 30 years        r = 0..29
      That is 30 separate trainings, each from scratch.
    * For every held-out field and every roll we score the 30-year forecast; the per-field curves are
      pooled across rolls and the AREA UNDER the resulting skill-vs-horizon curve is that field's AUC.

HOW A NEVER-SEEN FIELD IS PREDICTED AT ALL — the point of the embedding architecture:
    the SHARED parts (decoder, sky geometry) are trained on the 226 training fields only. For a
    held-out field the shared parts are FROZEN and only its own embedding vector is inferred, from its
    OWN HISTORY UP TO THE WALL. Nothing after the wall is ever touched, and no other field's future is
    used. This is the standard "new entity" setup: shared structure is transferred, the entity's
    coordinates are read off its past.

    BASELINE — v10-style: fit the deployed per-field receiver to that field's own history alone (it has
    no shared part to transfer). This is the honest control: does structure learned from 226 OTHER
    fields actually help a new field, versus just fitting that field by itself?

METRIC (per field): pooled across the 30 rolls at each horizon h = 1..30,
        R²_j(h) = 1 − Σ_rolls (y − ŷ)² / Σ_rolls (y − μ_train)²
        AUC_j   = mean_h R²_j(h)                     ← reported per field, and plotted in prod
μ_train is that field's mean over its own valid years before the wall — the same carry-forward baseline
the project has always scored against.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/rolling_holdout.py [n_rolls]
"""
import os, sys, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import torch as T
import torch.nn as nn
import importlib.util as u

_s = u.spec_from_file_location("e2e", os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_embed.py"))
e2e = u.module_from_spec(_s); _s.loader.exec_module(e2e)
DEV, NB, BI, BODS = e2e.DEV, e2e.NB, e2e.BI, e2e.BODS

SPLIT_SEED = 0
HOLD_FRAC = 0.10
N_ROLLS = 30
ROLL0 = 60                                   # first wall = n − 60

_rng = np.random.RandomState(SPLIT_SEED)
_perm = _rng.permutation(Tn)
N_HELD = int(round(HOLD_FRAC * Tn))
HELD = np.sort(_perm[:N_HELD])
TRAIN = np.sort(_perm[N_HELD:])
WALLS = [n - ROLL0 + r for r in range(N_ROLLS)]

CFG = dict(e2e.DEFAULT)
CFG.update(dim=64, decoder="mlp", depth=2, head="physical", emb_init="random",
           emb_norm="none", dropout=0.15, wd=0.0, lr=5e-3, steps=16000)


def _prep(wall, rows, wexp, anchor_k):
    """Loss weights + anchor for a subset of fields, using ONLY years < wall."""
    Ysq = np.sqrt(Y[rows])
    tv = train_mask(wall)[rows].astype(np.float32)
    wy = np.clip(N[:wall], 0, None) ** wexp
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    Wa = np.zeros_like(W); Wa[:, wall - anchor_k:] = (tv * wy[None])[:, wall - anchor_k:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    m = (Ysq[:, :wall] * Wa).sum(1)[:, None]
    vmean = (Ysq[:, :wall] * tv).sum(1) / np.maximum(tv.sum(1), 1.0)
    return Ysq, W, m, vmean


class Shared(nn.Module):
    """The transferable part: decoder from an embedding to (phases, amplitudes, anchor)."""

    def __init__(self, cfg, n_rows, vmean):
        super().__init__()
        d = cfg["dim"]; self.cfg = cfg
        self.emb = nn.Parameter(T.tensor(np.random.randn(n_rows, d).astype(np.float32) * 0.1))
        self.drop = nn.Dropout(cfg["dropout"]) if cfg["dropout"] > 0 else None
        out = NB * 3 + 1
        layers, i = [], d
        for _ in range(cfg["depth"]):
            layers += [nn.Linear(i, cfg["width"]), nn.SiLU()]; i = cfg["width"]
        layers += [nn.Linear(i, out)]
        self.dec = nn.Sequential(*layers)
        with T.no_grad():
            self.dec[-1].weight.mul_(0.05); self.dec[-1].bias.zero_()
            self.dec[-1].bias[NB * 2:NB * 3] = -2.0

    def decode(self, e):
        if self.drop is not None and self.training: e = self.drop(e)
        o = self.dec(e)
        pv = o[:, :NB * 2].reshape(-1, NB, 2)
        return T.atan2(pv[:, :, 0], pv[:, :, 1]), nn.functional.softplus(o[:, NB * 2:NB * 3]), o[:, NB * 3]

    def forward(self, e, cth, sth):
        p, a, b = self.decode(e)
        C = b[:, None] + (a * T.cos(p)) @ cth + (a * T.sin(p)) @ sth
        return T.clamp(C, min=1e-4) ** self.cfg["kpow"] + 1e-8


def _loop(model, opt, params, cth, sth, Yt, Wt, m_anchor, wall, hz, steps, emb=None):
    best, stall, state = np.inf, 0, None
    for it in range(steps):
        model.train()
        sig = T.sqrt(model(emb if emb is not None else model.emb, cth, sth) + 1e-8)
        per = ((sig[:, :wall] - Yt[:, :wall]).abs() * Wt).sum(1)
        d = (sig[:, wall:hz] - m_anchor) / T.clamp(m_anchor, min=1e-3)
        loss = (per + CFG["lam_h"] * (d ** 2).mean(1)).sum() / sig.shape[0]
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            lv = loss.item()
            if lv < best - 1e-7: best, stall, state = lv, 0, [x.detach().clone() for x in params]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip(params, state): x.copy_(sv)


def roll_e2e(wall, seed=7):
    """Train the shared model on the TRAIN fields, then infer embeddings for the UNSEEN ones."""
    T.manual_seed(seed); np.random.seed(seed)
    TH = TH_ALL[:, BI]; ne = TH.shape[0]; hz = min(wall + HORIZON, ne)
    tb = lambda a: T.tensor(np.asarray(a, np.float32), device=DEV)
    cth, sth = tb(np.cos(TH).T), tb(np.sin(TH).T)
    Ysq, W, m, vmean = _prep(wall, TRAIN, CFG["wexp"], CFG["anchor_k"])
    model = Shared(CFG, len(TRAIN), vmean).to(DEV)
    opt = T.optim.Adam(model.parameters(), lr=CFG["lr"])
    _loop(model, opt, list(model.parameters()), cth, sth, tb(Ysq), tb(W), tb(m), wall, hz, CFG["steps"])

    for prm in model.dec.parameters(): prm.requires_grad_(False)     # FREEZE the transferable part
    Ysq_h, W_h, m_h, vmean_h = _prep(wall, HELD, CFG["wexp"], CFG["anchor_k"])
    eh = T.tensor(np.random.randn(len(HELD), CFG["dim"]).astype(np.float32) * 0.1,
                  device=DEV, requires_grad=True)
    opt2 = T.optim.Adam([eh], lr=CFG["lr"])
    model.eval()
    _loop(model, opt2, [eh], cth, sth, tb(Ysq_h), tb(W_h), tb(m_h), wall, hz, CFG["steps"], emb=eh)
    with T.no_grad():
        return np.clip(model(eh, cth, sth).cpu().numpy(), 0, None)


def roll_solo(wall, seed=7):
    """CONTROL: the deployed per-field receiver fitted to each held-out field's own history alone."""
    T.manual_seed(seed); np.random.seed(seed)
    TH = TH_ALL[:, BI]; ne = TH.shape[0]; hz = min(wall + HORIZON, ne)
    tb = lambda a: T.tensor(np.asarray(a, np.float32), device=DEV)
    cth, sth = tb(np.cos(TH).T), tb(np.sin(TH).T)
    Ysq, W, m, vmean = _prep(wall, HELD, CFG["wexp"], CFG["anchor_k"])
    nh = len(HELD)
    inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
    A0 = np.full((nh, NB), -2.0, np.float32); A0[:, BODS.index("pluto")] = inv_sp(np.clip(vmean, 1e-3, None))
    Ar = T.tensor(A0, device=DEV, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (nh, NB, 1)).astype(np.float32) +
                 np.random.randn(nh, NB, 2).astype(np.float32) * 0.01, device=DEV, requires_grad=True)
    Bp = T.tensor(vmean.astype(np.float32), device=DEV, requires_grad=True)
    Yt, Wt, ma = tb(Ysq), tb(W), tb(m)
    opt = T.optim.Adam([Ar, U, Bp], lr=2e-2)
    def fwd():
        p = T.atan2(U[:, :, 0], U[:, :, 1]); A = nn.functional.softplus(Ar)
        C = Bp[:, None] + (A * T.cos(p)) @ cth + (A * T.sin(p)) @ sth
        return T.clamp(C, min=1e-4) ** 2 + 1e-8
    best, stall, state = np.inf, 0, None
    for it in range(9000):
        sig = T.sqrt(fwd() + 1e-8)
        per = ((sig[:, :wall] - Yt[:, :wall]).abs() * Wt).sum(1)
        d = (sig[:, wall:hz] - ma) / T.clamp(ma, min=1e-3)
        loss = (per + CFG["lam_h"] * (d ** 2).mean(1)).sum() / nh
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            lv = loss.item()
            if lv < best - 1e-7: best, stall, state = lv, 0, [x.detach().clone() for x in (Ar, U, Bp)]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip((Ar, U, Bp), state): x.copy_(sv)
        return np.clip(fwd().cpu().numpy(), 0, None)


if __name__ == "__main__":
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else N_ROLLS
    walls = WALLS[:nr]
    print(f"═══ ROLLING HOLD-OUT · {len(HELD)} unseen fields ({HOLD_FRAC:.0%}) · {len(TRAIN)} training fields ═══", flush=True)
    print(f"  {len(walls)} rolls · first wall {YEARS[walls[0]]} → scores {YEARS[walls[0]]}..{YEARS[walls[0]+29]}"
          f" · last wall {YEARS[walls[-1]]} → scores {YEARS[walls[-1]]}..{YEARS[walls[-1]+29]}", flush=True)
    acc = {k: {"sse": np.zeros((len(HELD), HORIZON)), "sst": np.zeros((len(HELD), HORIZON))}
           for k in ("e2e", "solo")}
    for r, wall in enumerate(walls):
        tvw = TV[HELD, :wall].astype(float)
        mu = (Y[HELD, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
        hi = min(wall + HORIZON, n)
        for key, fn in (("e2e", roll_e2e), ("solo", roll_solo)):
            yh = fn(wall)
            yt = Y[HELD, wall:hi]
            acc[key]["sse"][:, :hi - wall] += (yt - yh[:, wall:hi]) ** 2
            acc[key]["sst"][:, :hi - wall] += (yt - mu[:, None]) ** 2
        e_auc = float(np.mean(1 - acc["e2e"]["sse"].sum(0) / np.maximum(acc["e2e"]["sst"].sum(0), 1e-9)))
        s_auc = float(np.mean(1 - acc["solo"]["sse"].sum(0) / np.maximum(acc["solo"]["sst"].sum(0), 1e-9)))
        print(f"  roll {r + 1:2d}/{len(walls)} wall {YEARS[wall]}  running pooled AUC — "
              f"shared-embedding {e_auc:+.4f} · fit-alone {s_auc:+.4f}", flush=True)

    out = {}
    for key in ("e2e", "solo"):
        curve = 1 - acc[key]["sse"].sum(0) / np.maximum(acc[key]["sst"].sum(0), 1e-9)      # pooled
        per = 1 - acc[key]["sse"].sum(1) / np.maximum(acc[key]["sst"].sum(1), 1e-9)        # per field
        per_curve = 1 - acc[key]["sse"] / np.maximum(acc[key]["sst"], 1e-9)                # (field, h)
        out[key] = {"pooled_curve": [round(float(v), 4) for v in curve],
                    "pooled_auc": round(float(np.mean(curve)), 4),
                    "per_field_auc": {NAMES[HELD[i]]: round(float(np.mean(per_curve[i])), 4)
                                      for i in range(len(HELD))},
                    "per_field_r2": {NAMES[HELD[i]]: round(float(per[i]), 4) for i in range(len(HELD))},
                    "median_field_auc": round(float(np.median(np.mean(per_curve, 1))), 4),
                    "pct_positive": round(float((np.mean(per_curve, 1) > 0).mean() * 100), 1)}
        print(f"\n  {key:5s} POOLED AUC {out[key]['pooled_auc']:+.4f} · median field AUC "
              f"{out[key]['median_field_auc']:+.4f} · {out[key]['pct_positive']:.1f}% of unseen fields >0", flush=True)
    out["held_out_fields"] = [NAMES[i] for i in HELD]
    out["walls"] = [YEARS[w] for w in walls]
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rolling_holdout.json"), "w"), indent=1)
    print("ROLLDONE", flush=True)
