#!/usr/bin/env python3
"""STABILISING THE EMBEDDING MODEL (2026-07-26).

The competition metric reported that it CANNOT rank the two architectures: the lead was +0.0137 while
the embedding model's own training-seed spread was 0.0551 — twenty-five times the per-field control's
0.0020. No metric change fixes that (averaging over walls does not touch seed variance, and even ten
seeds leaves SE ≈ 0.010 against a 0.014 lead). The MODEL has to become reproducible before any
scoreboard can rank it. So this measures seed variance directly and attacks it.

Variance is measured where it lives: ONE wall (1996), FIVE training seeds, the spread of the resulting
held-out AUC. Cheap and diagnostic — a config that halves the spread is worth more here than one that
adds a hundredth of AUC.

CANDIDATES (all use training data only; none touches the forecast years):
  baseline      the current config (random embedding init, dropout 0.15)
  det-init      DETERMINISTIC embedding init — a fixed projection of train-window features (level,
                trend, variability), so the starting point no longer depends on the seed at all
  ema           Polyak/EMA averaging of the weights over the tail of training — the standard cure for
                run-to-run jitter, and free of new parameters
  restarts      best of 3 restarts selected by TRAINING loss (never by anything held out)
  low-dropout   dropout 0.05 — dropout is itself a source of stochasticity
  det+ema       the two most promising, combined

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/e2e_stabilize.py
"""
import os, sys, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import torch as T
import torch.nn as nn
import importlib.util as u

_s = u.spec_from_file_location("rh", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rolling_holdout.py"))
rh = u.module_from_spec(_s); _s.loader.exec_module(rh)
_m = u.spec_from_file_location("cm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "comp_metric.py"))
cm = u.module_from_spec(_m); _m.loader.exec_module(cm)

DEV, NB, BI = rh.DEV, rh.NB, rh.BI
WALL = cm.WALL
rh.HELD, rh.TRAIN = cm.HELD, cm.TRAIN


def det_features(rows, wall):
    """Deterministic per-field features from the TRAIN WINDOW ONLY (level, trend, variability)."""
    tv = train_mask(wall)[rows].astype(float)
    ys = np.sqrt(Y[rows, :wall])
    w = tv / np.maximum(tv.sum(1, keepdims=True), 1e-9)
    lvl = (ys * w).sum(1)
    t = np.arange(wall)[None, :] / wall
    tbar = (t * w).sum(1, keepdims=True)
    trend = ((t - tbar) * (ys - lvl[:, None]) * w).sum(1) / np.maximum(((t - tbar) ** 2 * w).sum(1), 1e-9)
    var = np.sqrt(np.maximum(((ys - lvl[:, None]) ** 2 * w).sum(1), 0))
    age = tv.sum(1) / wall
    F = np.stack([lvl, trend, var, age, np.log1p(lvl * 1e3)], 1)
    F = (F - F.mean(0)) / np.maximum(F.std(0), 1e-9)
    return F.astype(np.float32)


def fit_variant(wall, variant="baseline", seed=7):
    """One competition entry: train shared on TRAIN, freeze, infer HELD embeddings. Returns yhat."""
    cfg = dict(rh.CFG)
    if variant == "low-dropout": cfg["dropout"] = 0.05
    use_det = variant in ("det-init", "det+ema")
    use_ema = variant in ("ema", "det+ema")
    n_rs = 3 if variant == "restarts" else 1

    TH = TH_ALL[:, BI]; ne = TH.shape[0]; hz = min(wall + HORIZON, ne)
    tb = lambda a: T.tensor(np.asarray(a, np.float32), device=DEV)
    cth, sth = tb(np.cos(TH).T), tb(np.sin(TH).T)
    Ysq, W, m, vmean = rh._prep(wall, rh.TRAIN, cfg["wexp"], cfg["anchor_k"])
    Yt, Wt, ma = tb(Ysq), tb(W), tb(m)

    def train_shared(sd):
        T.manual_seed(sd); np.random.seed(sd)
        model = rh.Shared(cfg, len(rh.TRAIN), vmean).to(DEV)
        if use_det:
            P = np.random.RandomState(0).randn(5, cfg["dim"]).astype(np.float32) * 0.3   # FIXED projection
            with T.no_grad(): model.emb.copy_(tb(det_features(rh.TRAIN, wall) @ P))
        opt = T.optim.Adam(model.parameters(), lr=cfg["lr"])
        ema, best, stall, state = None, np.inf, 0, None
        for it in range(cfg["steps"]):
            model.train()
            sig = T.sqrt(model(model.emb, cth, sth) + 1e-8)
            per = ((sig[:, :wall] - Yt[:, :wall]).abs() * Wt).sum(1)
            d = (sig[:, wall:hz] - ma) / T.clamp(ma, min=1e-3)
            loss = (per + cfg["lam_h"] * (d ** 2).mean(1)).sum() / sig.shape[0]
            opt.zero_grad(); loss.backward(); opt.step()
            if use_ema and it > cfg["steps"] * 0.5:
                with T.no_grad():
                    sd_now = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    ema = sd_now if ema is None else {k: 0.999 * ema[k] + 0.001 * sd_now[k] for k in ema}
            if it % 200 == 199:
                lv = loss.item()
                if lv < best - 1e-7: best, stall, state = lv, 0, copy.deepcopy(model.state_dict())
                else:
                    stall += 1
                    if stall >= 10: break
        model.load_state_dict(ema if (use_ema and ema is not None) else state)
        return model, best

    cands = [train_shared(seed * 100 + k) for k in range(n_rs)]
    model = min(cands, key=lambda c: c[1])[0]                      # selected by TRAIN loss only
    for prm in model.dec.parameters(): prm.requires_grad_(False)
    model.eval()

    Ysq_h, W_h, m_h, _ = rh._prep(wall, rh.HELD, cfg["wexp"], cfg["anchor_k"])
    T.manual_seed(seed); np.random.seed(seed)
    e0 = (det_features(rh.HELD, wall) @ np.random.RandomState(0).randn(5, cfg["dim"]).astype(np.float32) * 0.3
          if use_det else np.random.randn(len(rh.HELD), cfg["dim"]).astype(np.float32) * 0.1)
    eh = T.tensor(e0, device=DEV, requires_grad=True)
    opt2 = T.optim.Adam([eh], lr=cfg["lr"])
    rh._loop(model, opt2, [eh], cth, sth, tb(Ysq_h), tb(W_h), tb(m_h), wall, hz, cfg["steps"], emb=eh)
    with T.no_grad():
        return np.clip(model(eh, cth, sth).cpu().numpy(), 0, None)


if __name__ == "__main__":
    SEEDS = (7, 11, 23, 3, 42)
    print(f"═══ SEED-VARIANCE PROBE · wall {YEARS[WALL]} · {len(SEEDS)} seeds · "
          f"{len(rh.HELD)} held-out fields ═══", flush=True)
    print(f"  target: shrink the spread below the +0.0137 lead the metric must resolve", flush=True)
    print(f"  (per-field control's own spread, for scale: 0.0020)", flush=True)
    res = {}
    for v in ("baseline", "det-init", "ema", "restarts", "low-dropout", "det+ema"):
        aucs = [cm.score(fit_variant(WALL, v, seed=s), wall=WALL)["auc"] for s in SEEDS]
        sp = max(aucs) - min(aucs)
        res[v] = {"aucs": [round(a, 4) for a in aucs], "mean": round(float(np.mean(aucs)), 4),
                  "spread": round(float(sp), 4), "sd": round(float(np.std(aucs)), 4)}
        print(f"  {v:12s} mean {res[v]['mean']:+.4f} · spread {sp:.4f} · sd {res[v]['sd']:.4f} · "
              f"{res[v]['aucs']}", flush=True)
    base = res["baseline"]["spread"]
    best = min(res, key=lambda k: res[k]["spread"])
    print(f"\n  MOST STABLE: {best} — spread {res[best]['spread']:.4f} "
          f"(baseline {base:.4f}, {base / max(res[best]['spread'], 1e-9):.1f}x reduction)", flush=True)
    print(f"  {'CAN now resolve a 0.0137 lead' if res[best]['spread'] < 0.0137 else 'STILL cannot resolve a 0.0137 lead'}", flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_stabilize.json"), "w"), indent=1)
    print("STABDONE", flush=True)
