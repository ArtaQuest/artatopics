#!/usr/bin/env python3
"""FULL-COVERAGE ROLLING HOLD-OUT (operator 2026-07-26: "each topic would get an average R2 on each
30 year roll ... report the AUC per topic (it should also be plotted in prod)").

The single-fold run scores only the 10% that were held out. To give EVERY field an honest
unseen-field AUC, the hold-out is rotated over TEN FOLDS, so each field is held out exactly once and
is scored by a model that never saw it:

    for each of 10 folds:            (10% held out, 90% used to train the shared model)
        for each of 30 rolling walls: train shared model from scratch on the training fields ≤ wall,
                                      FREEZE it, infer each held-out field's embedding from its OWN
                                      history ≤ wall, forecast the next 30 years, accumulate SSE/SST
    → 300 trainings, and every one of the 251 fields ends up with a score from a model blind to it.

CONTROL (fold-free, so computed once over all fields): the deployed per-field receiver fitted to each
field's own history alone — the honest question being whether structure transferred from 226 OTHER
fields beats simply fitting the field by itself.

PER-FIELD METRIC, pooled across the 30 rolls at each horizon h = 1..30:
    R²_j(h) = 1 − Σ_rolls (y − ŷ)² / Σ_rolls (y − μ_train)²         AUC_j = mean_h R²_j(h)

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/rolling_cv.py [n_folds] [n_rolls]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import torch as T
import importlib.util as u

_s = u.spec_from_file_location("rh", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rolling_holdout.py"))
rh = u.module_from_spec(_s); _s.loader.exec_module(rh)

N_FOLDS = 10
HERE = os.path.dirname(os.path.abspath(__file__))


def folds(nf):
    rng = np.random.RandomState(rh.SPLIT_SEED)
    perm = rng.permutation(Tn)
    return [np.sort(perm[i::nf]) for i in range(nf)]


if __name__ == "__main__":
    nf = int(sys.argv[1]) if len(sys.argv) > 1 else N_FOLDS
    nr = int(sys.argv[2]) if len(sys.argv) > 2 else rh.N_ROLLS
    F = folds(nf); walls = rh.WALLS[:nr]
    print(f"═══ FULL-COVERAGE ROLLING HOLD-OUT · {nf} folds × {len(walls)} rolls = {nf*len(walls)} trainings ═══", flush=True)
    print(f"  every one of {Tn} fields is scored by a model that never saw it", flush=True)
    print(f"  walls {YEARS[walls[0]]}..{YEARS[walls[-1]]}, each forecasting the next {HORIZON} years", flush=True)
    sse = np.zeros((Tn, HORIZON)); sst = np.zeros((Tn, HORIZON))
    sse_s = np.zeros((Tn, HORIZON)); sst_s = np.zeros((Tn, HORIZON))
    t0 = time.time()
    for fi, held in enumerate(F):
        train = np.setdiff1d(np.arange(Tn), held)
        rh.HELD, rh.TRAIN = held, train                      # the module's roll fns read these
        for r, wall in enumerate(walls):
            tvw = TV[held, :wall].astype(float)
            mu = (Y[held, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
            hi = min(wall + HORIZON, n)
            yh = rh.roll_e2e(wall)
            yt = Y[held, wall:hi]
            sse[held, :hi - wall] += (yt - yh[:, wall:hi]) ** 2
            sst[held, :hi - wall] += (yt - mu[:, None]) ** 2
        cov = int((sst.sum(1) > 0).sum())
        el = (time.time() - t0) / 60
        auc_sofar = float(np.mean(1 - sse[held].sum(0) / np.maximum(sst[held].sum(0), 1e-9)))
        print(f"  fold {fi+1}/{nf} done ({len(held)} fields) · fold pooled AUC {auc_sofar:+.4f} · "
              f"coverage {cov}/{Tn} · {el:.0f} min elapsed", flush=True)

    print("  — control: fitting each field ALONE (fold-free) —", flush=True)
    allf = np.arange(Tn)
    rh.HELD = allf
    for r, wall in enumerate(walls):
        tvw = TV[:, :wall].astype(float)
        mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
        hi = min(wall + HORIZON, n)
        yh = rh.roll_solo(wall)
        yt = Y[:, wall:hi]
        sse_s[:, :hi - wall] += (yt - yh[:, wall:hi]) ** 2
        sst_s[:, :hi - wall] += (yt - mu[:, None]) ** 2
    print(f"  control done · {(time.time()-t0)/60:.0f} min total", flush=True)

    def summarise(SSE, SST):
        per_curve = 1 - SSE / np.maximum(SST, 1e-9)                     # (field, horizon)
        pooled = 1 - SSE.sum(0) / np.maximum(SST.sum(0), 1e-9)
        auc = per_curve.mean(1)
        return per_curve, pooled, auc

    pc, pooled, auc = summarise(sse, sst)
    pcs, pooled_s, auc_s = summarise(sse_s, sst_s)
    print(f"\n  UNSEEN-FIELD (transfer)  pooled AUC {np.mean(pooled):+.4f} · median field AUC "
          f"{np.median(auc):+.4f} · {(auc>0).mean()*100:.1f}% of fields >0", flush=True)
    print(f"  FIT-ALONE control        pooled AUC {np.mean(pooled_s):+.4f} · median field AUC "
          f"{np.median(auc_s):+.4f} · {(auc_s>0).mean()*100:.1f}% of fields >0", flush=True)
    best = np.argsort(-auc)[:8]; worst = np.argsort(auc)[:5]
    print("\n  best-generalising fields: " + ", ".join(f"{NAMES[i]} {auc[i]:+.3f}" for i in best), flush=True)
    print("  worst: " + ", ".join(f"{NAMES[i]} {auc[i]:+.3f}" for i in worst), flush=True)

    out = {"protocol": {"folds": nf, "rolls": len(walls), "hold_frac": 1.0 / nf,
                        "walls": [YEARS[w] for w in walls], "horizon": HORIZON},
           "pooled_curve": [round(float(v), 4) for v in pooled],
           "pooled_auc": round(float(np.mean(pooled)), 4),
           "median_field_auc": round(float(np.median(auc)), 4),
           "pct_positive": round(float((auc > 0).mean() * 100), 1),
           "control": {"pooled_auc": round(float(np.mean(pooled_s)), 4),
                       "median_field_auc": round(float(np.median(auc_s)), 4),
                       "pct_positive": round(float((auc_s > 0).mean() * 100), 1)},
           "per_field": {NAMES[i]: {"auc": round(float(auc[i]), 4),
                                    "curve": [round(float(v), 4) for v in pc[i]],
                                    "auc_alone": round(float(auc_s[i]), 4)} for i in range(Tn)}}
    json.dump(out, open(os.path.join(HERE, "rolling_cv.json"), "w"), indent=1)
    print(f"\n  wrote rolling_cv.json ({Tn} fields, each with a 30-point curve)", flush=True)
    print("CVDONE", flush=True)
