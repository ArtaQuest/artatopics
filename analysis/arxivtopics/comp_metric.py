#!/usr/bin/env python3
"""THE COMPETITION METRIC (operator 2026-07-26, simplified & final).

    Train on 80% of the fields. Score the 30-YEAR FORECAST on the held-out 20%.
    One split, FIVE origins (no cross-validation).

        walls  = 1966, 1974, 1981, 1989, 1995 — each forecasting the next 30 years
        split  = 20% of the 251 fields drawn at random (fixed seed), held out ENTIRELY
        metric = pooled AUC over horizons 1..30 on the held-out fields, skill measured against each
                 field's own train-window mean (the project's long-standing baseline)

WHY A HELD-OUT SUBSET: the competition is judging whether a model has learned something transferable
about how fields respond to the sky, not whether it can memorise 251 individual histories. A model
that only fits per-field parameters cannot answer at all for a field it never saw, which is why the
shared-decoder embedding architecture exists.

    shared    — decoder + sky geometry trained on the 80%, then FROZEN; a held-out field's embedding
                is inferred from its OWN history ≤ wall. Nothing after the wall is ever touched.
    fit-alone — the deployed per-field receiver fitted to each held-out field's own history alone.
                The honest control: is transferred structure worth anything over just fitting it?

PROD IS SEPARATE AND UNCHANGED: the deployed model trains on ALL fields, twice — once with the last
30 years excluded (which is the AUC shown on the page) and once on everything (which produces the
forecast to 2055). No out-of-sample-topic number is displayed in prod, because in prod there are no
out-of-sample topics.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/comp_metric.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import importlib.util as u

_s = u.spec_from_file_location("rh", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rolling_holdout.py"))
rh = u.module_from_spec(_s); _s.loader.exec_module(rh)

SPLIT_SEED = 0
HOLD_FRAC = 0.20
# FIVE WALLS, not one. A single split at a single wall could not separate two close architectures —
# the control's score sat inside the leading model's own seed spread. Five origins spread across the
# allowed range (each with a full 30-year window inside the record) average that noise down by ~sqrt(5)
# for 5x the cost, versus 30x for the full rolling version. Chosen for resolution, not for score.
WALLS = [n - 60, n - 52, n - 45, n - 37, n - 30]
WALL = WALLS[-1]                                    # the canonical wall: train ≤1995 → 1996..2025

_rng = np.random.RandomState(SPLIT_SEED)
_perm = _rng.permutation(Tn)
N_HELD = int(round(HOLD_FRAC * Tn))
HELD = np.sort(_perm[:N_HELD])
TRAIN = np.sort(_perm[N_HELD:])


def score(yh, wall=WALL, held=None):
    """Pooled AUC + per-field skill over the held-out fields' 30-year forecast at one wall."""
    held = HELD if held is None else held
    tvw = TV[held, :wall].astype(float)
    mu = (Y[held, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    hi = min(wall + HORIZON, n)
    yt = Y[held, wall:hi]; yp = yh[:, wall:hi]
    curve = [1 - ((yt[:, h] - yp[:, h]) ** 2).sum() / max(((yt[:, h] - mu) ** 2).sum(), 1e-9)
             for h in range(hi - wall)]
    skill = 1 - ((yt - yp) ** 2).sum(1) / np.maximum(((yt - mu[:, None]) ** 2).sum(1), 1e-9)
    return {"auc": round(float(np.mean(curve)), 4), "median_skill": round(float(np.median(skill)), 4),
            "pct": round(float((skill > 0).mean() * 100), 1), "curve": [round(float(v), 4) for v in curve]}


def evaluate_entry(fn, seeds=(7, 11, 23)):
    """THE COMPETITION SCORE: mean AUC over the five walls, at each seed; report the seed median.
    Reporting the per-seed spread matters — an entry whose spread exceeds its lead has not won."""
    per_seed = []
    for sd in seeds:
        aucs = [score(fn(w, seed=sd), wall=w)["auc"] for w in WALLS]
        per_seed.append(float(np.mean(aucs)))
    last = [score(fn(WALL, seed=sd), wall=WALL) for sd in seeds]
    return {"auc": round(float(np.median(per_seed)), 4),
            "spread": [round(min(per_seed), 4), round(max(per_seed), 4)],
            "per_seed": [round(v, 4) for v in per_seed],
            "median_skill": round(float(np.median([r["median_skill"] for r in last])), 4),
            "pct": round(float(np.median([r["pct"] for r in last])), 1),
            "curve": last[int(np.argsort([r["auc"] for r in last])[len(last) // 2])]["curve"]}


if __name__ == "__main__":
    rh.HELD, rh.TRAIN = HELD, TRAIN
    print(f"═══ COMPETITION METRIC · {len(HELD)} held-out fields (20%) · {len(TRAIN)} training fields ═══", flush=True)
    print(f"  walls {', '.join(str(YEARS[w]) for w in WALLS)} — each forecasting the next 30 years", flush=True)
    out = {}
    for name, fn in (("shared-embedding (transfer)", rh.roll_e2e), ("fit-alone control", rh.roll_solo)):
        r = evaluate_entry(fn)
        out[name] = r
        print(f"  {name:28s} AUC {r['auc']:+.4f} [{r['spread'][0]:+.4f}..{r['spread'][1]:+.4f}] · "
              f"per-seed {r['per_seed']} · median field skill {r['median_skill']:+.4f} · "
              f"{r['pct']:.1f}% >0", flush=True)
    win = max(out, key=lambda k: out[k]["auc"])
    other = [k for k in out if k != win][0]
    lead = out[win]["auc"] - out[other]["auc"]
    noise = max(out[win]["spread"][1] - out[win]["spread"][0], out[other]["spread"][1] - out[other]["spread"][0])
    print(f"\n  LEADER: {win} → {out[win]['auc']:+.4f} (lead {lead:+.4f}, worst seed-spread {noise:.4f})", flush=True)
    print(f"  RESOLUTION: the metric {'SEPARATES them' if lead > noise else 'CANNOT separate them'} "
          f"— lead {'>' if lead > noise else '<='} noise", flush=True)
    out["resolution"] = {"lead": round(lead, 4), "worst_spread": round(noise, 4), "separates": bool(lead > noise)}
    print(f"  (for reference, the deployed model scores +0.8193 on fields it was TRAINED on — a far "
          f"easier task than meeting a field for the first time)", flush=True)
    out["held_out_fields"] = [NAMES[i] for i in HELD]
    out["protocol"] = {"split_seed": SPLIT_SEED, "hold_frac": HOLD_FRAC, "wall_years": [YEARS[w] for w in WALLS],
                       "horizon": HORIZON, "n_held": len(HELD), "n_train": len(TRAIN)}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "comp_metric.json"), "w"), indent=1)
    print("METRICDONE", flush=True)
