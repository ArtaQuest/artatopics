#!/usr/bin/env python3
"""WHICH MODEL SHIPS — the deterministic receiver or the gradient one (operator 2026-07-28: finalize).

The two candidates disagree about which is better, and the disagreement is entirely about WHERE you
look. At the single 1996 wall the gradient model wins (+0.8193 vs +0.7990). Over twelve origins the
deterministic model averages +0.8751. Those cannot both be the whole story, and the campaign has
already been burned once by reading a single wall (`drop_pluto` looked like a +0.02 win and was worth
+0.0000). So measure BOTH at the SAME twelve origins, in the SAME data world the page is exported
from, before anything is deployed.

  gradient v10    15 params/topic — seven arrows, seven free tunings, one level. Adam, softplus
                  amplitudes, L1 on √share, horizon anchor, twelve-sign KL term. Seed-dependent.
  deterministic   9 params/topic — seven SIGNED arrows, ONE shared tuning, one level. A 1° grid over
                  the tuning; amplitudes and level in closed form with the anchor folded in as
                  Tikhonov rows. No seed, no optimiser, no early stopping — refit it and you get the
                  same bits.

Same target, same per-topic mask, same N^¾ weights, same anchor constants, same 30-year horizon.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/final_decide.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import arxiv_fit as af

fit_final = af.fit_final          # the model of record itself — never a second copy of it


def auc_at(Y, Yw, wall, tv, n):
    tvw = tv[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    return float(np.mean([1.0 - ((Y[:, wall + h] - Yw[:, wall + h]) ** 2).sum() /
                          max(((Y[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(n - wall)]))


def main():
    names, Y, labels, future = af.load_lunar()
    Tn, n = Y.shape
    TH, R = af.sky_lunar(labels + future)
    F = np.ones_like(R); tv = af.META["topic_valid"]
    dev = "cpu"
    try:
        import torch as T
        if T.backends.mps.is_available(): dev = "mps"
    except Exception: pass
    WALLS = list(range(n - 63, n - 29, 3))
    print(f"═══ FINALIZE · {Tn} topics · origins {labels[WALLS[0]]}..{labels[WALLS[-1]]} "
          f"(each → next 30 years) · device {dev} ═══", flush=True)

    def persistence(wall):
        tvw = tv[:, :wall].astype(float)
        mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
        last = Y[:, wall - 1]
        return float(np.mean([1.0 - ((Y[:, wall + h] - last) ** 2).sum() /
                              max(((Y[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(30)]))

    rows = {}
    t0 = time.time()
    rows["deterministic (9)"] = [auc_at(Y, af.fit_final(Y, TH, w)[0], w, tv, w + 30) for w in WALLS]
    print(f"  deterministic (9)  " + " ".join(f"{a:+.3f}" for a in rows["deterministic (9)"])
          + f"   [{time.time()-t0:.0f}s]", flush=True)
    t0 = time.time()
    rows["gradient v10 (15)"] = [auc_at(Y, af.fit_market(Y, TH, F, w, dev)[0], w, tv, w + 30) for w in WALLS]
    print(f"  gradient v10 (15)  " + " ".join(f"{a:+.3f}" for a in rows["gradient v10 (15)"])
          + f"   [{time.time()-t0:.0f}s]", flush=True)
    rows["persistence (0)"] = [persistence(w) for w in WALLS]
    print(f"  persistence  (0)   " + " ".join(f"{a:+.3f}" for a in rows["persistence (0)"]), flush=True)

    det = np.array(rows["deterministic (9)"]); grd = np.array(rows["gradient v10 (15)"])
    per = np.array(rows["persistence (0)"])
    print(f"\n    {'model':22s}{'par':>4s}{'mean':>9s}{'median':>9s}{'1996':>9s}   beats persistence", flush=True)
    for lab, r, k in (("deterministic", det, 9), ("gradient v10", grd, 15), ("persistence", per, 0)):
        print(f"    {lab:22s}{k:>4d}{r.mean():>+9.4f}{np.median(r):>+9.4f}{r[-1]:>+9.4f}"
              f"{int((r>per).sum()):>13d}/{len(r)}", flush=True)
    d = det - grd; se = float(d.std(ddof=1) / np.sqrt(len(d)))
    print(f"\n  deterministic − gradient: {d.mean():+.4f} ± {se:.4f} (t={d.mean()/max(se,1e-9):.2f}), "
          f"wins {int((d>0).sum())}/{len(d)}", flush=True)
    verdict = ("DETERMINISTIC — better on average AND simpler" if d.mean() > 2*se else
               "GRADIENT — measurably better on average" if d.mean() < -2*se else
               "TIE on accuracy — decide on reproducibility and parameter count")
    print(f"  VERDICT: {verdict}", flush=True)
    json.dump({"origins": [labels[w] for w in WALLS], "auc": {k: [round(v, 4) for v in rows[k]] for k in rows},
               "delta_det_minus_grad": {"mean": round(float(d.mean()), 4), "se": round(se, 4),
                                        "wins": int((d > 0).sum())}, "verdict": verdict},
              open("analysis/arxivtopics/final_decide.json", "w"), indent=1)
    print("DECIDEDONE", flush=True)


if __name__ == "__main__":
    main()
