#!/usr/bin/env python3
"""TWELVE ORIGINS — buying enough resolution to actually decide (operator 2026-07-28).

Across five walls the candidates sat inside the wall-to-wall noise: L1 won 1996 by +0.0099 and lost the
early mean by +0.0031. Deciding from that is coin-flipping. Wall-to-wall spread here is ~0.10, roughly
thirty times the differences being compared, so the fix is more origins, not more cleverness.

Twelve walls at three-year spacing, every one forecasting its own next thirty years. Overlapping
windows are correlated, so this does not divide the noise by √12 — but it does stop a single unlucky
decade from choosing the model, which is exactly the failure mode that made `drop_pluto` look like a
+0.02 win when it was worth +0.0000.

Reported: mean over all twelve, plus how often each variant beats the 9-parameter reference and beats
persistence. A variant that wins on the mean but loses most walls has not won.

  python3 analysis/arxivtopics/eight_final.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import eight_param as EP
from eight_irls import fit

WALLS = list(range(n - 63, n - 29, 3))          # 12 origins, 3-year spacing


def persistence(wall):
    tvw = TV[:, :wall].astype(float)
    mu = (Y[:, :wall]*tvw).sum(1)/np.maximum(tvw.sum(1), 1.0)
    hi = min(wall+HORIZON, n)
    last = Y[:, wall-1]
    yt = Y[:, wall:hi]
    return float(np.mean([1-((yt[:,h]-last)**2).sum()/max(((yt[:,h]-mu)**2).sum(),1e-9) for h in range(hi-wall)]))


if __name__ == "__main__":
    print(f"═══ TWELVE ORIGINS · {YEARS[WALLS[0]]}..{YEARS[WALLS[-1]]} · each → next 30 years ═══", flush=True)
    ent = [("9-param L2 (reference)", 9, dict(drop=None,      l1=False, rect=False)),
           ("8-param L2",             8, dict(drop="jupiter", l1=False, rect=False)),
           ("8-param L1 (IRLS)",      8, dict(drop="jupiter", l1=True,  rect=False))]
    tab, npar = {}, {}
    for name, K, kw in ent:
        tab[name] = [fit(w, **kw) for w in WALLS]; npar[name] = K
        print(f"  {name:24s} " + " ".join(f"{a:+.3f}" for a in tab[name]), flush=True)
    pers = [persistence(w) for w in WALLS]
    tab["persistence"] = pers; npar["persistence"] = 0
    print(f"  {'persistence':24s} " + " ".join(f"{a:+.3f}" for a in pers), flush=True)

    ref = tab["9-param L2 (reference)"]
    print(f"\n    {'model':24s}{'par':>4s}{'mean':>9s}{'median':>9s}   beats 9-param   beats persistence", flush=True)
    out = {}
    for name in tab:
        r = np.array(tab[name])
        w9 = int((r > np.array(ref)).sum()); wp = int((r > np.array(pers)).sum())
        out[name] = {"params": npar[name], "mean": round(float(r.mean()), 4),
                     "median": round(float(np.median(r)), 4), "beats_ref": w9, "beats_pers": wp,
                     "per_wall": [round(float(v), 4) for v in r]}
        print(f"    {name:24s}{npar[name]:>4d}{r.mean():>+9.4f}{np.median(r):>+9.4f}"
              f"{w9:>10d}/12{wp:>14d}/12", flush=True)
    a8 = np.array(tab["8-param L1 (IRLS)"]); a9 = np.array(ref)
    d = a8 - a9
    se = float(d.std(ddof=1)/np.sqrt(len(d)))
    print(f"\n  8-param L1 − 9-param L2:  mean {d.mean():+.4f} ± {se:.4f} (SE over walls), "
          f"wins {int((d>0).sum())}/12", flush=True)
    print(f"  → {'REAL: one fewer parameter AND higher AUC' if d.mean() > 2*se else 'NOT RESOLVED: the difference is inside the noise'}", flush=True)
    json.dump({"walls": [int(YEARS[w]) for w in WALLS], "results": out,
               "delta_8L1_vs_9L2": {"mean": round(float(d.mean()), 4), "se": round(se, 4),
                                    "wins": int((d > 0).sum())}},
              open("analysis/arxivtopics/eight_final.json", "w"), indent=1)
    print("FINALDONE", flush=True)
