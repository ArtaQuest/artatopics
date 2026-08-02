#!/usr/bin/env python3
"""THE ANCHOR, SWITCHED OFF — the one README number that had no committed artifact behind it.

"The headline is +0.7990 with the horizon anchor and +0.6287 without it" was first measured by an
audit agent during the finalization review, which left it quotable but not regenerable from this
repository. A quoted number a stranger cannot re-run is exactly what this campaign exists to avoid,
so this script IS the measurement: the model of record at the headline wall, once as shipped and once
with LAM_HORIZON = 0, everything else identical.

  python3 analysis/arxivtopics/anchor_off.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import arxiv_fit as af

names, Y, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
n = Y.shape[1]
wall = n - 30


def auc_at(Yh):
    tv = af.META["topic_valid"]; tvw = tv[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    return float(np.mean([1.0 - ((Y[:, wall + h] - Yh[:, wall + h]) ** 2).sum() /
                          max(((Y[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(30)]))


print(f"═══ THE HORIZON ANCHOR, ON AND OFF · wall {labels[wall]} → 30 years ═══", flush=True)
a_on = auc_at(af.fit_final(Y, TH, wall)[0])
print(f"  with the anchor (LAM_HORIZON={af.LAM_HORIZON}):  AUC {a_on:+.4f}", flush=True)
af.LAM_HORIZON = 0.0
a_off = auc_at(af.fit_final(Y, TH, wall)[0])
print(f"  without it (LAM_HORIZON=0):            AUC {a_off:+.4f}", flush=True)
print(f"  the anchor is worth {a_on - a_off:+.4f} — the single largest design decision in the model", flush=True)
json.dump({"wall": labels[wall], "with_anchor": round(a_on, 4), "without_anchor": round(a_off, 4),
           "delta": round(a_on - a_off, 4)},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "anchor_off.json"), "w"), indent=1)
print("ANCHORDONE", flush=True)
