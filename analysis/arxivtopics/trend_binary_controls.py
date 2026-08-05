#!/usr/bin/env python3
"""WHAT DID THE TRENDING CLASSIFIER ACTUALLY LEARN? Two controls that settle it (2026-08-04).

Publication counts grew secularly, so 'Δ above the topic's own median rise' is partly an ERA label —
and the slow bodies barely complete a cycle over a 200-year history, so their angles can serve as a
CALENDAR. If the accuracy survives with only the fast bodies (mars 1.9y, jupiter 11.9y — dozens of
cycles, no calendar information) it is rhythm; if it needs the slow bodies, or if a bare linear-year
feature matches it, it is a clock.

  python3 analysis/arxivtopics/trend_binary_controls.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import trend_binary as TB
import arxiv_fit as af

BOD = af.BODIES                       # mars jupiter saturn uranus neptune pluto node
SETS = {
    "all seven (the model)": list(range(7)),
    "fast only (mars, jupiter)": [BOD.index("mars"), BOD.index("jupiter")],
    "slow only (ur, ne, pl)": [BOD.index("uranus"), BOD.index("neptune"), BOD.index("pluto")],
}


def run(featset):
    if featset == "time":
        t = np.arange(TB.Z_ALL.shape[0]) / TB.Z_ALL.shape[0]
        Zf = np.stack([np.ones_like(t), t], 1)
    else:
        idx = SETS[featset]
        TH = af.sky_lunar(TB.labels_y + TB.future)[0][:, idx]
        Zf = np.concatenate([np.ones((TH.shape[0], 1)), np.sin(TH), np.cos(TH)], 1)
    old = TB.Z_ALL; TB.Z_ALL = Zf
    TB.RNG = np.random.RandomState(0)                        # identical split every arm
    te, tt = [], []
    for j in range(TB.J):
        r = TB.run_topic(j)
        if r: te.append(r["test"]); tt.append(r["test_temporal"])
    TB.Z_ALL = old
    return float(np.mean(te)), float(np.mean(tt))


if __name__ == "__main__":
    print("═══ CONTROLS: rhythm or clock? ═══", flush=True)
    out = {}
    for k in list(SETS) + ["time"]:
        s, t = run(k)
        lab = "bare linear year [1, t]" if k == "time" else k
        out[lab] = {"shuffled": round(s, 4), "temporal": round(t, 4)}
        print(f"  {lab:28s} shuffled test {s:.3f} · temporal test {t:.3f}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trend_binary_controls.json"), "w"), indent=1)
    print("CTLDONE", flush=True)
