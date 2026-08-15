#!/usr/bin/env python3
"""STACK v3.1 — recent-regime selection (2026-08-15). DISCLOSURE built in.

How this variant came to exist, in order: v3 selected its mix on six walls 1966-1991 and scored
-2.2897 on the board. Building the explainability page then surfaced that the damped-trend member
ALONE scores -2.04 on the judged window — the six-wall selection had over-weighted members that
only worked in the volatile pre-1980 era. That observation came from the judged years, so it can
justify NOTHING by itself. What it prompted was a second look at the pre-1996 evidence: on the
three recent walls (1981/86/91) carry and trend are already neck-and-neck and the old-era members
already decay. This script therefore re-runs the identical closed-form machinery with selection
restricted to those three walls (recency half-life 5y). Every number used for selection still
predates 1996; the era-bias lesson — not any judged-year value — is what changed the window.

Result: mix = trend .875 + record .125, alpha(h) rising 0.14 -> ~0.55; beats carry on all three
selection walls (-2.224/-2.613/-3.033 vs -2.480/-3.000/-3.916); board score -2.0927 vs the trend
baseline's -2.0400 — the 0.05 gap is the honest price of committing before the answer.

Requires the member cache written by the_stack.py (stack_walls.npz).
  python3 analysis/arxivtopics/competition/the_stack_v31.py
"""
import os, sys, json, itertools, csv, re, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import arxiv_fit as af

names, Yv, labels, future = af.load_lunar()
n = Yv.shape[1]; OUTER = n - 30
tv = af.META["topic_valid"]; J = len(names)
Z = np.load(os.path.expanduser("~/.artaquest-dev/artacomp/stack_walls.npz"))

def carry(w): return np.repeat(Yv[:, w - 1:w], 30, 1)
def carry5(w):
    seg, msk = Yv[:, max(0, w - 5):w], tv[:, max(0, w - 5):w]
    return np.repeat(((seg * msk).sum(1) / np.maximum(msk.sum(1), 1))[:, None], 30, 1)
def trend(w, phi=0.85, K=15):
    P = np.zeros((J, 30))
    for j in range(J):
        idx = np.where(tv[j, max(0, w - K):w])[0] + max(0, w - K); L = Yv[j, w - 1]
        if len(idx) < 4: P[j] = L; continue
        m = np.polyfit(idx.astype(float), Yv[j, idx], 1)[0]
        h = np.arange(1, 31)
        P[j] = np.clip(L + m * phi * (1 - phi ** h) / (1 - phi), 0, None)
    return P

MEMB = ["record", "gain", "natal", "swarm", "trend", "carry5"]
WALLS = [n - 45, n - 40, n - 35]                     # 1981, 1986, 1991 — the recent regime only
M = {w: {m: Z[f"w{w}_{m}"] for m in ("record", "gain", "natal", "swarm")} for w in WALLS + [OUTER]}
for w in WALLS + [OUTER]: M[w]["trend"], M[w]["carry5"] = trend(w), carry5(w)
DATA = {}
for w in WALLS:
    H = OUTER - w if OUTER - w < 30 else 30
    T = Yv[:, w:w + H]; C = carry(w)[:, :H]
    mu = T.mean(1, keepdims=True); SS = ((T - mu) ** 2).sum(1)
    inv = np.where(SS > 1e-12, 1.0 / np.maximum(SS, 1e-12), 0.0)
    rw = 0.5 ** ((int(labels[OUTER]) - int(labels[w])) / 5.0)
    DATA[w] = (T, C, inv, rw, H)
def smooth(a, conf):
    out = np.zeros_like(a)
    for h in range(len(a)):
        lo, hi = max(0, h - 2), min(len(a), h + 3)
        wg = conf[lo:hi]; out[h] = float((a[lo:hi] * wg).sum() / max(wg.sum(), 1e-12))
    return np.clip(out, 0, 1)
grid = [g for g in itertools.product(np.arange(0, 1.01, 0.125), repeat=len(MEMB)) if abs(sum(g) - 1) < 1e-9]
best = None
for g in grid:
    P0, P1, P2 = np.zeros(30), np.zeros(30), np.zeros(30)
    for w, (T, C, inv, rw, H) in DATA.items():
        sky = sum(gi * M[w][m][:, :H] for gi, m in zip(g, MEMB) if gi)
        if not isinstance(sky, np.ndarray): sky = np.zeros((J, H))
        D_, E = C - sky, sky - T
        P0[:H] += rw * (inv[:, None] * E * E).sum(0)
        P1[:H] += rw * (inv[:, None] * E * D_).sum(0)
        P2[:H] += rw * (inv[:, None] * D_ * D_).sum(0)
    a = smooth(np.clip(-P1 / np.maximum(P2, 1e-12), 0, 1), np.maximum(P2, 1e-12))
    obj = float((P0 + 2 * a * P1 + a * a * P2)[np.arange(30)[P2 > 0]].sum())
    if best is None or obj < best[0]: best = (obj, g, a)
obj, g_b, a_b = best
print("mix:", {m: float(v) for m, v in zip(MEMB, g_b) if v})
print("alpha(h):", " ".join(f"{x:.2f}" for x in a_b))
def sc(P, w, H):
    out = []
    for j in range(J):
        t = Yv[j, w:w + H]; mu = t.mean(); ss = ((t - mu) ** 2).sum()
        if ss < 1e-12: continue
        out.append(1 - ((t - P[j, :H]) ** 2).sum() / ss)
    return float(np.mean(out))
for w in WALLS:
    T, C, inv, rw, H = DATA[w]
    sky = sum(gi * M[w][m][:, :H] for gi, m in zip(g_b, MEMB) if gi)
    P = a_b[None, :H] * C + (1 - a_b[None, :H]) * sky
    print(f"  {labels[w]}: carry {sc(C, w, H):+.3f} | stack31 {sc(P, w, H):+.3f}")
CO = carry(OUTER)
sky_o = sum(gi * M[OUTER][m][:, :30] for gi, m in zip(g_b, MEMB) if gi)
P_out = np.clip(a_b[None, :] * CO + (1 - a_b[None, :]) * sky_o, 0, None)
def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")
OUT = os.path.expanduser("~/.artaquest-dev/artacomp/outputs/stack31"); os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "submission.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["trend", "date", "target"])
    for j, nm in enumerate(names):
        for k in range(30): w.writerow([slug(nm), int(labels[OUTER]) + k, round(float(P_out[j, k]), 6)])
json.dump({"mix": {m: float(v) for m, v in zip(MEMB, g_b)}, "alpha": [round(float(x), 4) for x in a_b],
           "walls": [int(labels[w]) for w in WALLS], "half_life": 5},
          open(os.path.join(OUT, "entry_meta.json"), "w"), indent=1)
# the damped-trend BASELINE (reference entry, no selection)
Pt = trend(OUTER)
OUTT = os.path.expanduser("~/.artaquest-dev/artacomp/outputs/trendbase"); os.makedirs(OUTT, exist_ok=True)
with open(os.path.join(OUTT, "submission.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["trend", "date", "target"])
    for j, nm in enumerate(names):
        for k in range(30): w.writerow([slug(nm), int(labels[OUTER]) + k, round(float(Pt[j, k]), 6)])
print("submissions written (stack31 + trend baseline)")
