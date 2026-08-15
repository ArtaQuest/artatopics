#!/usr/bin/env python3
"""STACK v5 — the full member pool: closed-form locals + every kernel family at the recent walls.

Round 3 made each kernel family fit the recent walls (1981/86/91) on Kaggle and write the raw
predictions (wall<year>.csv). Those join the simplex beside the local members; at the 1996 wall
each family's member forecast is its RAW round-2 submission (round 3's submissions are shrunk to
carry by their own lam=0 verdict, so they cannot serve as members). Selection machinery identical
to v3.1/v4; still nothing after 1995 is touched.

  python3 analysis/arxivtopics/competition/the_stack_v5.py
"""
import os, sys, json, itertools, csv, re, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import arxiv_fit as af

names, Yv, labels, future = af.load_lunar()
n = Yv.shape[1]; OUTER = n - 30
tv = af.META["topic_valid"]; J = len(names)
Y0 = int(labels[0])
store = dict(np.load(os.path.expanduser("~/.artaquest-dev/artacomp/stack_walls.npz")))

def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")
SLUG2J = {slug(nm): j for j, nm in enumerate(names)}
def read_pred(path, y_from, H):
    P = np.zeros((J, H))
    with open(path) as f:
        for r in csv.DictReader(f):
            k = int(r["date"]) - y_from
            if 0 <= k < H: P[SLUG2J[r["trend"]], k] = float(r["target"])
    return P

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

OUTP = os.path.expanduser("~/.artaquest-dev/artacomp/outputs")
KFAM = ["arash0ash", "artafather", "ashraasn", "ashranet"]
MEMB = ["record", "gain", "natal", "swarm", "trend", "carry5", "xgbres"] + [f"k_{a}" for a in KFAM]
WALLS = [n - 45, n - 40, n - 35]
M = {}
for w, wy in zip(WALLS + [OUTER], (1981, 1986, 1991, 1996)):
    H = 1996 - wy if wy < 1996 else 30
    M[w] = {m: store[f"w{w}_{m}"] for m in ("record", "gain", "natal", "swarm", "xgbres")}
    M[w]["trend"], M[w]["carry5"] = trend(w), carry5(w)
    for a in KFAM:
        src = (f"{OUTP}/{a}-r3/wall{wy}.csv" if wy < 1996 else f"{OUTP}/{a}-r2/submission.csv")
        M[w][f"k_{a}"] = read_pred(src, wy, H)
print("members loaded:", len(MEMB), flush=True)

DATA = {}
for w in WALLS:
    H = OUTER - w if OUTER - w < 30 else 30
    T = Yv[:, w:w + H]; C = carry(w)[:, :H]
    mu = T.mean(1, keepdims=True); SS = ((T - mu) ** 2).sum(1)
    inv = np.where(SS > 1e-12, 1.0 / np.maximum(SS, 1e-12), 0.0)
    rw = 0.5 ** ((int(labels[OUTER]) - int(labels[w])) / 5.0)
    DATA[w] = (T, C, inv, rw, H)
def sc_window(P, w, H):
    out = []
    for j in range(J):
        t = Yv[j, w:w + H]; mu = t.mean(); ss = ((t - mu) ** 2).sum()
        if ss < 1e-12: continue
        out.append(1 - ((t - P[j, :H]) ** 2).sum() / ss)
    return float(np.mean(out))
print("— kernel families at the recent walls (raw, their own fits):")
for a in KFAM:
    row = []
    for w in WALLS:
        H = DATA[w][4]
        row.append(f"{labels[w]} {sc_window(M[w][f'k_{a}'][:, :H], w, H):+.3f}")
    print(f"   {a}: " + " · ".join(row), flush=True)

def smooth(a, conf):
    out = np.zeros_like(a)
    for h in range(len(a)):
        lo, hi = max(0, h - 2), min(len(a), h + 3)
        wg = conf[lo:hi]; out[h] = float((a[lo:hi] * wg).sum() / max(wg.sum(), 1e-12))
    return np.clip(out, 0, 1)
# 11 members at 0.125 steps is too many simplex points; use 0.25 steps first, then refine
def search(step, anchor=None, span=0.25):
    vals = np.arange(0, 1.0001, step)
    if anchor is not None:
        opts = [[v for v in vals if abs(v - a) <= span + 1e-9] for a in anchor]
    else:
        opts = [list(vals)] * len(MEMB)
    best = None
    def rec(i, left, cur):
        nonlocal best
        if i == len(MEMB) - 1:
            if left < -1e-9 or left > opts[i][-1] + 1e-9: return
            if not any(abs(left - v) < 1e-9 for v in opts[i]): return
            g = cur + [left]
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
            return
        for v in opts[i]:
            if v > left + 1e-9: break
            rec(i + 1, left - v, cur + [v])
    rec(0, 1.0, [])
    return best
b1 = search(0.25)
print("coarse mix:", {m: v for m, v in zip(MEMB, b1[1]) if v}, flush=True)
b2 = search(0.125, anchor=b1[1])
obj, g_b, a_b = b2
print("refined mix:", {m: v for m, v in zip(MEMB, g_b) if v})
print("alpha(h):", " ".join(f"{x:.2f}" for x in a_b))
for w in WALLS:
    T, C, inv, rw, H = DATA[w]
    sky = sum(gi * M[w][m][:, :H] for gi, m in zip(g_b, MEMB) if gi)
    P = a_b[None, :H] * C + (1 - a_b[None, :H]) * sky
    print(f"  {labels[w]}: carry {sc_window(C, w, H):+.3f} | v5 {sc_window(P, w, H):+.3f}")
CO = carry(OUTER)
sky_o = sum(gi * M[OUTER][m][:, :30] for gi, m in zip(g_b, MEMB) if gi)
P_out = np.clip(a_b[None, :] * CO + (1 - a_b[None, :]) * sky_o, 0, None)
OUT = os.path.expanduser("~/.artaquest-dev/artacomp/outputs/stack5"); os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "submission.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["trend", "date", "target"])
    for j, nm in enumerate(names):
        for k in range(30): w.writerow([slug(nm), int(labels[OUTER]) + k, round(float(P_out[j, k]), 6)])
json.dump({"mix": {m: float(v) for m, v in zip(MEMB, g_b)}, "alpha": [round(float(x), 4) for x in a_b],
           "walls": [int(labels[w]) for w in WALLS]},
          open(os.path.join(OUT, "entry_meta.json"), "w"), indent=1)
print("v5 submission written")
