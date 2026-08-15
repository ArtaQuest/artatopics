#!/usr/bin/env python3
"""STACK v4 — v3.1's recent-regime selector, plus the residual booster as a member (2026-08-15).

Round 2 gave every kernel family an inner.csv, but those live at the 1966 wall — the era v3.1
rejected as unrepresentative. The one family cheap enough to refit at the recent walls locally is
the residual booster (xgboost, the round-2 config: relative targets, inverse-volatility weights,
depth 4 / eta .05 / 400 rounds). It joins the simplex at walls 1981/86/91. Everything else about
the selection is byte-identical to the_stack_v31.py, and still touches nothing after 1995.

Also prints the diagnostic the round-2 inner.csvs DO support: each kernel family's score on the
late slice of the inner window (1981-95, horizons 16-30 of the 1966 fit).

  python3 analysis/arxivtopics/competition/the_stack_v4.py
"""
import os, sys, json, itertools, csv, re, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import arxiv_fit as af

names, Yv, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
n = Yv.shape[1]; OUTER = n - 30
tv = af.META["topic_valid"]; J = len(names)
Y0 = int(labels[0])
CACHE = os.path.expanduser("~/.artaquest-dev/artacomp/stack_walls.npz")
store = dict(np.load(CACHE))

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

# ── the residual booster, exactly as the round-2 kernel fits it ──
import xgboost as xgb
STARTS = tv.argmax(1); Yz = Yv
PAIRS = [(i, k) for i in range(7) for k in range(i + 1, 7)]
SEP = np.stack([TH[:, i] - TH[:, k] for i, k in PAIRS], 1)
SKY = np.concatenate([np.sin(TH), np.cos(TH), np.sin(SEP), np.cos(SEP)], 1)
def slope_at(j, w):
    if w - 10 < STARTS[j] + 5: return 0.0
    return float(np.polyfit(np.arange(10.0), Yz[j, w - 10:w], 1)[0])
def rel_vol(j, w, K=15):
    lo = max(STARTS[j] + 1, w - K)
    if w - lo < 5: return 1.0
    seg = Yz[j, lo - 1:w]
    d = (seg[1:] - seg[:-1]) / np.maximum(seg[:-1], 1e-4)
    return float(np.var(d))
def xgb_rows(w_hi):
    X, y, wt = [], [], []
    for j in range(J):
        for w in range(max(60, STARTS[j] + 5), w_hi):
            if not tv[j, w - 1]: continue
            L = Yz[j, w - 1]; sl = slope_at(j, w); iv = 1.0 / (rel_vol(j, w) + 0.02)
            for h in range(1, 31):
                t = w + h - 1
                if t >= w_hi or not tv[j, t]: continue
                X.append(np.concatenate([SKY[t], [h, L, sl, j]]))
                y.append((Yz[j, t] - L) / max(L, 1e-4)); wt.append(iv)
    return np.array(X, np.float32), np.array(y, np.float32), np.array(wt, np.float32)
def xgbres(w):
    key = f"w{w}_xgbres"
    if key in store: return store[key][:, :30]
    X, y, wt = xgb_rows(w)
    bst = xgb.train({"max_depth": 4, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
                     "min_child_weight": 4, "tree_method": "hist",
                     "objective": "reg:squarederror", "seed": 7},
                    xgb.DMatrix(X, label=y, weight=wt), 400)
    L = Yz[:, w - 1]
    F = [np.concatenate([SKY[w + h - 1], [h, L[j], slope_at(j, w), j]])
         for j in range(J) for h in range(1, 31)]
    r = np.clip(bst.predict(xgb.DMatrix(np.array(F, np.float32))), -1, 5).reshape(J, 30)
    P = np.clip(L[:, None] * (1 + r), 0, None)
    store[key] = P
    print(f"  xgbres fitted at wall {labels[w]} ({len(y)} pairs)", flush=True)
    return P

MEMB = ["record", "gain", "natal", "swarm", "trend", "carry5", "xgbres"]
WALLS = [n - 45, n - 40, n - 35]
M = {w: {m: store[f"w{w}_{m}"] for m in ("record", "gain", "natal", "swarm")} for w in WALLS + [OUTER]}
for w in WALLS + [OUTER]:
    M[w]["trend"], M[w]["carry5"], M[w]["xgbres"] = trend(w), carry5(w), xgbres(w)
np.savez_compressed(CACHE, **store)

def sc_window(P, w, H):
    out = []
    for j in range(J):
        t = Yv[j, w:w + H]; mu = t.mean(); ss = ((t - mu) ** 2).sum()
        if ss < 1e-12: continue
        out.append(1 - ((t - P[j, :H]) ** 2).sum() / ss)
    return float(np.mean(out))

# diagnostic: round-2 kernel families on the late slice of the inner window (1981-95)
def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")
SLUG2J = {slug(nm): j for j, nm in enumerate(names)}
w81 = n - 45
print("— round-2 families, fit ≤1965, scored on 1981-95 only (their long horizons):")
for a in ("arash0ash", "artafather", "ashraasn", "ashranet"):
    path = os.path.expanduser(f"~/.artaquest-dev/artacomp/outputs/{a}-r2/inner.csv")
    P = np.zeros((J, 30))
    with open(path) as f:
        for r in csv.DictReader(f):
            P[SLUG2J[r["trend"]], int(r["date"]) - 1966] = float(r["target"])
    print(f"   {a}: {sc_window(P[:, 15:], w81, 15):+.4f}")

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
print(f"simplex points: {len(grid)}")
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
for w in WALLS:
    T, C, inv, rw, H = DATA[w]
    sky = sum(gi * M[w][m][:, :H] for gi, m in zip(g_b, MEMB) if gi)
    P = a_b[None, :H] * C + (1 - a_b[None, :H]) * sky
    print(f"  {labels[w]}: carry {sc_window(C, w, H):+.3f} | v4 {sc_window(P, w, H):+.3f}")
CO = carry(OUTER)
sky_o = sum(gi * M[OUTER][m][:, :30] for gi, m in zip(g_b, MEMB) if gi)
P_out = np.clip(a_b[None, :] * CO + (1 - a_b[None, :]) * sky_o, 0, None)
OUT = os.path.expanduser("~/.artaquest-dev/artacomp/outputs/stack4"); os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "submission.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["trend", "date", "target"])
    for j, nm in enumerate(names):
        for k in range(30): w.writerow([slug(nm), int(labels[OUTER]) + k, round(float(P_out[j, k]), 6)])
json.dump({"mix": {m: float(v) for m, v in zip(MEMB, g_b)}, "alpha": [round(float(x), 4) for x in a_b],
           "walls": [int(labels[w]) for w in WALLS], "half_life": 5},
          open(os.path.join(OUT, "entry_meta.json"), "w"), indent=1)
print("v4 submission written")
