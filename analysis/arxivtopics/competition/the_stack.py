#!/usr/bin/env python3
"""THE ENSEMBLE STACK v3 — late walls, horizon-shaped alpha, closed form (2026-08-15).

v2's lesson: no single pre-1996 wall can rank carry-forward against the sky — 1966-95 re-ranked
fields violently (carry -12.97 there, even on mature fields) while 1996-2025 is ossified (carry
-2.56 on the board). What DOES transfer is the shape: how fast a held level decays with horizon,
and how much steadier a 5-year mean is than a last value. So:

  pred_j(h) = a(h) * carry_j + (1 - a(h)) * sky_j(h)
  sky       = one simplex mix over {record, gain, natal, swarm, trend, carry5}
  a(h)      = closed-form optimum per horizon, pooled over SIX walls 1966..1991 (judging windows
              all end before 1996), recency-weighted 0.5^((1996-w)/10), then smoothed.

Minimising sum_j SSE_j/SS_j IS maximising the competition's mean per-field R2; with
D = carry - sky, E = sky - truth the objective is quadratic in a per horizon:
P0 + 2 a P1 + a^2 P2 with P_i field-and-wall sums, so a*(h) = clip(-P1/P2, 0, 1).
"""
import os, sys, json, itertools, time, csv, re, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import arxiv_fit as af
import global_phasor as GP

names, Yv, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
n = Yv.shape[1]; ne = TH.shape[0]
OUTER = n - 30
tv = af.META["topic_valid"]; J = len(names); starts = tv.argmax(1)
WALLS = [n - 60, n - 55, n - 50, n - 45, n - 40, n - 35]          # 1966..1991
CACHE = os.path.expanduser("~/.artaquest-dev/artacomp/stack_walls.npz")
OLD = os.path.expanduser("~/.artaquest-dev/artacomp/stack_members.npz")
OUT = os.path.expanduser("~/.artaquest-dev/artacomp/outputs/stack"); os.makedirs(OUT, exist_ok=True)
MEMB = ["record", "gain", "natal", "swarm", "trend", "carry5"]

def carry(wall): return np.repeat(Yv[:, wall - 1:wall], ne - wall, 1)
def carry5(wall):
    lo = max(0, wall - 5); seg = Yv[:, lo:wall]; msk = tv[:, lo:wall]
    L = (seg * msk).sum(1) / np.maximum(msk.sum(1), 1)
    return np.repeat(L[:, None], ne - wall, 1)
def trend(wall, phi=0.85, K=15):
    P = np.zeros((J, ne - wall))
    for j in range(J):
        idx = np.where(tv[j, max(0, wall - K):wall])[0] + max(0, wall - K)
        L = Yv[j, wall - 1]
        if len(idx) < 4: P[j] = L; continue
        m = np.polyfit(idx.astype(float), Yv[j, idx], 1)[0]
        h = np.arange(1, ne - wall + 1)
        P[j] = np.clip(L + m * phi * (1 - phi ** h) / (1 - phi), 0, None)
    return P
def natal(wall):
    Ysq = np.sqrt(Yv); hz = min(wall + 30, ne)
    tvf = tv[:, :wall].astype(float)
    wy = np.clip(af.META["evidence"][:wall], 0, None) ** 0.75
    Wm = tvf * wy[None]; Wm = Wm / np.maximum(Wm.sum(1, keepdims=True), 1e-9)
    Wa = np.zeros_like(Wm); Wa[:, wall - af.ANCHOR_K:] = (tvf * wy[None])[:, wall - af.ANCHOR_K:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tvf * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    MJ = np.maximum((Ysq[:, :wall] * Wa).sum(1), 1e-3)
    P = np.zeros((J, ne))
    for j in range(J):
        X = np.concatenate([np.ones((ne, 1)), np.cos(TH - TH[starts[j]][None, :])], 1)
        Xt, Xa = X[:wall], X[wall:hz]
        aw = af.LAM_HORIZON / (MJ[j] ** 2) / max(hz - wall, 1)
        A = Xt.T @ (Xt * Wm[j][:, None]) + aw * (Xa.T @ Xa) + 1e-8 * np.eye(8)
        b = Xt.T @ (Wm[j] * Ysq[j, :wall]) + aw * Xa.sum(0) * MJ[j]
        P[j] = np.maximum(X @ np.linalg.solve(A, b), 0) ** 2
    return P[:, wall:]
def swarm(wall, width=64, ridge=1e-3, members=400, seed=42):
    rng = np.random.RandomState(seed)
    PAIRS = [(i, k) for i in range(7) for k in range(i + 1, 7)]
    ANG = np.concatenate([TH, np.stack([TH[:, i] - TH[:, k] for i, k in PAIRS], 1)], 1)
    S = np.sqrt(Yv); hz = min(wall + 30, ne)
    tvf = tv[:, :wall].astype(float)
    wy = np.clip(af.META["evidence"][:wall], 0, None) ** 0.75
    Wm = tvf * wy[None]; Wm = Wm / np.maximum(Wm.sum(1, keepdims=True), 1e-9)
    m = np.maximum((S[:, max(0, wall - 5):wall]).mean(1), 1e-4)
    aw = 0.03 / (m ** 2) / (hz - wall)
    acc = np.zeros((J, ne))
    for _ in range(members):
        idx = rng.randint(0, ANG.shape[1], width); ph = rng.rand(width) * 2 * np.pi
        F = np.cos(ANG[:, idx] + ph[None, :])
        Ft = np.concatenate([np.ones((wall, 1)), F[:wall]], 1)
        Fa = np.concatenate([np.ones((hz - wall, 1)), F[wall:hz]], 1)
        Fl = np.concatenate([np.ones((ne, 1)), F], 1)
        G0 = np.einsum("tp,jt,tq->jpq", Ft, Wm, Ft)
        A = G0 + aw[:, None, None] * (Fa.T @ Fa)[None] + ridge * np.eye(width + 1)[None]
        r = np.einsum("tp,jt->jp", Ft, Wm * S[:, :wall]) + aw[:, None] * (Fa.sum(0)[None] * m[:, None])
        C = np.linalg.solve(A, r[..., None])[..., 0]
        acc += np.clip(C @ Fl.T, 0, None) ** 2
    return (acc / members)[:, wall:]

def members_at(wall):
    return {"record": af.fit_final(Yv, TH, wall)[0][:, wall:], "gain": GP.fit_wall(wall)[3][:, wall:],
            "natal": natal(wall), "swarm": swarm(wall), "trend": trend(wall), "carry5": carry5(wall)}

store = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
if os.path.exists(OLD):
    Z = np.load(OLD)
    for k in Z:
        if k.startswith("out_"): store.setdefault(f"w{OUTER}_{k[4:]}", Z[k])
        if k.startswith("in_"): store.setdefault(f"w{n-60}_{k[3:]}", Z[k])
changed = False
for w in WALLS + [OUTER]:
    missing = [m for m in MEMB if f"w{w}_{m}" not in store]
    if not missing: continue
    t0 = time.time(); allm = members_at(w)
    for m in MEMB: store[f"w{w}_{m}"] = allm[m]
    changed = True
    print(f"  wall {labels[w]} members fitted · {time.time()-t0:.0f}s", flush=True)
if changed: np.savez_compressed(CACHE, **store)
M = {w: {m: store[f"w{w}_{m}"] for m in MEMB} for w in WALLS + [OUTER]}
for w in WALLS + [OUTER]:                                    # trend/carry5 cheap: recompute fresh
    M[w]["trend"], M[w]["carry5"] = trend(w), carry5(w)

def r2bar(P, wall, hi):
    sc = []
    for j in range(J):
        t = Yv[j, wall:hi]; mu = t.mean(); ss = ((t - mu) ** 2).sum()
        if ss < 1e-12: continue
        sc.append(1 - ((t - P[j, :hi - wall]) ** 2).sum() / ss)
    return float(np.mean(sc))

# pooled quadratic pieces per mix, then closed-form a*(h)
DATA = {}
for w in WALLS:
    H = min(30, OUTER - w) if w != n - 60 else 30
    H = OUTER - w if OUTER - w < 30 else 30
    T = Yv[:, w:w + H]; C = carry(w)[:, :H]
    mu = T.mean(1, keepdims=True); SS = ((T - mu) ** 2).sum(1)
    inv = np.where(SS > 1e-12, 1.0 / np.maximum(SS, 1e-12), 0.0)
    rw = 0.5 ** ((int(labels[OUTER]) - int(labels[w])) / 10.0)
    DATA[w] = (T, C, inv, rw, H)
grid = [g for g in itertools.product(np.arange(0, 1.01, 0.25), repeat=len(MEMB)) if abs(sum(g) - 1) < 1e-9]
def smooth(a, conf, win=5):
    out = np.zeros_like(a)
    for h in range(len(a)):
        lo, hi = max(0, h - win // 2), min(len(a), h + win // 2 + 1)
        wgt = conf[lo:hi]
        out[h] = float((a[lo:hi] * wgt).sum() / max(wgt.sum(), 1e-12))
    return np.clip(out, 0, 1)
best = None
for g in grid:
    P0, P1, P2 = np.zeros(30), np.zeros(30), np.zeros(30)
    for w, (T, C, inv, rw, H) in DATA.items():
        sky = sum(gi * M[w][m][:, :H] for gi, m in zip(g, MEMB))
        D, E = C - sky, sky - T
        P0[:H] += rw * (inv[:, None] * E * E).sum(0)
        P1[:H] += rw * (inv[:, None] * E * D).sum(0)
        P2[:H] += rw * (inv[:, None] * D * D).sum(0)
    a = smooth(np.clip(-P1 / np.maximum(P2, 1e-12), 0, 1), np.maximum(P2, 1e-12))
    obj = float((P0 + 2 * a * P1 + a * a * P2).sum())
    if best is None or obj < best[0]: best = (obj, g, a, (P0, P1, P2))
obj, g_b, a_b, _ = best
print("mix:", {m: float(v) for m, v in zip(MEMB, g_b) if v}, flush=True)
print("alpha(h):", " ".join(f"{x:.2f}" for x in a_b), flush=True)
print("— walk-forward check (carry | stack) per wall:", flush=True)
for w in WALLS:
    T, C, inv, rw, H = DATA[w]
    sky = sum(gi * M[w][m][:, :H] for gi, m in zip(g_b, MEMB))
    P = a_b[None, :H] * C + (1 - a_b[None, :H]) * sky
    print(f"   {labels[w]}: {r2bar(C, w, w + H):+.3f} | {r2bar(P, w, w + H):+.3f}", flush=True)

CO = carry(OUTER)[:, :30]
sky_o = sum(gi * M[OUTER][m][:, :30] for gi, m in zip(g_b, MEMB))
P_out = np.clip(a_b[None, :] * CO + (1 - a_b[None, :]) * sky_o, 0, None)
def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")
years = [int(y) for y in labels]
sub = os.path.join(OUT, "submission.csv")
with open(sub, "w", newline="") as f:
    wtr = csv.writer(f); wtr.writerow(["trend", "date", "target"])
    for j, nm in enumerate(names):
        for k in range(30):
            wtr.writerow([slug(nm), years[OUTER + k], round(float(P_out[j, k]), 6)])
json.dump({"members": MEMB, "mix": {m: float(v) for m, v in zip(MEMB, g_b)},
           "alpha_by_horizon": [round(float(x), 4) for x in a_b],
           "walls": [int(labels[w]) for w in WALLS], "recency_half_life_years": 10},
          open(os.path.join(OUT, "entry_meta.json"), "w"), indent=1)
print("written:", sub, flush=True)
print("STACKDONE", flush=True)
