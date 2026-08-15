#!/usr/bin/env python3
"""THE ENSEMBLE STACK v2 — carry-forward level + sky modulation, selected honestly (2026-08-15).

v1 trap, kept as a warning: selecting on ALL fields let the inner wall (1966-95) be dominated by
fields born mid-century, where carry-forward's level is zero and any sky model looks better; the
grid chose alpha=0 (no carry at all), which cannot transfer to the 1996 wall where every field is
mature. Selection now runs on MATURE fields only (>=15 valid years before the inner wall) — the
population the outer wall actually presents.

  pred_j = alpha_j * carry_j + (1 - alpha_j) * mix_j
  mix    = one global simplex over {record, gain, natal, swarm, trend}, picked on the inner wall
  alpha  = global grid value; per-field only where it beats the global by a clear margin
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
INNER, OUTER = n - 60, n - 30
tv = af.META["topic_valid"]; J = len(names); starts = tv.argmax(1)
CACHE = os.path.expanduser("~/.artaquest-dev/artacomp/stack_members.npz")
OUT = os.path.expanduser("~/.artaquest-dev/artacomp/outputs/stack"); os.makedirs(OUT, exist_ok=True)

def carry(wall): return np.repeat(Yv[:, wall - 1:wall], ne - wall, 1)

def trend(wall, phi=0.85, K=15):
    P = np.zeros((J, ne - wall))
    for j in range(J):
        idx = np.where(tv[j, max(0, wall - K):wall])[0] + max(0, wall - K)
        L = Yv[j, wall - 1]
        if len(idx) < 4: P[j] = L; continue
        x, y = idx.astype(float), Yv[j, idx]
        m = np.polyfit(x, y, 1)[0]
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
        nat = TH[starts[j]]
        X = np.concatenate([np.ones((ne, 1)), np.cos(TH - nat[None, :])], 1)
        Xt, Xa = X[:wall], X[wall:hz]
        aw = af.LAM_HORIZON / (MJ[j] ** 2) / max(hz - wall, 1)
        A = Xt.T @ (Xt * Wm[j][:, None]) + aw * (Xa.T @ Xa) + 1e-8 * np.eye(8)
        b = Xt.T @ (Wm[j] * np.sqrt(Yv)[j, :wall]) + aw * Xa.sum(0) * MJ[j]
        c = np.linalg.solve(A, b)
        P[j] = np.maximum(X @ c, 0) ** 2
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
        Ga = Fa.T @ Fa
        A = G0 + aw[:, None, None] * Ga[None] + ridge * np.eye(width + 1)[None]
        r = np.einsum("tp,jt->jp", Ft, Wm * S[:, :wall]) + aw[:, None] * (Fa.sum(0)[None] * m[:, None])
        C = np.linalg.solve(A, r[..., None])[..., 0]
        acc += np.clip(C @ Fl.T, 0, None) ** 2
    return (acc / members)[:, wall:]

if os.path.exists(CACHE):
    Z = np.load(CACHE)
    MI = {k[3:]: Z[k] for k in Z if k.startswith("in_")}
    MO = {k[4:]: Z[k] for k in Z if k.startswith("out_")}
    print("members loaded from cache", flush=True)
else:
    t0 = time.time()
    MI, MO = {}, {}
    for wall, M in ((INNER, MI), (OUTER, MO)):
        print(f"  fitting members at wall {labels[wall]} …", flush=True)
        M["record"] = af.fit_final(Yv, TH, wall)[0][:, wall:]
        M["gain"] = GP.fit_wall(wall)[3][:, wall:]
        M["natal"] = natal(wall); M["swarm"] = swarm(wall); M["trend"] = trend(wall)
    np.savez_compressed(CACHE, **{f"in_{k}": v for k, v in MI.items()},
                        **{f"out_{k}": v for k, v in MO.items()})
    print(f"members fitted + cached · {time.time()-t0:.0f}s", flush=True)
MI["trend"], MO["trend"] = trend(INNER), trend(OUTER)   # cheap; recompute so phi tweaks bite
CI, CO = carry(INNER), carry(OUTER)
W30 = OUTER - INNER
MATURE = tv[:, :INNER].sum(1) >= 15
print(f"mature fields for selection: {int(MATURE.sum())}/{J}", flush=True)

def pr2(P, wall, hi, mask):
    sc = []
    for j in range(J):
        if not mask[j]: continue
        t = Yv[j, wall:hi]; mu = t.mean(); ss = ((t - mu) ** 2).sum()
        if ss < 1e-12: continue
        sc.append(1 - ((t - P[j, :hi - wall]) ** 2).sum() / ss)
    return float(np.mean(sc))

ALLF = np.ones(J, bool)
print("— solo inner scores (mature | all): carry "
      f"{pr2(CI, INNER, OUTER, MATURE):+.4f} | {pr2(CI, INNER, OUTER, ALLF):+.4f}", flush=True)
for k in MI:
    print(f"   {k:<7} {pr2(MI[k][:, :W30], INNER, OUTER, MATURE):+.4f} | "
          f"{pr2(MI[k][:, :W30], INNER, OUTER, ALLF):+.4f}", flush=True)

MEMB = sorted(MI)
grid = [w for w in itertools.product(np.arange(0, 1.01, 0.25), repeat=len(MEMB)) if abs(sum(w) - 1) < 1e-9]
best = None
for w in grid:
    sky = sum(wi * MI[k][:, :W30] for wi, k in zip(w, MEMB))
    for alpha in np.arange(0, 1.001, 0.125):
        sc = pr2(alpha * CI[:, :W30] + (1 - alpha) * sky, INNER, OUTER, MATURE)
        if best is None or sc > best[0]: best = (sc, w, float(alpha))
sc_g, w_g, a_g = best
print(f"global stack (mature inner): {sc_g:+.4f}  mix {dict((k, float(v)) for k, v in zip(MEMB, w_g) if v)}  alpha {a_g}", flush=True)

sky_i = sum(wi * MI[k][:, :W30] for wi, k in zip(w_g, MEMB))
alphas = np.full(J, a_g); MARGIN = 0.05
AGRID = np.arange(0, 1.001, 0.125)
for j in range(J):
    if not MATURE[j]: continue
    t = Yv[j, INNER:OUTER]; mu = t.mean(); ss = ((t - mu) ** 2).sum()
    if ss < 1e-12: continue
    r2 = lambda p: 1 - ((t - p) ** 2).sum() / ss
    base = r2(a_g * CI[j, :W30] + (1 - a_g) * sky_i[j])
    cand = max(AGRID, key=lambda a: r2(a * CI[j, :W30] + (1 - a) * sky_i[j]))
    if r2(cand * CI[j, :W30] + (1 - cand) * sky_i[j]) > base + MARGIN: alphas[j] = cand
P_in = alphas[:, None] * CI[:, :W30] + (1 - alphas[:, None]) * sky_i
sc_f = pr2(P_in, INNER, OUTER, MATURE)
print(f"+ per-field alpha (margin {MARGIN}): mature {sc_f:+.4f} · all {pr2(P_in, INNER, OUTER, ALLF):+.4f} · "
      f"{int((alphas != a_g).sum())} fields deviate", flush=True)

sky_o = sum(wi * MO[k][:, :30] for wi, k in zip(w_g, MEMB))
P_out = alphas[:, None] * CO[:, :30] + (1 - alphas[:, None]) * sky_o
def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")
years = [int(y) for y in labels]
sub = os.path.join(OUT, "submission.csv")
with open(sub, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["trend", "date", "target"])
    for j, nm in enumerate(names):
        for k in range(30):
            w.writerow([slug(nm), years[OUTER + k], round(float(P_out[j, k]), 6)])
json.dump({"members": MEMB, "mix": {k: float(v) for k, v in zip(MEMB, w_g)}, "alpha_global": a_g,
           "fields_with_own_alpha": int((alphas != a_g).sum()), "margin": MARGIN,
           "inner_mature_global": sc_g, "inner_mature_final": sc_f,
           "mature_fields": int(MATURE.sum())},
          open(os.path.join(OUT, "entry_meta.json"), "w"), indent=1)
print("written:", sub, flush=True)
print("STACKDONE", flush=True)
