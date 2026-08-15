# ══ shared harness: data, walls, scoring, submission (inlined in every entry) ══
import os, glob, time, json
import numpy as np, pandas as pd
T0 = time.time()
BUDGET_H = float(os.environ.get("BUDGET_H", "10.5"))           # wall-clock training budget
def left(): return BUDGET_H * 3600 - (time.time() - T0)
ROOT = os.environ.get("KDATA")
if not ROOT:
    for c in glob.glob("/kaggle/input/**/train.csv", recursive=True):
        ROOT = os.path.dirname(c); break
assert ROOT, "dataset not attached"
tr = pd.read_csv(f"{ROOT}/train.csv"); te = pd.read_csv(f"{ROOT}/test.csv")
eph = pd.read_csv(f"{ROOT}/ephemeris.csv")
FIELDS = sorted(tr["field"].unique()); J = len(FIELDS); FI = {f: j for j, f in enumerate(FIELDS)}
YRS = eph["year"].to_numpy(int); YI = {int(y): i for i, y in enumerate(YRS)}
BOD = [c[:-8] for c in eph.columns if c.endswith("_lon_deg")]
TH = np.deg2rad(eph[[f"{b}_lon_deg" for b in BOD]].to_numpy(float))   # (ne, 7)
Y0, WALL_Y = 1700, 1996
ne = len(YRS); nyr = WALL_Y - Y0                          # train years index range [0, nyr)
Y = np.full((J, nyr), np.nan)
for f, y, s in tr[["field", "year", "share"]].itertuples(index=False):
    Y[FI[f], y - Y0] = s
VALID = ~np.isnan(Y)
STARTS = VALID.argmax(1)
Yz = np.nan_to_num(Y, nan=0.0)
TEST_YEARS = list(range(1996, 2026))
INNER = 1966 - Y0                                          # inner wall: fit <1966, judge 1966..1995
def perfield_r2(pred, lo, hi):
    """The competition metric on any window we hold the truth for: per-field R2 vs the window mean."""
    sc = []
    for j in range(J):
        t = Yz[j, lo:hi]; p = pred[j]
        if VALID[j, lo:hi].sum() < 2: continue
        mu = t.mean(); ss = ((t - mu) ** 2).sum()
        if ss < 1e-12: continue
        sc.append(1 - ((t - p) ** 2).sum() / ss)
    return float(np.mean(sc))
def write_submission(pred30, path="submission.csv", meta=None):
    rows = [{"trend": f, "date": y, "target": round(float(pred30[FI[f], k]), 6)}
            for f in FIELDS for k, y in enumerate(TEST_YEARS)]
    pd.DataFrame(rows).to_csv(path, index=False)
    if meta: json.dump(meta, open("entry_meta.json", "w"), indent=1)
    print("submission written:", path, len(rows), "rows")
def write_inner(pred30, path="inner.csv"):
    """Inner-wall predictions (1966-95) from the chosen config — the stacking signal."""
    rows = [{"trend": f, "date": y, "target": round(float(pred30[FI[f], k]), 6)}
            for f in FIELDS for k, y in enumerate(range(1966, 1996))]
    pd.DataFrame(rows).to_csv(path, index=False)
    print("inner predictions written:", path, len(rows), "rows")

# ══ ashranet · THE SWARM — thousands of random sky-feature ridge models, averaged ══
# Extreme-learning-machine ensemble: each member draws random frequencies/phases over the seven
# bodies and pair separations, builds random cosine features, and solves a tiny anchored ridge in
# closed form on the GPU. Thousands of members, averaged — an ensemble by construction, and every
# member is sky-only. Member count and feature width chosen on the inner wall.
import torch as T
def _probe():
    if not T.cuda.is_available(): return "cpu"
    try:
        (T.zeros(2, device="cuda") + 1).sum().item(); return "cuda"
    except Exception as e:
        print("cuda unusable, cpu fallback:", str(e)[:80]); return "cpu"
DEV = _probe()
print("device:", DEV)
THt = T.tensor(TH, dtype=T.float32, device=DEV)
PAIRS = [(i, k) for i in range(7) for k in range(i + 1, 7)]
SEP = T.stack([THt[:, i] - THt[:, k] for i, k in PAIRS], 1)            # (ne, 21)
ANG = T.cat([THt, SEP], 1)                                             # (ne, 28) base angles
def member(wall, width, ridge, gen):
    idx = T.randint(0, ANG.shape[1], (width,), generator=gen, device=DEV)
    ph = T.rand(width, generator=gen, device=DEV) * 2 * np.pi
    F = T.cos(ANG[:, idx] + ph[None, :])                               # (ne, width) random sky features
    S = T.tensor(np.sqrt(Yz), dtype=T.float32, device=DEV)
    Vm = T.tensor(VALID.astype(np.float32), device=DEV)
    w = Vm[:, :wall] * T.tensor((np.arange(wall) + 1.0) ** 0.75, dtype=T.float32, device=DEV)[None, :]
    w = w / w.sum(1, keepdim=True).clamp(min=1e-9)
    hz = min(wall + 30, ne)
    m = S[:, max(0, wall - 5):wall].mean(1).clamp(min=1e-4)
    Ft = T.cat([T.ones(wall, 1, device=DEV), F[:wall]], 1)
    Fa = T.cat([T.ones(hz - wall, 1, device=DEV), F[wall:hz]], 1)
    Fall = T.cat([T.ones(ne, 1, device=DEV), F], 1)
    aw = (0.03 / (m ** 2) / (hz - wall))                               # (J,)
    G0 = T.einsum("tp,jt,tq->jpq", Ft, w, Ft)
    Ga = Fa.T @ Fa
    A = G0 + aw[:, None, None] * Ga[None] + ridge * T.eye(width + 1, device=DEV)[None]
    rhs = T.einsum("tp,jt->jp", Ft, w * S[:, :wall]) + aw[:, None] * (Fa.sum(0)[None, :] * m[:, None])
    C = T.linalg.solve(A, rhs.unsqueeze(2)).squeeze(2)                 # (J, width+1)
    Z = C @ Fall.T                                                     # (J, ne)
    return T.clamp(Z, min=0) ** 2
def swarm(wall, width, ridge, n, seed):
    gen = T.Generator(device=DEV); gen.manual_seed(seed)
    acc = None; done = 0
    for i in range(n):
        p = member(wall, width, ridge, gen)
        acc = p if acc is None else acc + p; done += 1
        if left() < 900: break
    if acc is None: return np.zeros((J, ne)), 0
    return (acc / done).cpu().numpy(), done

# ── round 3: recent-wall fits + shrink-toward-carry, all selection pre-1996 ──
WALLS_Y = (1981, 1986, 1991)
def write_wall(pred, wy):
    """RAW recent-wall predictions (year wy .. 1995) — members for the ensemble stack."""
    H = 1996 - wy
    rows = [{"trend": f, "date": y, "target": round(float(pred[FI[f], k]), 6)}
            for f in FIELDS for k, y in enumerate(range(wy, 1996))]
    pd.DataFrame(rows).to_csv(f"wall{wy}.csv", index=False)
    print(f"wall{wy}.csv written ({len(rows)} rows)", flush=True)
def lam_star(preds_by_wall):
    """Shrink toward carry, chosen on the pooled recent walls: P' = carry + lam (P - carry).
    lam=0 IS carry-forward — a family that cannot beat it on the recent regime ships as it."""
    best = (None, None)
    for lam in (0, .125, .25, .375, .5, .625, .75, .875, 1):
        tot = []
        for wy, P in preds_by_wall.items():
            w = wy - Y0; H = 1996 - wy
            C = np.repeat(Yz[:, w - 1:w], H, 1)
            tot.append(perfield_r2(np.clip(C + lam * (P[:, :H] - C), 0, None), w, w + H))
        m = float(np.mean(tot))
        if best[0] is None or m > best[0]: best = (m, lam)
    print(f"shrink chosen on walls {WALLS_Y}: lam={best[1]} (pooled {best[0]:+.4f})", flush=True)
    return best[1]
# round 3: wider grid, 40k final, recent-wall fits + shrink chosen there
scores = []
for width, ridge in [(32, 1e-3), (48, 1e-3), (64, 1e-3), (96, 1e-3), (128, 1e-3), (192, 1e-3),
                     (48, 3e-3), (64, 1e-2), (128, 1e-2), (256, 1e-2), (384, 3e-2)]:
    if left() < 0.5 * BUDGET_H * 3600: break
    p, dn = swarm(INNER, width, ridge, 400, 7)
    sc = perfield_r2(p[:, INNER:INNER + 30], INNER, INNER + 30)
    scores.append((sc, width, ridge)); print(f"width {width} ridge {ridge} ({dn} members) inner {sc:+.4f}", flush=True)
_, W_, R_ = max(scores, key=lambda x: x[0])
print("chosen on the inner wall: width", W_, "ridge", R_)
BY = {}
for wy in WALLS_Y:
    w = wy - Y0
    Pw, dn = swarm(w, W_, R_, 12000, 43 + wy)
    BY[wy] = Pw[:, w:w + (1996 - wy)]
    write_wall(BY[wy], wy)
LAM = lam_star(BY)
P, done = swarm(nyr, W_, R_, 40000, 42)
print("final swarm members:", done)
C30 = np.repeat(Yz[:, nyr - 1:nyr], 30, 1)
P30 = np.clip(C30 + LAM * (P[:, YI[1996]:YI[1996] + 30] - C30), 0, None)
write_submission(P30, meta={"family": "random sky-feature ridge swarm + carry shrink",
                            "width": W_, "ridge": R_, "members": done, "lam": LAM})
P_in, done_in = swarm(INNER, W_, R_, 8000, 43)
if done_in: write_inner(P_in[:, INNER:INNER + 30])
