# ══ shared harness: data, walls, scoring, submission (inlined in every entry) ══
import os, glob, time, json
import numpy as np, pandas as pd
T0 = time.time()
BUDGET_H = float(os.environ.get("BUDGET_H", "8"))           # wall-clock training budget
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

# ══ ashranet · THE SWARM — thousands of random sky-feature ridge models, averaged ══
# Extreme-learning-machine ensemble: each member draws random frequencies/phases over the seven
# bodies and pair separations, builds random cosine features, and solves a tiny anchored ridge in
# closed form on the GPU. Thousands of members, averaged — an ensemble by construction, and every
# member is sky-only. Member count and feature width chosen on the inner wall.
import torch as T
DEV = "cuda" if T.cuda.is_available() else "cpu"
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
    return (acc / done).cpu().numpy(), done
scores = []
for width, ridge in [(64, 1e-3), (128, 1e-3), (128, 1e-2), (256, 1e-2)]:
    if left() < 0.4 * BUDGET_H * 3600: break
    p, dn = swarm(INNER, width, ridge, 150, 7)
    sc = perfield_r2(p[:, INNER:INNER + 30], INNER, INNER + 30)
    scores.append((sc, width, ridge)); print(f"width {width} ridge {ridge} ({dn} members) inner {sc:+.4f}", flush=True)
_, W_, R_ = max(scores, key=lambda x: x[0])
print("chosen on the inner wall: width", W_, "ridge", R_)
P, done = swarm(nyr, W_, R_, 4000, 42)
print("final swarm members:", done)
write_submission(P[:, YI[1996]:YI[1996] + 30],
                 meta={"family": "random sky-feature ridge swarm", "width": W_, "ridge": R_, "members": done})
