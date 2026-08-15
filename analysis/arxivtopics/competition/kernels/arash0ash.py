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

# ══ arash0ash · DEEP PHASOR — per-field complex weights, anchored Adam, seed ensemble ══
# y_j(t) = |b_j + sum_i c_ji e^{i theta_i}|^2 on the sqrt scale, the anchored objective of the open
# benchmark, trained by Adam over a config x seed sweep; configs picked on the INNER wall only.
import torch as T
def _probe():
    # Kaggle sometimes allocates a P100 (sm_60) that current torch builds cannot run kernels on:
    # cuda.is_available() says yes, the first real op throws. Probe with an actual op.
    if not T.cuda.is_available(): return "cpu"
    try:
        (T.zeros(2, device="cuda") + 1).sum().item(); return "cuda"
    except Exception as e:
        print("cuda unusable on this worker, falling back to cpu:", str(e)[:80]); return "cpu"
DEV = _probe()
print("device:", DEV)
CT = T.tensor(np.cos(TH), dtype=T.float32, device=DEV)      # (ne, 7)
ST = T.tensor(np.sin(TH), dtype=T.float32, device=DEV)
def fit_phasor(wall, lam, lr, steps, seed, anchor_k=5):
    T.manual_seed(seed)
    S = T.tensor(np.sqrt(Yz[:, :wall]), dtype=T.float32, device=DEV)
    Vm = T.tensor(VALID[:, :wall].astype(np.float32), device=DEV)
    w = Vm * T.tensor((np.arange(wall) + 1.0) ** 0.75, dtype=T.float32, device=DEV)[None, :]
    w = w / w.sum(1, keepdim=True).clamp(min=1e-9)
    hz = min(wall + 30, ne)
    tail = S[:, max(0, wall - anchor_k):wall]
    m = tail.mean(1).clamp(min=1e-4)
    b = (S * w).sum(1).clone().requires_grad_(True)
    U = (T.randn(J, 7, 2, device=DEV) * 0.01).requires_grad_(True)
    opt = T.optim.Adam([b, U], lr=lr)
    for it in range(steps):
        R = b[:, None] + U[:, :, 0] @ CT.T[:, :hz] - U[:, :, 1] @ ST.T[:, :hz]
        I = U[:, :, 0] @ ST.T[:, :hz] + U[:, :, 1] @ CT.T[:, :hz]
        z = T.sqrt(R * R + I * I + 1e-9)
        per = ((z[:, :wall] - S).abs() * w).sum(1)
        d = (z[:, wall:hz] - m[:, None]) / m[:, None]
        loss = (per + lam * (d * d).mean(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if left() < 600: break
    with T.no_grad():
        R = b[:, None] + U[:, :, 0] @ CT.T - U[:, :, 1] @ ST.T
        I = U[:, :, 0] @ ST.T + U[:, :, 1] @ CT.T
        return (R * R + I * I).cpu().numpy()
# round 2: 27-config sweep x 3 seeds, 10-seed final, and the inner-wall run the stack needs
CFGS = [dict(lam=l, lr=r, steps=s) for l in (0.003, 0.01, 0.03, 0.1) for r in (0.005, 0.01, 0.03)
        for s in (3000, 8000, 16000)]
scores = []
for ci, cfg in enumerate(CFGS):
    if left() < 0.5 * BUDGET_H * 3600: print("budget guard: stopping config sweep"); break
    preds = [fit_phasor(INNER, cfg["lam"], cfg["lr"], cfg["steps"], seed)[:, INNER:INNER + 30] for seed in (7, 11, 23)]
    sc = perfield_r2(np.mean(preds, 0), INNER, INNER + 30)
    scores.append((sc, cfg)); print(f"cfg {ci} {cfg} inner {sc:+.4f}", flush=True)
best = max(scores, key=lambda x: x[0])[1]
print("chosen on the inner wall:", best)

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
BY = {}
for wy in WALLS_Y:
    w = wy - Y0
    Pw = np.mean([fit_phasor(w, best["lam"], best["lr"], best["steps"] * 2, sd) for sd in (7, 11, 23, 42)], 0)
    BY[wy] = Pw[:, w:w + (1996 - wy)]
    write_wall(BY[wy], wy)
LAM = lam_star(BY)
# round 4: budget-filling seed ensemble — seeds until the clock
finals = []
sd = 7
while left() > 2400:
    finals.append(fit_phasor(nyr, best["lam"], best["lr"], best["steps"] * 2, sd))
    print(f"final seed {sd} done · {left()/3600:.1f}h left ({len(finals)} seeds)", flush=True)
    sd += 2
if not finals:
    finals.append(fit_phasor(nyr, best["lam"], best["lr"], 1500, 7))
P = np.mean(finals, 0)
C30 = np.repeat(Yz[:, nyr - 1:nyr], 30, 1)
P30 = np.clip(C30 + LAM * (P[:, YI[1996]:YI[1996] + 30] - C30), 0, None)
write_submission(P30, meta={"family": "deep per-field phasor + carry shrink", "cfg": best,
                            "seeds": len(finals), "lam": LAM})
inners = []
for sd in (7, 11, 23, 42):
    if left() < 600: break
    inners.append(fit_phasor(INNER, best["lam"], best["lr"], best["steps"] * 2, sd)[:, INNER:INNER + 30])
if inners: write_inner(np.mean(inners, 0))
