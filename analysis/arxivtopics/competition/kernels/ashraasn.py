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

# ══ ashraasn · SHARED-BASIS RECEIVER — global rank-k basis, per-field loadings, SWA ensemble ══
# The open benchmark's strongest global finding, trained end to end: arrows live in a GLOBAL rank-k
# basis (B: 7 x k), each field owns loadings u_j (k), a level and the anchored objective; k chosen
# on the inner wall; stochastic-weight-averaged over the training tail, seed-ensembled.
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
CT = T.tensor(np.cos(TH), dtype=T.float32, device=DEV); ST = T.tensor(np.sin(TH), dtype=T.float32, device=DEV)
def fit_rank(wall, k, lam, lr, steps, seed):
    T.manual_seed(seed)
    S = T.tensor(np.sqrt(Yz[:, :wall]), dtype=T.float32, device=DEV)
    Vm = T.tensor(VALID[:, :wall].astype(np.float32), device=DEV)
    w = Vm * T.tensor((np.arange(wall) + 1.0) ** 0.75, dtype=T.float32, device=DEV)[None, :]
    w = w / w.sum(1, keepdim=True).clamp(min=1e-9)
    hz = min(wall + 30, ne)
    m = S[:, max(0, wall - 5):wall].mean(1).clamp(min=1e-4)
    B = (T.randn(7, k, 2, device=DEV) * 0.1).requires_grad_(True)     # basis, cos/sin parts
    U = (T.randn(J, k, device=DEV) * 0.1).requires_grad_(True)
    b = (S * w).sum(1).clone().requires_grad_(True)
    opt = T.optim.Adam([B, U, b], lr=lr)
    swa, ns = None, 0
    for it in range(steps):
        Ac = U @ B[:, :, 0].T; As = U @ B[:, :, 1].T                   # (J,7) effective cos/sin arrows
        R = b[:, None] + Ac @ CT.T[:, :hz] + As @ ST.T[:, :hz]
        z = T.clamp(R, min=1e-4)
        per = ((z[:, :wall] - S).abs() * w).sum(1)
        d = (z[:, wall:hz] - m[:, None]) / m[:, None]
        loss = (per + lam * (d * d).mean(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it > steps * 0.7 and it % 25 == 0:
            with T.no_grad():
                cur = [x.detach().clone() for x in (B, U, b)]
                swa = cur if swa is None else [s + c for s, c in zip(swa, cur)]; ns += 1
        if left() < 600: break
    with T.no_grad():
        Bf, Uf, bf = [s / max(ns, 1) for s in swa] if swa else (B, U, b)
        Ac = Uf @ Bf[:, :, 0].T; As = Uf @ Bf[:, :, 1].T
        R = bf[:, None] + Ac @ CT.T + As @ ST.T
        return (T.clamp(R, min=0) ** 2).cpu().numpy()
# round 2: rank x lam grid, 2-seed judging, 8-seed final, and the inner-wall run the stack needs
scores = []
for k, lam in [(k, lam) for k in (2, 3, 4, 6, 8, 12) for lam in (0.01, 0.03, 0.1)]:
    if left() < 0.5 * BUDGET_H * 3600: print("budget guard: stopping sweep"); break
    p = np.mean([fit_rank(INNER, k, lam, 0.02, 8000, sd)[:, INNER:INNER + 30] for sd in (7, 11)], 0)
    sc = perfield_r2(p, INNER, INNER + 30)
    scores.append((sc, k, lam)); print(f"rank {k} lam {lam} inner {sc:+.4f}", flush=True)
_, kbest, lbest = max(scores, key=lambda x: x[0])
print("chosen on the inner wall: rank", kbest, "lam", lbest)

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
    Pw = np.mean([fit_rank(w, kbest, lbest, 0.02, 16000, sd) for sd in (7, 11, 23)], 0)
    BY[wy] = Pw[:, w:w + (1996 - wy)]
    write_wall(BY[wy], wy)
LAM = lam_star(BY)
finals = []
for sd in (7, 11, 23, 42, 5, 31, 13, 59):
    if left() < 2400: break
    finals.append(fit_rank(nyr, kbest, lbest, 0.02, 24000, sd))
    print(f"final seed {sd} · {left()/3600:.1f}h left", flush=True)
if not finals:
    finals.append(fit_rank(nyr, kbest, lbest, 0.02, 3000, 7))
P = np.mean(finals, 0)
C30 = np.repeat(Yz[:, nyr - 1:nyr], 30, 1)
P30 = np.clip(C30 + LAM * (P[:, YI[1996]:YI[1996] + 30] - C30), 0, None)
write_submission(P30, meta={"family": "neural shared-basis receiver + carry shrink",
                            "k": kbest, "lam": lbest, "seeds": len(finals), "lam_shrink": LAM})
inners = []
for sd in (7, 11, 23):
    if left() < 600: break
    inners.append(fit_rank(INNER, kbest, lbest, 0.02, 16000, sd)[:, INNER:INNER + 30])
if inners: write_inner(np.mean(inners, 0))
