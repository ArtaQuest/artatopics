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

# ══ artafather · RESIDUAL BOOSTING — gradient-boosted sky residuals over carry-forward ══
# Round 1 pooled a tree over raw shares and scored -62: a pooled tree cannot represent 251
# per-field levels. Round 2 boosts the RESIDUAL over each field's carried level: for a wall w and
# horizon h, target = (share(w+h) - share(w-1)) / share(w-1) — RELATIVE, because the metric is
# per-field R2: an absolute pooled objective is content to be off by 0.01 everywhere, which is
# R2 of minus millions on a stable micro-field. Features = the sky at the target year (sin/cos of
# the seven bodies and their 21 pair separations), the horizon, the field's level and recent
# slope at the wall, and the field id. Walk-forward pairs from every historical wall; depth, eta
# and round count chosen on the inner wall; targets never cross the judging window.
import xgboost as xgb
PAIRS = [(i, k) for i in range(7) for k in range(i + 1, 7)]
SEP = np.stack([TH[:, i] - TH[:, k] for i, k in PAIRS], 1)
SKY = np.concatenate([np.sin(TH), np.cos(TH), np.sin(SEP), np.cos(SEP)], 1)   # (ne, 56)

def slope_at(j, w):
    if w - 10 < STARTS[j] + 5: return 0.0
    return float(np.polyfit(np.arange(10.0), Yz[j, w - 10:w], 1)[0])

def rel_vol(j, w, K=15):
    """Variance of the field's own relative year-to-year moves — the tree's error budget there."""
    lo = max(STARTS[j] + 1, w - K)
    if w - lo < 5: return 1.0
    seg = Yz[j, lo - 1:w]
    d = (seg[1:] - seg[:-1]) / np.maximum(seg[:-1], 1e-4)
    return float(np.var(d))

def rows(w_lo, w_hi, t_max):
    """Pairs whose TARGET year stays below t_max — the judging window is never trained on.
    Row weight is inverse relative volatility: the per-field R2 metric punishes any move
    predicted on a field that historically does not move."""
    X, y, wt = [], [], []
    for j in range(J):
        for w in range(max(w_lo, STARTS[j] + 5), w_hi):
            if not VALID[j, w - 1]: continue
            L = Yz[j, w - 1]; sl = slope_at(j, w); iv = 1.0 / (rel_vol(j, w) + 0.02)
            for h in range(1, 31):
                t = w + h - 1
                if t >= t_max or not VALID[j, t]: continue
                X.append(np.concatenate([SKY[t], [h, L, sl, j]]))
                y.append((Yz[j, t] - L) / max(L, 1e-4))
                wt.append(iv)
    return np.array(X, np.float32), np.array(y, np.float32), np.array(wt, np.float32)

DEV = "cuda"
try:
    xgb.train({"device": "cuda", "tree_method": "hist"},
              xgb.DMatrix(np.zeros((4, 2), np.float32), label=np.zeros(4, np.float32)), 2)
except Exception as e:
    DEV = "cpu"; print("xgb cpu fallback:", str(e)[:80])
print("xgb device:", DEV)

def params(depth, eta):
    return {"max_depth": depth, "eta": eta, "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 4, "device": DEV, "tree_method": "hist",
            "objective": "reg:squarederror", "seed": 7}

def preds_from(bst, wall):
    L = Yz[:, wall - 1]
    F = [np.concatenate([SKY[wall + h - 1], [h, L[j], slope_at(j, wall), j]])
         for j in range(J) for h in range(1, 31)]
    r = np.clip(bst.predict(xgb.DMatrix(np.array(F, np.float32))), -1, 5).reshape(J, 30)
    return np.clip(L[:, None] * (1 + r), 0, None)

Xi, yi, wi = rows(60, INNER, INNER)
di = xgb.DMatrix(Xi, label=yi, weight=wi)
print("inner pairs:", len(yi), flush=True)
scores = []
for depth, eta, rounds in [(4, 0.05, 400), (5, 0.05, 800), (6, 0.05, 800), (6, 0.02, 2000),
                           (8, 0.02, 2000), (8, 0.01, 4000), (10, 0.01, 4000)]:
    if left() < 0.4 * BUDGET_H * 3600: print("budget guard: stopping sweep"); break
    bst = xgb.train(params(depth, eta), di, rounds)
    sc = perfield_r2(preds_from(bst, INNER), INNER, INNER + 30)
    scores.append((sc, depth, eta, rounds)); print(f"depth {depth} eta {eta} rounds {rounds} inner {sc:+.4f}", flush=True)
_, D_, E_, R_ = max(scores, key=lambda x: x[0])
print("chosen on the inner wall: depth", D_, "eta", E_, "rounds", R_)

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
    Xw, yw, ww = rows(60, w, w)
    bw = xgb.train(params(D_, E_), xgb.DMatrix(Xw, label=yw, weight=ww), R_)
    BY[wy] = preds_from(bw, w)[:, :1996 - wy]
    write_wall(BY[wy], wy)
LAM = lam_star(BY)
Xo, yo, wo = rows(60, nyr, nyr)
print("final pairs:", len(yo), flush=True)
bst = xgb.train(params(D_, E_), xgb.DMatrix(Xo, label=yo, weight=wo), R_)
P30 = preds_from(bst, nyr)
C30 = np.repeat(Yz[:, nyr - 1:nyr], 30, 1)
P30 = np.clip(C30 + LAM * (P30 - C30), 0, None)
write_submission(P30, meta={"family": "residual boosting over carry-forward + carry shrink",
                            "depth": D_, "eta": E_, "rounds": R_, "pairs": len(yo), "lam": LAM})
bst_i = xgb.train(params(D_, E_), di, R_)
write_inner(preds_from(bst_i, INNER))
