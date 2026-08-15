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

# ══ artafather · GRADIENT-BOOSTED SKY — pooled XGBoost on ephemeris harmonics ══
# One pooled model over (field, year) rows. Features are SKY-ONLY at prediction time: sin/cos of
# each body, first harmonics of every pair separation, plus per-field constants COMPUTED FROM THE
# TRAIN WINDOW (level, trend, age) — frozen statistics, not future-readers. Target: sqrt share
# minus the field's train-mean sqrt level (predict the deviation, add the level back).
import xgboost as xgb
PAIRS = [(i, k) for i in range(7) for k in range(i + 1, 7)]
def sky_feats(idx):
    F = [np.sin(TH[idx]), np.cos(TH[idx])]
    for i, k in PAIRS:
        D = TH[idx, i] - TH[idx, k]
        F += [np.sin(D)[:, None], np.cos(D)[:, None]]
    return np.concatenate([f if f.ndim == 2 else f[:, None] for f in F], 1)   # (n, 14+42)
def field_stats(wall):
    S = np.sqrt(Yz[:, :wall]); V = VALID[:, :wall]
    lvl = (S * V).sum(1) / np.maximum(V.sum(1), 1)
    t = np.arange(wall)[None, :].repeat(J, 0)
    tm = (t * V).sum(1) / np.maximum(V.sum(1), 1)
    tr = ((t - tm[:, None]) * (S - lvl[:, None]) * V).sum(1) / np.maximum((((t - tm[:, None]) ** 2) * V).sum(1), 1e-9)
    return lvl, tr, V.sum(1) / wall
def build(wall, years_out):
    lvl, trend, age = field_stats(wall)
    rows_X, rows_y, rows_j = [], [], []
    for j in range(J):
        ts = np.arange(STARTS[j], wall)
        Xs = sky_feats(ts)
        per = np.concatenate([Xs, np.full((len(ts), 1), lvl[j]), np.full((len(ts), 1), trend[j]),
                              np.full((len(ts), 1), age[j])], 1)
        rows_X.append(per); rows_y.append(np.sqrt(Yz[j, ts]) - lvl[j]); rows_j += [j] * len(ts)
    Xtr = np.concatenate(rows_X); ytr = np.concatenate(rows_y)
    wts = np.concatenate([(np.arange(STARTS[j], wall) + 1.0) ** 0.75 for j in range(J)])
    ti = np.array([YI[y] for y in years_out])
    Xte = []
    for j in range(J):
        Xs = sky_feats(ti)
        Xte.append(np.concatenate([Xs, np.full((len(ti), 1), lvl[j]), np.full((len(ti), 1), trend[j]),
                                   np.full((len(ti), 1), age[j])], 1))
    return Xtr, ytr, wts, np.stack(Xte), lvl
def run(wall, years_out, params, rounds, seeds):
    Xtr, ytr, wts, Xte, lvl = build(wall, years_out)
    preds = []
    for sd in seeds:
        m = xgb.XGBRegressor(tree_method="hist", device="cuda", random_state=sd,
                             n_estimators=rounds, **params)
        m.fit(Xtr, ytr, sample_weight=wts)
        Z = m.predict(Xte.reshape(-1, Xte.shape[2])).reshape(J, len(years_out))
        preds.append(np.clip(Z + lvl[:, None], 0, None) ** 2)
        if left() < 900: break
    return np.mean(preds, 0)
GRID = [dict(max_depth=d, learning_rate=lr, subsample=0.8, colsample_bytree=0.8)
        for d in (4, 6, 8) for lr in (0.03, 0.1)]
scores = []
for gi, g in enumerate(GRID):
    if left() < 0.4 * BUDGET_H * 3600: break
    p = run(INNER, list(range(1966, 1996)), g, 600, (0,))
    sc = perfield_r2(p, INNER, INNER + 30)
    scores.append((sc, g)); print(f"grid {gi} {g} inner {sc:+.4f}", flush=True)
best = max(scores, key=lambda x: x[0])[1]
print("chosen on the inner wall:", best)
P30 = run(nyr, TEST_YEARS, best, 1200, (0, 1, 2, 3, 4))
write_submission(P30, meta={"family": "pooled XGBoost on sky harmonics", "cfg": best})
