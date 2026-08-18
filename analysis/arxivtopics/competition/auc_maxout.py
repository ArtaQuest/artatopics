#!/usr/bin/env python3
"""MAX OUT THE AUC — memoryless, on the pie target, selected across MULTIPLE walls.

The obstacle is not expressiveness, it is transfer: promise x transits reached 0.6028 on one inner
wall and 0.5326 on the benchmark. So this searches the axes that govern transfer, not more features:

  window     fit on ALL history vs only the last 30/50 years before the wall
  featureset full (287) · lean (first-harmonic transits) · slow (outer planets only)
  form       concatenate promise with transits, or MULTIPLY them (the astrological claim is that a
             transit ACTIVATES a natal promise, which concatenation cannot express)
  centering  centre features within each year, since only cross-sectional order is ever scored
  lambda     shrink the sky term toward the promise, weight chosen on the walls

Selection runs on THREE inner walls (1985, 1989, 1993, each judged on its own following years, all
of it before 1997) and the winner is scored ONCE on 1997-2024. Nothing reads the field's recent
state at prediction time: the model sees which field it is and what year it is, nothing else.

  python3 analysis/arxivtopics/competition/auc_maxout.py
"""
import os, sys, json, itertools, importlib.util
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.expanduser("~/.artaquest-dev/artatopics/analysis/arxivtopics"))

BUN = os.path.expanduser("~/.artaquest-dev/artacomp/piecomp")
tr = pd.read_csv(f"{BUN}/train.csv"); te = pd.read_csv(f"{BUN}/test.csv"); sol = pd.read_csv(f"{BUN}/solution.csv")
yte = sol.set_index("id").loc[te["id"]]["target"].to_numpy()
WALL_Y = 1997
ALL = pd.concat([tr[["field", "year", "target"]],
                 te[["field", "year"]].assign(target=yte)], ignore_index=True)
spec = importlib.util.spec_from_file_location("amax", os.path.expanduser(
    "~/.artaquest-dev/artatopics/analysis/arxivtopics/competition/auc_astrology_max.py"))
src = open(spec.origin).read().split('print("building features')[0]
ns = {"__file__": spec.origin}; exec(compile(src, spec.origin, "exec"), ns)
rf = ns["row_feats"]; ns["birth"].clear()
bmap = tr.groupby("field")["year"].min().to_dict()
for f in te["field"].unique(): bmap.setdefault(f, WALL_Y)
ns["birth"].update(bmap)
print("building the feature matrix once for every row …", flush=True)
X = np.asarray([rf(f, int(t)) for f, t in zip(ALL["field"], ALL["year"])], float)
Y = ALL["target"].to_numpy(); YR = ALL["year"].to_numpy(); FLD = ALL["field"].to_numpy()
print(f"  {X.shape[0]} rows x {X.shape[1]} features", flush=True)
NB = 7
LEAN = np.r_[0:2*NB]                                   # first-harmonic transits only
SLOW = np.r_[[2, 3, 4, 5, 6, 9, 10, 11, 12, 13]]       # outer-planet cos/sin slots
SETS = {"full": np.arange(X.shape[1]), "lean": LEAN, "slow": SLOW}

def promise_of(fit_mask, K=25):
    d = pd.DataFrame({"f": FLD[fit_mask], "y": Y[fit_mask]}).groupby("f")["y"].agg(["mean", "count"])
    pri = Y[fit_mask].mean()
    return {f: (r["mean"]*r["count"] + pri*K)/(r["count"] + K) for f, r in d.iterrows()}, pri

def centre(A, yrs):
    B = A.copy()
    for y in np.unique(yrs):
        m = yrs == y
        B[m] -= B[m].mean(0)
    return B

def wauc(y, s, yrs):
    out = []
    for yv in np.unique(yrs):
        m = yrs == yv
        if len(set(y[m])) < 2: continue
        out.append(roc_auc_score(y[m], s[m]))
    return float(np.mean(out))

def run(wall, judge_hi, window, fs, form, cen, lam, C=0.03):
    fit = (YR < wall) & (YR >= (wall - window if window else -10**9))
    jud = (YR >= wall) & (YR < judge_hi)
    if fit.sum() < 500 or jud.sum() < 200: return None
    pm, pri = promise_of(fit)
    pf = np.array([pm.get(f, pri) for f in FLD])
    A = X[:, SETS[fs]]
    if cen: A = centre(A, YR)
    A = np.hstack([pf[:, None] * A, pf[:, None]]) if form == "mult" else np.hstack([pf[:, None], A])
    sc = StandardScaler().fit(A[fit])
    m = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(A[fit]), Y[fit])
    sky = m.decision_function(sc.transform(A))
    sky = (sky - sky[fit].mean()) / (sky[fit].std() + 1e-9)
    base = (pf - pf[fit].mean()) / (pf[fit].std() + 1e-9)
    s = base + lam * sky
    return wauc(Y[jud], s[jud], YR[jud]), (pf, sky, base)

WALLS = [(1985, 1991), (1989, 1995), (1993, 1997)]
print("\n— selection across three inner walls (all pre-1997):", flush=True)
best = None
for window, fs, form, cen, lam in itertools.product(
        (None, 30, 50), ("full", "lean", "slow"), ("cat", "mult"), (False, True), (0.0, 0.25, 0.5, 1.0)):
    scores = [run(w, h, window, fs, form, cen, lam) for w, h in WALLS]
    scores = [s[0] for s in scores if s]
    if len(scores) < 3: continue
    mu = float(np.mean(scores))
    if best is None or mu > best[0]:
        best = (mu, dict(window=window, fs=fs, form=form, cen=cen, lam=lam), scores)
        print(f"   new best {mu:.4f}  {best[1]}  walls {[round(s,4) for s in scores]}", flush=True)
mu, cfg, walls = best
print(f"\n  chosen on the walls: {cfg} · mean {mu:.4f}", flush=True)

# ── ONE shot at the benchmark ──
fit = (YR < WALL_Y) & (YR >= (WALL_Y - cfg["window"] if cfg["window"] else -10**9))
pm, pri = promise_of(fit)
pf = np.array([pm.get(f, pri) for f in FLD])
A = X[:, SETS[cfg["fs"]]]
if cfg["cen"]: A = centre(A, YR)
A = np.hstack([pf[:, None]*A, pf[:, None]]) if cfg["form"] == "mult" else np.hstack([pf[:, None], A])
sc = StandardScaler().fit(A[fit])
m = LogisticRegression(max_iter=3000, C=0.03).fit(sc.transform(A[fit]), Y[fit])
sky = m.decision_function(sc.transform(A)); sky = (sky - sky[fit].mean())/(sky[fit].std()+1e-9)
base = (pf - pf[fit].mean())/(pf[fit].std()+1e-9)
tmask = YR >= WALL_Y
final = base + cfg["lam"]*sky
print(f"\n— the 1997-2024 benchmark, scored within year:", flush=True)
print(f"  promise alone                    {wauc(Y[tmask], base[tmask], YR[tmask]):.4f}", flush=True)
print(f"  sky alone                        {wauc(Y[tmask], sky[tmask], YR[tmask]):.4f}", flush=True)
print(f"  SELECTED MODEL (lam={cfg['lam']})       {wauc(Y[tmask], final[tmask], YR[tmask]):.4f}", flush=True)
json.dump({"cfg": {k: str(v) for k, v in cfg.items()}, "wall_mean": round(mu, 4),
           "promise": round(wauc(Y[tmask], base[tmask], YR[tmask]), 4),
           "sky": round(wauc(Y[tmask], sky[tmask], YR[tmask]), 4),
           "selected": round(wauc(Y[tmask], final[tmask], YR[tmask]), 4)},
          open(f"{BUN}/maxout.json", "w"), indent=1)
