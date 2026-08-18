#!/usr/bin/env python3
"""THE STACK — learn the best combination of memoryless models, weights chosen on inner walls.

Base models, each producing a within-year RANK (the only thing the metric reads), all fitted on a
30-year rolling window before whatever wall they are asked about:
  promise    per-field natal constant, no sky
  slow       outer-planet transits, multiplied by the promise (transit activates promise)
  lean       first-harmonic transits to natal
  full       all 287 classical features
  phasor     the deployed |b + A sum a_i e^{i(theta-p)}|^2, its predicted change in share
  west/ved/chi/may/num  one tradition at a time

Weights come from a simplex grid scored on THREE inner walls (1985/1989/1993), then the stack is
scored ONCE on 1997-2024. Ranking within year before blending is deliberate: the models have wildly
different scales, and only order is scored.

  python3 analysis/arxivtopics/competition/auc_stack.py
"""
import os, sys, json, itertools, importlib.util
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.expanduser("~/.artaquest-dev/artatopics/analysis/arxivtopics"))
import arxiv_fit as af, global_phasor as GP

BUN = os.path.expanduser("~/.artaquest-dev/artacomp/piecomp")
tr = pd.read_csv(f"{BUN}/train.csv"); te = pd.read_csv(f"{BUN}/test.csv"); sol = pd.read_csv(f"{BUN}/solution.csv")
yte = sol.set_index("id").loc[te["id"]]["target"].to_numpy()
WALL_Y = 1997; WINDOW = 30
ALL = pd.concat([tr[["field","year","target"]], te[["field","year"]].assign(target=yte)], ignore_index=True)
spec = importlib.util.spec_from_file_location("amax", os.path.expanduser(
    "~/.artaquest-dev/artatopics/analysis/arxivtopics/competition/auc_astrology_max.py"))
src = open(spec.origin).read().split('print("building features')[0]
ns = {"__file__": spec.origin}; exec(compile(src, spec.origin, "exec"), ns)
rf = ns["row_feats"]; ns["birth"].clear()
bmap = tr.groupby("field")["year"].min().to_dict()
for f in te["field"].unique(): bmap.setdefault(f, WALL_Y)
ns["birth"].update(bmap)
print("building features …", flush=True)
X = np.asarray([rf(f, int(t)) for f, t in zip(ALL["field"], ALL["year"])], float)
Y = ALL["target"].to_numpy(); YR = ALL["year"].to_numpy(); FLD = ALL["field"].to_numpy()
names, Yv, labels, future = af.load_lunar(); years=[int(y) for y in labels]; Y0=years[0]
FI = {nm: i for i, nm in enumerate(names)}
NB = 7
SETS = {"slow": np.r_[[2,3,4,5,6,9,10,11,12,13]], "lean": np.r_[0:2*NB], "full": np.arange(X.shape[1]),
        "west": np.r_[0:2*NB], "ved": np.r_[2*NB+2*NB*6+5 : 2*NB+2*NB*6+5+4],
        "chi": np.r_[X.shape[1]-26 : X.shape[1]-17], "may": np.r_[X.shape[1]-17 : X.shape[1]-9],
        "num": np.r_[X.shape[1]-9 : X.shape[1]]}
def wauc(y, s, yrs):
    out=[]
    for v in np.unique(yrs):
        m = yrs==v
        if len(set(y[m]))<2: continue
        out.append(roc_auc_score(y[m], s[m]))
    return float(np.mean(out))
def wrank(s, yrs):
    r = np.zeros(len(s))
    for v in np.unique(yrs):
        m = yrs==v; r[m] = rankdata(s[m])/max(m.sum(),1)
    return r
def promise_vec(fit, K=25):
    d = pd.DataFrame({"f":FLD[fit],"y":Y[fit]}).groupby("f")["y"].agg(["mean","count"]); pri=Y[fit].mean()
    pm = {f:(r["mean"]*r["count"]+pri*K)/(r["count"]+K) for f,r in d.iterrows()}
    return np.array([pm.get(f,pri) for f in FLD])
def sky_vec(fit, key, pf, C=0.03):
    A = X[:, SETS[key]]
    A = np.hstack([pf[:,None]*A, pf[:,None]]) if key=="slow" else np.hstack([pf[:,None], A])
    sc = StandardScaler().fit(A[fit])
    m = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(A[fit]), Y[fit])
    return m.decision_function(sc.transform(A))
_phcache = {}
def phasor_vec(wall_y):
    if wall_y not in _phcache:
        P = GP.fit_wall(years.index(wall_y))[3]
        _phcache[wall_y] = np.array([P[FI[f], int(t)-Y0+1]-P[FI[f], int(t)-Y0] for f,t in zip(FLD,YR)])
    return _phcache[wall_y]
MODELS = ["promise","slow","lean","full","phasor","west","ved","chi","may","num"]
def preds_at(wall_y):
    fit = (YR < wall_y) & (YR >= wall_y - WINDOW)
    pf = promise_vec(fit)
    out = {"promise": pf, "phasor": phasor_vec(wall_y)}
    for k in ("slow","lean","full","west","ved","chi","may","num"):
        out[k] = sky_vec(fit, k, pf)
    return out
WALLS = [(1985,1991),(1989,1995),(1993,1997)]
print("scoring base models on the inner walls …", flush=True)
wall_ranks = []
for w,h in WALLS:
    P = preds_at(w); jud = (YR>=w)&(YR<h)
    wall_ranks.append((jud, {k: wrank(P[k], YR)[jud] for k in MODELS}))
    print(f"  wall {w}: " + " ".join(f"{k} {wauc(Y[jud], P[k][jud], YR[jud]):.4f}" for k in MODELS), flush=True)
print("\nsearching the simplex …", flush=True)
CAND = ["promise","slow","lean","full","phasor"]
best=None
grid=[g for g in itertools.product(np.arange(0,1.01,0.125), repeat=len(CAND)) if abs(sum(g)-1)<1e-9]
for g in grid:
    sc=[]
    for jud, R in wall_ranks:
        s = sum(gi*R[k] for gi,k in zip(g,CAND))
        sc.append(wauc(Y[jud], s, YR[jud]))
    mu=float(np.mean(sc))
    if best is None or mu>best[0]: best=(mu, g, sc)
mu,g,sc = best
mix={k:float(v) for k,v in zip(CAND,g) if v}
print(f"  chosen mix {mix} · wall mean {mu:.4f} · walls {[round(x,4) for x in sc]}", flush=True)
print("\n— ONE shot at 1997-2024, scored within year:", flush=True)
P = preds_at(WALL_Y); tm = YR>=WALL_Y
for k in MODELS: print(f"  {k:<10} {wauc(Y[tm], P[k][tm], YR[tm]):.4f}", flush=True)
stack = sum(gi*wrank(P[k],YR) for gi,k in zip(g,CAND))
S = wauc(Y[tm], stack[tm], YR[tm])
print(f"\n  THE STACK  {S:.4f}", flush=True)
json.dump({"mix":mix,"wall_mean":round(mu,4),"stack_benchmark":round(S,4),
           "solo":{k:round(wauc(Y[tm],P[k][tm],YR[tm]),4) for k in MODELS}},
          open(f"{BUN}/stack.json","w"), indent=1)
pd.DataFrame({"id":te["id"],"target":stack[tm]}).to_csv(f"{BUN}/submission_stack.csv", index=False)
