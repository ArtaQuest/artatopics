"""Train the astrology features to RANK within each year — the shape the metric actually scores.
Pointwise binary loss treats every row independently; the label is defined by a within-year median,
so a pairwise ranking objective grouped by year matches the target's construction. Selection on the
inner wall (fit <1960, judge 1960-84); the benchmark is touched once, at the end."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/.artaquest-dev/artatopics/analysis/arxivtopics"))
sys.path.insert(0, os.path.expanduser("~/.artaquest-dev/artatopics/analysis/arxivtopics/competition"))
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import xgboost as xgb
import importlib.util
spec = importlib.util.spec_from_file_location("amax", os.path.expanduser(
    "~/.artaquest-dev/artatopics/analysis/arxivtopics/competition/auc_astrology_max.py"))
# reuse only the feature builder without re-running the script's main body
src = open(spec.origin).read().split('print("building features')[0]
ns = {"__file__": spec.origin}; exec(compile(src, spec.origin, "exec"), ns)
build, tr, te, sol = ns["build"], ns["tr"], ns["te"], ns["sol"]
yte = sol.set_index("id").loc[te["id"]]["target"].to_numpy()
Xtr, Xte = build(tr), build(te); ytr = tr["target"].to_numpy()
ytr_y, yte_y = tr["year"].to_numpy(), te["year"].to_numpy()
def grouped(X, y, yrs):
    o = np.argsort(yrs, kind="stable")
    _, cnt = np.unique(yrs[o], return_counts=True)
    return X[o], y[o], cnt
inner = ytr_y < 1960
best = None
for depth in (3, 4, 6):
    for nest in (300, 800):
        Xi, yi, gi = grouped(Xtr[inner], ytr[inner], ytr_y[inner])
        m = xgb.XGBRanker(objective="rank:pairwise", max_depth=depth, n_estimators=nest,
                          learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
                          min_child_weight=20, tree_method="hist", random_state=7)
        m.fit(Xi, yi, group=gi)
        s = roc_auc_score(ytr[~inner], m.predict(Xtr[~inner]))
        print(f"   ranker d{depth} n{nest}: inner {s:.4f}", flush=True)
        if best is None or s > best[0]: best = (s, depth, nest)
_, D, N = best
Xf, yf, gf = grouped(Xtr, ytr, ytr_y)
m = xgb.XGBRanker(objective="rank:pairwise", max_depth=D, n_estimators=N, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.6, min_child_weight=20,
                  tree_method="hist", random_state=7)
m.fit(Xf, yf, group=gf)
p = m.predict(Xte)
print(f"\n  ranker (d{D} n{N}) on the benchmark: {roc_auc_score(yte, p):.4f}   [bar 0.5320]")
for lo, hi in [(1985,1994),(1995,2004),(2005,2014),(2015,2024)]:
    msk = (yte_y >= lo) & (yte_y <= hi)
    print(f"    {lo}-{hi}: {roc_auc_score(yte[msk], p[msk]):.4f}")
pd.DataFrame({"id": te["id"], "target": rankdata(p)/len(p)}).to_csv(
    os.path.expanduser("~/.artaquest-dev/artacomp/aucomp/submission_ranker.csv"), index=False)
