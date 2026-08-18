"""Re-score the momentum bar on the information a competitor actually has.

test.csv gives (field, year) only; train.csv stops at 1984. So a competitor CANNOT read the field's
works in 2015 — but my earlier momentum baseline did, which is why it reached 0.5820. Two honest
bars, both computed from train-era data only:
  frozen momentum  — the field's growth over its last pre-wall years, held constant for every test year
  field disposition— the field's train-era rate of being above the within-year median (its natal luck)
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/.artaquest-dev/artatopics/analysis/arxivtopics"))
from sklearn.metrics import roc_auc_score
import arxiv_fit as af
BUN = os.path.expanduser("~/.artaquest-dev/artacomp/aucomp")
tr = pd.read_csv(f"{BUN}/train.csv"); te = pd.read_csv(f"{BUN}/test.csv"); sol = pd.read_csv(f"{BUN}/solution.csv")
yte = sol.set_index("id").loc[te["id"]]["target"].to_numpy(); usage = sol.set_index("id").loc[te["id"]]["Usage"].to_numpy()
names, Yv, labels, future = af.load_lunar(); n = Yv.shape[1]
years = [int(y) for y in labels]; Y0 = years[0]; WALL = years.index(1985)
REPO = os.path.expanduser("~/.artaquest-dev/artatopics")
_w = pd.read_csv(f"{REPO}/analysis/citations/rail_works_yearly.csv"); _c = pd.read_csv(f"{REPO}/analysis/citations/citations_received_yearly.csv")
_w = _w.set_index("subfield_id").loc[_c["subfield_id"]].reset_index()
W = _w[[c for c in _w.columns if c[:1].isdigit()][:n]].to_numpy(float)
FI = {nm: i for i, nm in enumerate(names)}
def rep(tag, s):
    s = np.nan_to_num(np.asarray(s, float))
    o = roc_auc_score(yte, s); pu = roc_auc_score(yte[usage=='Public'], s[usage=='Public']); pr = roc_auc_score(yte[usage=='Private'], s[usage=='Private'])
    print(f"  {tag:<46} overall {o:.4f} · public {pu:.4f} · private {pr:.4f}", flush=True); return o
# frozen momentum: growth over the last K pre-wall years
for K in (5, 10, 20):
    g = {}
    for nm, j in FI.items():
        a, b = W[j, WALL-1-K], W[j, WALL-1]
        g[nm] = (b - a) / max(a, 1) / K
    rep(f"frozen momentum, last {K}y before the wall", [g.get(f, 0) for f in te["field"]])
# field disposition: train-era rate of being above the within-year median
disp = tr.groupby("field")["target"].mean().to_dict()
rep("field disposition (train-era label rate)", [disp.get(f, .5) for f in te["field"]])
# and the leaky one, for the record
def leaky(f, t):
    j = FI[f]; i = t - Y0
    return np.mean([(W[j, i-k+1]-W[j, i-k])/max(W[j, i-k],1) for k in (1,2,3,5,8) if i-k >= 1] or [0])
rep("LEAKY momentum (reads works at test year)", [leaky(f, int(t)) for f, t in zip(te["field"], te["year"])])
