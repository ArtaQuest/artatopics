"""Pure sky: the SAME models with the promise column REMOVED. This is astrology alone."""
import os, sys, json, importlib.util, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.expanduser("~/.artaquest-dev/artatopics/analysis/arxivtopics"))
BUN=os.path.expanduser("~/.artaquest-dev/artacomp/piecomp")
tr=pd.read_csv(f"{BUN}/train.csv"); te=pd.read_csv(f"{BUN}/test.csv"); sol=pd.read_csv(f"{BUN}/solution.csv")
yte=sol.set_index("id").loc[te["id"]]["target"].to_numpy(); WALL_Y=1997; WINDOW=30
ALL=pd.concat([tr[["field","year","target"]], te[["field","year"]].assign(target=yte)],ignore_index=True)
spec=importlib.util.spec_from_file_location("amax", os.path.expanduser(
 "~/.artaquest-dev/artatopics/analysis/arxivtopics/competition/auc_astrology_max.py"))
src=open(spec.origin).read().split('print("building features')[0]
ns={"__file__":spec.origin}; exec(compile(src,spec.origin,"exec"),ns)
rf=ns["row_feats"]; ns["birth"].clear()
bmap=tr.groupby("field")["year"].min().to_dict()
for f in te["field"].unique(): bmap.setdefault(f,WALL_Y)
ns["birth"].update(bmap)
X=np.asarray([rf(f,int(t)) for f,t in zip(ALL["field"],ALL["year"])],float)
Y=ALL["target"].to_numpy(); YR=ALL["year"].to_numpy(); FLD=ALL["field"].to_numpy()
NB=7
SETS={"slow":np.r_[[2,3,4,5,6,9,10,11,12,13]],"lean":np.r_[0:2*NB],"full":np.arange(X.shape[1])}
def wauc(y,s,yrs):
    o=[]
    for v in np.unique(yrs):
        m=yrs==v
        if len(set(y[m]))<2: continue
        o.append(roc_auc_score(y[m],s[m]))
    return float(np.mean(o))
fit=(YR<WALL_Y)&(YR>=WALL_Y-WINDOW); tm=YR>=WALL_Y
d=pd.DataFrame({"f":FLD[fit],"y":Y[fit]}).groupby("f")["y"].agg(["mean","count"]); pri=Y[fit].mean()
pm={f:(r["mean"]*r["count"]+pri*25)/(r["count"]+25) for f,r in d.iterrows()}
pf=np.array([pm.get(f,pri) for f in FLD])
print(f"  promise alone (reference)          {wauc(Y[tm],pf[tm],YR[tm]):.4f}")
for k,idx in SETS.items():
    for C in (0.003,0.03,0.3):
        A=X[:,idx]; sc=StandardScaler().fit(A[fit])
        m=LogisticRegression(max_iter=3000,C=C).fit(sc.transform(A[fit]),Y[fit])
        s=m.decision_function(sc.transform(A))
        print(f"  PURE SKY {k:<6} C={C:<6}            {wauc(Y[tm],s[tm],YR[tm]):.4f}")
