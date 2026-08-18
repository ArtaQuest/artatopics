"""THE MEMORYLESS ASTROLOGY MODEL — natal promise x transiting timing. No momentum, no state.

At prediction time the model reads exactly two things: WHICH FIELD it is (→ its natal chart and its
fitted natal constant) and WHAT YEAR it is (→ the transiting sky). It never looks at the field's
recent growth. Everything is fitted on train years only; scored within year, so the calendar cannot
contribute. Astrologically this is the honest decomposition: the birth chart promises, the transits
time it.
"""
import os, sys, json, numpy as np, pandas as pd, importlib.util
sys.path.insert(0, os.path.expanduser("~/.artaquest-dev/artatopics/analysis/arxivtopics"))
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
BUN=os.path.expanduser("~/.artaquest-dev/artacomp/piecomp")
tr=pd.read_csv(f"{BUN}/train.csv"); te=pd.read_csv(f"{BUN}/test.csv"); sol=pd.read_csv(f"{BUN}/solution.csv")
yte=sol.set_index("id").loc[te["id"]]["target"].to_numpy()
ytr=tr["target"].to_numpy(); tyr=tr["year"].to_numpy(); ty=te["year"].to_numpy()
WALL_Y=1997
def wauc(s):
    s=np.nan_to_num(np.asarray(s,float)); out=[]
    for y in np.unique(ty):
        m=ty==y
        if len(set(yte[m]))<2: continue
        out.append(roc_auc_score(yte[m],s[m]))
    return float(np.mean(out))
def rep(t,s): v=wauc(s); print(f"  {t:<52} {v:.4f}",flush=True); return v
spec=importlib.util.spec_from_file_location("amax", os.path.expanduser(
  "~/.artaquest-dev/artatopics/analysis/arxivtopics/competition/auc_astrology_max.py"))
src=open(spec.origin).read().split('print("building features')[0]
ns={"__file__":spec.origin}; exec(compile(src,spec.origin,"exec"),ns)
rf=ns["row_feats"]; ns["birth"].clear()
bmap=tr.groupby("field")["year"].min().to_dict()
for f in te["field"].unique(): bmap.setdefault(f,WALL_Y)
ns["birth"].update(bmap)
B=lambda df: np.asarray([rf(f,int(t)) for f,t in zip(df["field"],df["year"])],float)
Xtr,Xte=B(tr),B(te)
# natal promise: smoothed per-field constant, fitted on TRAIN labels only
g=tr.groupby("field")["target"].agg(["mean","count"]); prior=ytr.mean(); K=25
promise={f:(r["mean"]*r["count"]+prior*K)/(r["count"]+K) for f,r in g.iterrows()}
ptr=np.array([promise.get(f,prior) for f in tr["field"]])[:,None]
pte=np.array([promise.get(f,prior) for f in te["field"]])[:,None]
print("— memoryless models, within-year AUC (wall 1997):",flush=True)
base=rep("natal promise alone (per-field constant)", pte[:,0])
inner=tyr<1975
def fit_eval(A,Bm,tag):
    best=None
    for C in (0.003,0.01,0.1,1.0):
        sc=StandardScaler().fit(A[inner])
        m=LogisticRegression(max_iter=3000,C=C).fit(sc.transform(A[inner]),ytr[inner])
        s=roc_auc_score(ytr[~inner],m.predict_proba(sc.transform(A[~inner]))[:,1])
        if best is None or s>best[0]: best=(s,("lr",C))
    for d,ne_,mcw in [(3,300,40),(4,300,20),(3,900,80),(6,300,20)]:
        m=xgb.XGBClassifier(max_depth=d,n_estimators=ne_,learning_rate=0.05,min_child_weight=mcw,
          subsample=.8,colsample_bytree=.6,eval_metric="auc",tree_method="hist",random_state=7)
        m.fit(A[inner],ytr[inner]); s=roc_auc_score(ytr[~inner],m.predict_proba(A[~inner])[:,1])
        if s>best[0]: best=(s,("gb",d,ne_,mcw))
    kind=best[1]
    if kind[0]=="lr":
        sc=StandardScaler().fit(A); p=LogisticRegression(max_iter=3000,C=kind[1]).fit(sc.transform(A),ytr).predict_proba(sc.transform(Bm))[:,1]
    else:
        m=xgb.XGBClassifier(max_depth=kind[1],n_estimators=kind[2],learning_rate=0.05,min_child_weight=kind[3],
          subsample=.8,colsample_bytree=.6,eval_metric="auc",tree_method="hist",random_state=7)
        m.fit(A,ytr); p=m.predict_proba(Bm)[:,1]
    print(f"    [inner {best[0]:.4f} · {kind}]",flush=True)
    return rep(tag,p),p
a_sky,_=fit_eval(Xtr,Xte,"transits alone (no promise)")
a_both,p_both=fit_eval(np.hstack([ptr,Xtr]),np.hstack([pte,Xte]),"natal promise x transits (THE MODEL)")
# promise + only the strongest tradition slice, as a lower-variance variant
sub=np.hstack([ptr,Xtr[:,:14]]); sub_te=np.hstack([pte,Xte[:,:14]])
a_lean,_=fit_eval(sub,sub_te,"natal promise + first-harmonic transits only")
print(f"\n  promise alone {base:.4f} · sky adds {a_both-base:+.4f}")
json.dump({"promise_alone":round(base,4),"transits_alone":round(a_sky,4),
           "promise_x_transits":round(a_both,4),"lean":round(a_lean,4),
           "sky_marginal":round(a_both-base,4)}, open(f"{BUN}/memoryless.json","w"),indent=1)
pd.DataFrame({"id":te["id"],"target":p_both}).to_csv(f"{BUN}/submission_memoryless.csv",index=False)
