"""The pie benchmark scored WITHIN YEAR (the only calendar-proof way to score a relative target),
with every model: bars, the deployed phasor, and the full classical-technique astrology stack."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/.artaquest-dev/artatopics/analysis/arxivtopics"))
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb, importlib.util
import arxiv_fit as af, global_phasor as GP
BUN=os.path.expanduser("~/.artaquest-dev/artacomp/piecomp")
tr=pd.read_csv(f"{BUN}/train.csv"); te=pd.read_csv(f"{BUN}/test.csv"); sol=pd.read_csv(f"{BUN}/solution.csv")
yte=sol.set_index("id").loc[te["id"]]["target"].to_numpy()
names,Yv,labels,future=af.load_lunar(); years=[int(y) for y in labels]; Y0=years[0]
WALL_Y=1997; wall=years.index(WALL_Y); FI={nm:i for i,nm in enumerate(names)}
ty=te["year"].to_numpy(); ytr=tr["target"].to_numpy(); tyr=tr["year"].to_numpy()

def wauc(s):
    """AUC computed inside each year, then averaged over years — the calendar cannot contribute."""
    s=np.nan_to_num(np.asarray(s,float)); out=[]
    for y in np.unique(ty):
        m=ty==y
        if len(set(yte[m]))<2: continue
        out.append(roc_auc_score(yte[m],s[m]))
    return float(np.mean(out))
def rep(tag,s):
    print(f"  {tag:<46} within-year AUC {wauc(s):.4f}",flush=True); return wauc(s)

print("— bars, scored within year (wall 1997):",flush=True)
rep("year alone (confound check)", ty)
disp=tr.groupby("field")["target"].mean().to_dict()
b_disp=rep("field disposition (train-era rate)", [disp.get(f,.5) for f in te["field"]])
bars={}
for K in (3,5,10,20):
    g={nm:(Yv[j,wall-1]-Yv[j,wall-1-K])/K for nm,j in FI.items()}
    bars[K]=rep(f"frozen share-momentum, last {K}y", [g.get(f,0) for f in te["field"]])
BAR=max(max(bars.values()), b_disp)

print("\n— the deployed phasor at the 1997 wall:",flush=True)
P_relax,P_exact,P_btopic,P_gain,*_ = GP.fit_wall(wall)
ph={}
for tag,P in (("phasor level+gain+phases (DEPLOYED)",P_gain),("phasor exact projection",P_exact),("57-feature relaxation",P_relax)):
    ph[tag]=rep(tag,[P[FI[f],int(t)-Y0+1]-P[FI[f],int(t)-Y0] for f,t in zip(te["field"],te["year"])])

print("\n— the full classical-technique stack (dashas, profections, returns, progressions,",flush=True)
print("  harmonics, midpoints, nakshatras, sexagenary, tzolkin, gematria):",flush=True)
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
print(f"  {Xtr.shape[1]} features · train {len(tr)} · test {len(te)}",flush=True)
inner=tyr<1975
best=None
for d,ne_,mcw in [(3,300,20),(4,300,20),(4,900,20),(6,300,20),(3,900,80)]:
    m=xgb.XGBClassifier(max_depth=d,n_estimators=ne_,learning_rate=0.05,min_child_weight=mcw,
        subsample=.8,colsample_bytree=.6,eval_metric="auc",tree_method="hist",random_state=7)
    m.fit(Xtr[inner],ytr[inner]); s=roc_auc_score(ytr[~inner],m.predict_proba(Xtr[~inner])[:,1])
    print(f"   gb d{d} n{ne_} mcw{mcw}: inner {s:.4f}",flush=True)
    if best is None or s>best[0]: best=(s,dict(max_depth=d,n_estimators=ne_,min_child_weight=mcw))
sc=StandardScaler().fit(Xtr[inner])
for C in (0.01,0.1):
    lm=LogisticRegression(max_iter=3000,C=C).fit(sc.transform(Xtr[inner]),ytr[inner])
    s=roc_auc_score(ytr[~inner],lm.predict_proba(sc.transform(Xtr[~inner]))[:,1])
    print(f"   logistic C={C}: inner {s:.4f}",flush=True)
    if s>best[0]: best=(s,{"lr_C":C})
print(f"  chosen: {best[1]} (inner {best[0]:.4f})",flush=True)
if "lr_C" in best[1]:
    sc2=StandardScaler().fit(Xtr); p=LogisticRegression(max_iter=3000,C=best[1]["lr_C"]).fit(sc2.transform(Xtr),ytr).predict_proba(sc2.transform(Xte))[:,1]
else:
    m=xgb.XGBClassifier(learning_rate=0.05,subsample=.8,colsample_bytree=.6,eval_metric="auc",
        tree_method="hist",random_state=7,**best[1]); m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
a=rep("ASTROLOGY — all techniques",p)
print(f"\n  bar to beat (best train-only baseline): {BAR:.4f}")
print(f"  best astrology: {max(a,max(ph.values())):.4f}")
json.dump({"bar":round(BAR,4),"astrology_all":round(a,4),
           **{k:round(v,4) for k,v in ph.items()},"disposition":round(b_disp,4),
           **{f"momentum_{k}y":round(v,4) for k,v in bars.items()}},
          open(f"{BUN}/pie_results.json","w"),indent=1)
