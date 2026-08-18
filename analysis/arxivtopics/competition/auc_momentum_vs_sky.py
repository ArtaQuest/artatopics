import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/.artaquest-dev/artatopics/analysis/arxivtopics"))
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import arxiv_fit as af
BUN = os.path.expanduser("~/.artaquest-dev/artacomp/aucomp")
tr=pd.read_csv(f"{BUN}/train.csv"); te=pd.read_csv(f"{BUN}/test.csv"); sol=pd.read_csv(f"{BUN}/solution.csv")
yte=sol.set_index("id").loc[te["id"]]["target"].to_numpy(); usage=sol.set_index("id").loc[te["id"]]["Usage"].to_numpy()
ytr=tr["target"].to_numpy()
names,Yv,labels,future=af.load_lunar(); n=Yv.shape[1]
TH,_=af.sky_lunar(labels+future); years=[int(y) for y in labels]; Y0=years[0]
REPO=os.path.expanduser("~/.artaquest-dev/artatopics")
_w=pd.read_csv(f"{REPO}/analysis/citations/rail_works_yearly.csv"); _c=pd.read_csv(f"{REPO}/analysis/citations/citations_received_yearly.csv")
_w=_w.set_index("subfield_id").loc[_c["subfield_id"]].reset_index()
W=_w[[c for c in _w.columns if c[:1].isdigit()][:n]].to_numpy(float)
FI={nm:i for i,nm in enumerate(names)}
birth=tr.groupby("field")["year"].min().to_dict()
AYAN=np.deg2rad(23.85)
ASP=[(0,10),(np.pi,10),(2*np.pi/3,8),(np.pi/2,8),(np.pi/3,6)]
LET={c:(i%9)+1 for i,c in enumerate("abcdefghijklmnopqrstuvwxyz")}
def droot(x):
    x=int(abs(x))
    while x>9: x=sum(int(c) for c in str(x))
    return x or 9
def astro(f,t):
    b=birth.get(f,t); nat=TH[b-Y0]; cur=TH[t-Y0]
    d=(cur-nat)%(2*np.pi); dS=((cur-AYAN)%(2*np.pi)-(nat-AYAN)%(2*np.pi))%(2*np.pi)
    a=[]
    for ang,orb in ASP:
        sep=np.abs(((d-ang+np.pi)%(2*np.pi))-np.pi); a.append(np.clip(1-sep/np.deg2rad(orb),0,1))
    nak=((np.floor(((cur-AYAN)%(2*np.pi))/(2*np.pi)*27)-np.floor(((nat-AYAN)%(2*np.pi))/(2*np.pi)*27))%27)
    bb,cb=(b-4)%12,(t-4)%12; bs,cs=(b-4)%10,(t-4)%10; db,ds=(cb-bb)%12,(cs-bs)%10
    g=sum(LET.get(c,0) for c in f.lower()); gr=droot(g); yr=droot(t); pers=droot(gr+yr)
    return np.concatenate([np.cos(d),np.sin(d),np.concatenate(a),np.cos(dS),np.sin(dS),
        np.cos(2*np.pi*nak/27),np.sin(2*np.pi*nak/27),
        [np.cos(2*np.pi*db/12),np.sin(2*np.pi*db/12),np.cos(2*np.pi*ds/10),np.sin(2*np.pi*ds/10),
         1.0*(db%4==0),1.0*(db==6),np.cos(2*np.pi*((t-b)%60)/60),np.sin(2*np.pi*((t-b)%60)/60),
         np.cos(2*np.pi*((t-b)%260)/260),np.sin(2*np.pi*((t-b)%260)/260),
         np.cos(2*np.pi*((t-b)%13)/13),np.sin(2*np.pi*((t-b)%13)/13),
         np.cos(2*np.pi*((t-b)%20)/20),np.sin(2*np.pi*((t-b)%20)/20),
         np.cos(2*np.pi*pers/9),np.sin(2*np.pi*pers/9),1.0*(pers in(1,9)),1.0*(g%11==0),
         1.0*(gr==yr),(gr*yr)/81.0]])
def mom(f,t):
    j=FI[f]; i=t-Y0; w=W[j]
    g=lambda k:(w[i-k+1]-w[i-k])/max(w[i-k],1) if i-k>=1 else 0.0
    last=[g(k) for k in (1,2,3,5,8)]
    return np.array(last+[np.mean(last),np.log1p(w[i]),w[i]/max(W[:,i].sum(),1)])
def build(df,fn): return np.asarray([fn(f,int(t)) for f,t in zip(df["field"],df["year"])],float)
Atr,Ate=build(tr,astro),build(te,astro); Mtr,Mte=build(tr,mom),build(te,mom)
def run(A,B,tag,depth=5,n_=700):
    m=xgb.XGBClassifier(max_depth=depth,n_estimators=n_,learning_rate=0.05,subsample=.8,
        colsample_bytree=.8,min_child_weight=8,eval_metric="auc",random_state=7)
    m.fit(A,ytr); p=m.predict_proba(B)[:,1]
    o,pu,pr=roc_auc_score(yte,p),roc_auc_score(yte[usage=='Public'],p[usage=='Public']),roc_auc_score(yte[usage=='Private'],p[usage=='Private'])
    print(f"  {tag:<32} overall {o:.4f} · public {pu:.4f} · private {pr:.4f}",flush=True)
    return p,o
p_a,_=run(Atr,Ate,"astrology + numerology only")
p_m,o_m=run(Mtr,Mte,"momentum only (no sky)")
p_c,o_c=run(np.hstack([Mtr,Atr]),np.hstack([Mte,Ate]),"momentum + all traditions")
best=max([(o_m,p_m,"momentum"),(o_c,p_c,"momentum+traditions")],key=lambda x:x[0])
print(f"\n  astrology's contribution on top of momentum: {o_c-o_m:+.4f} AUC")
pd.DataFrame({"id":te["id"],"target":best[1]}).to_csv(f"{BUN}/submission_best.csv",index=False)
json.dump({"astrology_only":round(float(roc_auc_score(yte,p_a)),4),"momentum_only":round(o_m,4),
           "combined":round(o_c,4),"delta":round(o_c-o_m,4),"best":best[2]},
          open(f"{BUN}/auc_final.json","w"),indent=1)
print(f"  best model: {best[2]} ({best[0]:.4f}) — submission_best.csv written")
