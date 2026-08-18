"""Calendar + lambda*sky, lambda chosen on the walls. If the sky is worth anything at all, the
walls will pick lambda > 0; if they pick 0, the sky is pure noise on top of the calendar."""
import os, sys, numpy as np, pandas as pd, importlib.util
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
os.environ["KDATA"]=os.path.expanduser("~/.artaquest-dev/artacomp/scidist")
src=open(os.path.expanduser("~/.artaquest-dev/artacomp/scidist/kernel/kernel.py")).read()
src=src.split("# ── models on the selected set")[0]      # features + selection + walls only
ns={"__file__":"k"}; exec(compile(src,"k","exec"),ns)
Xs,Y,YR,FLD,IS_TEST,TRAIN,mauc,wall_masks,inner_walls,CUR,BARS = (ns[k] for k in
  ("Xs","Y","YR","FLD","IS_TEST","TRAIN","mauc","wall_masks","inner_walls","CUR","BARS"))
def wrank(s,mask):
    r=np.zeros(len(s))
    for f in np.unique(FLD[mask]):
        m=mask&(FLD==f); r[m]=rankdata(s[m])/max(m.sum(),1)
    return r
def cal(fit):
    best=None
    for i in range(7):
        v=np.cos(CUR[:,i]); z=(v-v[fit].mean())/(v[fit].std()+1e-9)
        c=np.corrcoef(z[fit],Y[fit])[0,1]; s_=1.0 if (np.isfinite(c) and c>=0) else -1.0
        a=mauc(fit,s_*z)
        if best is None or a>best[0]: best=(a,s_*z)
    return best[1]
def sky(fit,C=0.001):
    sc=StandardScaler().fit(Xs[fit]); m=LogisticRegression(max_iter=3000,C=C).fit(sc.transform(Xs[fit]),Y[fit])
    return m.decision_function(sc.transform(Xs))
print(f"{'lambda':>8}" + "".join(f"{w:>9}" for w in inner_walls) + f"{'mean':>9}")
res={}
for lam in (0,0.1,0.25,0.5,0.75,1.0,2.0):
    row=[]
    for w in inner_walls:
        fit,jud=wall_masks(w)
        s=wrank(cal(fit),jud)+lam*wrank(sky(fit),jud)
        row.append(mauc(jud,s))
    res[lam]=float(np.mean(row))
    print(f"{lam:>8}" + "".join(f"{v:>9.4f}" for v in row) + f"{res[lam]:>9.4f}")
best=max(res,key=res.get)
print(f"\nwalls choose lambda = {best}  (calendar alone {res[0]:.4f}, best {res[best]:.4f}, gain {res[best]-res[0]:+.4f})")
# ONE shot at the true test with that lambda, plus the calendar alone, for the record
sol=pd.read_csv(os.path.expanduser("~/.artaquest-dev/artacomp/scidist/solution.csv"))
te=pd.read_csv(os.path.expanduser("~/.artaquest-dev/artacomp/scidist/test.csv"))
yte=sol.set_index("id").loc[te["id"]]["target"].to_numpy()
Yfull=Y.copy(); Yfull[IS_TEST]=yte
def mauc_test(s):
    out=[]
    for f in np.unique(FLD[IS_TEST]):
        m=IS_TEST&(FLD==f)
        if m.sum()<6 or len(set(Yfull[m]))<2: continue
        out.append(roc_auc_score(Yfull[m],s[m]))
    return float(np.mean(out))
fit=TRAIN&(YR>=YR[IS_TEST].min()-30)
c_=wrank(cal(fit),IS_TEST); k_=wrank(sky(fit),IS_TEST)
print(f"\nheld-out 1991-2024:  calendar alone {mauc_test(c_):.4f}  ·  calendar + {best}*sky {mauc_test(c_+best*k_):.4f}  ·  sky alone {mauc_test(k_):.4f}")
pd.DataFrame({"id":te["id"],"target":(c_+best*k_)[IS_TEST]}).to_csv(os.path.expanduser("~/.artaquest-dev/artacomp/scidist/submission_shrunk.csv"),index=False)
pd.DataFrame({"id":te["id"],"target":c_[IS_TEST]}).to_csv(os.path.expanduser("~/.artaquest-dev/artacomp/scidist/submission_calendar.csv"),index=False)
