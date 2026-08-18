"""Is the below-0.5 real? Score every single transiting body on held-out with direction fixed on
train, and with direction fixed on TEST (the oracle). If the oracle is well above 0.5 and the train
direction is below, the sign flipped across the wall — era shift at the level of the sign."""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
D=os.path.expanduser("~/.artaquest-dev/artacomp/scidist")
tr=pd.read_csv(f"{D}/train.csv"); te=pd.read_csv(f"{D}/test.csv"); sol=pd.read_csv(f"{D}/solution.csv")
eph=pd.read_csv(f"{D}/ephemeris.csv").set_index("year")
yte=sol.set_index("id").loc[te["id"]]["target"].to_numpy()
BOD=["mars","jupiter","saturn","uranus","neptune","pluto","node"]
def mauc(df,y,s):
    out=[]
    for f in df["field"].unique():
        m=(df["field"]==f).to_numpy()
        if m.sum()<6 or len(set(y[m]))<2: continue
        out.append(roc_auc_score(y[m],s[m]))
    return float(np.mean(out))
tr30=tr[tr.year>=1961]; ytr=tr30["target"].to_numpy()
print(f"{'body':<10}{'train dir→held':>16}{'oracle dir→held':>17}{'train mAUC':>12}")
for b in BOD:
    v_tr=np.cos(np.deg2rad(eph.loc[tr30.year,f"{b}_lon_deg"].to_numpy()))
    v_te=np.cos(np.deg2rad(eph.loc[te.year,f"{b}_lon_deg"].to_numpy()))
    c=np.corrcoef(v_tr,ytr)[0,1]; s=1.0 if c>=0 else -1.0
    a_tr=mauc(tr30,ytr,s*v_tr); a_te=mauc(te,yte,s*v_te)
    a_or=max(mauc(te,yte,v_te),mauc(te,yte,-v_te))
    print(f"{b:<10}{a_te:>16.4f}{a_or:>17.4f}{a_tr:>12.4f}")
