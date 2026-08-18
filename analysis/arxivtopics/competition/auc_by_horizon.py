import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/.artaquest-dev/artatopics/analysis/arxivtopics"))
sys.path.insert(0, os.path.expanduser("~/.artaquest-dev/artatopics/analysis/arxivtopics/competition"))
from sklearn.metrics import roc_auc_score
import arxiv_fit as af
BUN=os.path.expanduser("~/.artaquest-dev/artacomp/aucomp")
te=pd.read_csv(f"{BUN}/test.csv"); sol=pd.read_csv(f"{BUN}/solution.csv")
yte=sol.set_index("id").loc[te["id"]]["target"].to_numpy()
p_astro=pd.read_csv(f"{BUN}/submission_astro_max.csv")["target"].to_numpy()
names,Yv,labels,future=af.load_lunar(); n=Yv.shape[1]; years=[int(y) for y in labels]; Y0=years[0]; WALL=years.index(1985)
REPO=os.path.expanduser("~/.artaquest-dev/artatopics")
_w=pd.read_csv(f"{REPO}/analysis/citations/rail_works_yearly.csv"); _c=pd.read_csv(f"{REPO}/analysis/citations/citations_received_yearly.csv")
_w=_w.set_index("subfield_id").loc[_c["subfield_id"]].reset_index()
W=_w[[c for c in _w.columns if c[:1].isdigit()][:n]].to_numpy(float)
FI={nm:i for i,nm in enumerate(names)}
mom={nm:(W[j,WALL-1]-W[j,WALL-11])/max(W[j,WALL-11],1)/10 for nm,j in FI.items()}
p_mom=np.array([mom.get(f,0) for f in te["field"]])
yr=te["year"].to_numpy()
print(f"  {'horizon':<16}{'n':>7}{'astrology':>12}{'momentum':>11}")
for lo,hi in [(1985,1994),(1995,2004),(2005,2014),(2015,2024)]:
    m=(yr>=lo)&(yr<=hi)
    if len(set(yte[m]))<2: continue
    print(f"  {str(lo)+'-'+str(hi):<16}{m.sum():>7}{roc_auc_score(yte[m],p_astro[m]):>12.4f}{roc_auc_score(yte[m],p_mom[m]):>11.4f}")
print(f"  {'ALL':<16}{len(yr):>7}{roc_auc_score(yte,p_astro):>12.4f}{roc_auc_score(yte,p_mom):>11.4f}")
# does blending help?
from scipy.stats import rankdata
for w in (0.25,0.5,0.75):
    b=w*rankdata(p_astro)/len(p_astro)+(1-w)*rankdata(p_mom)/len(p_mom)
    print(f"  blend astro={w:.2f}: {roc_auc_score(yte,b):.4f}")
