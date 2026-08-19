#!/usr/bin/env python3
"""ARTAMODEL'S METHOD, ported to the daily record (2026-08-19).

What ArtaModel's 250-fit study (artamatch/research/sidereal/ARTAMODEL.md) established about this model form,
and what is therefore changed here versus my monolithic exact phasor:
  1. DIFFERENCE phasors only. Every absolute-phase term e^{iθ} hurt out of time; every difference term
     e^{i(θ₁−θ₂)} helped. My e^{i(θᵢ(t) − pᵢ)} with pᵢ fitted IS an absolute-phase term. Here each phasor is
     e^{i(θᵢ(t) − θᵢ(natal))} — the transiting body against the category's own birth sky — plus the pair
     separations e^{i(θᵢ(t) − θₖ(t))}, which are differences too. No free phase anywhere.
  2. SPLIT + BOOST, not one sum. Each phasor is its own field |b + w·e^{iφ}|² (complex w as Re/Im), stages
     are picked greedily on an INNER temporal split, each stage shrunk by α chosen there, logistic head.
     Splitting lets a useless term be weighted down instead of poisoning the shared fit.
  3. REPORT BESIDE THE REFERENCE ON THE SAME ROWS, and beside the temporal-half check: a member whose train
     OOF across the two halves of train is below chance while held-out is above is reading the era.
Everything else is the daily setup already in place: 127 arXiv categories, peak-within-7-days labels,
first 80% train / last 20% test per category, kerykeion sidereal Lahiri, per-category AUC averaged.

  python3 analysis/arxivtopics/competition/daily_artamodel.py
"""
import os, sys, json, datetime as dt
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
D = os.path.expanduser("~/.artaquest-dev/artacomp/daily")
daily = pd.read_csv(f"{D}/daily.csv", parse_dates=["date"]).set_index("date")
rel = pd.read_csv(f"{D}/reliable_from.csv", parse_dates=["reliable_from"]).set_index("category")
DAYS = np.array([d.date() for d in daily.index.to_pydatetime()])
E = np.load(f"{D}/ephemeris_ker_1991_2026.npz"); e0 = dt.date.fromisoformat(str(E["d0"])); EB = list(E["bodies"])
off = np.array([(d-e0).days for d in DAYS]); valid = (off>=0)&(off<E["lon"].shape[0])
BOD = ["sun","moon","mercury","venus","mars","jupiter","saturn","true_node"]; SEL = [EB.index(b) for b in BOD]
LON = np.zeros((len(DAYS),8)); LON[valid] = E["lon"][off[valid]][:,SEL]; TH = np.deg2rad(LON)
TAU = 2*np.pi

def peaks_labels(x):
    s = pd.Series(x).rolling(7, center=True, min_periods=1).mean().to_numpy()
    med = pd.Series(s).rolling(90, min_periods=30).median().shift(1).to_numpy()
    sd = pd.Series(s-np.nan_to_num(med, nan=s.mean())).rolling(90, min_periods=30).std().shift(1).to_numpy()
    thr = np.nan_to_num(med, nan=np.inf) + np.maximum(0.2*np.nan_to_num(med, nan=0), np.nan_to_num(sd, nan=np.inf))
    n=len(s); peak=np.zeros(n,bool)
    for i in range(3,n-3):
        if s[i]>=s[i-3:i+4].max() and s[i]>thr[i]: peak[i]=True
    lab=np.zeros(n,bool)
    for i in np.where(peak)[0]: lab[max(0,i-7):i]=True
    return lab.astype(int)

def phasors(th, nat):
    """Difference phasors only: transit-to-natal for each body, and transiting pair separations."""
    names, PH = [], []
    for i,b in enumerate(BOD): names.append(f"t_{b}"); PH.append(np.exp(1j*(th[:,i]-nat[i])))
    for i in range(8):
        for k in range(i+1,8): names.append(f"sep_{BOD[i]}_{BOD[k]}"); PH.append(np.exp(1j*(th[:,i]-th[:,k])))
    return names, np.stack(PH,1)                                   # (T, P) complex

def field_feats(z, w):
    """|b + w·z|² is quadratic in (b_re,b_im,w_re,w_im); as a SCORE we use the closed form:
       |b + w z|² = |b|² + |w|² + 2 Re(conj(b) w z). The linear part in z is 2Re(conj(b)w·z) = c1·Re z + c2·Im z
       for real c1,c2 — so the field's information is exactly [Re z, Im z] with two free real weights. We keep the
       exact |·|² by fitting (c1,c2) then recovering the field; for the stagewise logistic, [Re z, Im z] is it."""
    return np.stack([z.real, z.imag], 1)

def boost(Z, names, y, tr_fit, tr_val, n_stages=8, alphas=(0.25,0.5,1.0)):
    """ArtaModel's greedy stagewise boost over split per-phasor fields, selection on the inner split.
    Each stage: for every phasor not yet used, fit a 2-param logistic on [Re z, Im z] to the current
    residual (a gradient step on the logistic loss), pick the one that most improves inner AUC, shrink by α."""
    F = np.zeros(Z.shape[0]); used=[]; stages=[]
    # F0: the base rate
    p0 = np.clip(y[tr_fit].mean(),1e-3,1-1e-3); F[:] = np.log(p0/(1-p0))
    for s in range(n_stages):
        best=None
        sig = 1/(1+np.exp(-F)); resid = y - sig                   # negative gradient of logistic loss
        for j in range(Z.shape[1]):
            if j in used: continue
            X = field_feats(Z[:,j], None)
            # least-squares step on the residual (the gradient-boosting working response)
            A = X[tr_fit]; c = np.linalg.lstsq(np.c_[np.ones(len(A)),A], resid[tr_fit], rcond=None)[0]
            h = c[0] + X@c[1:]
            for a in alphas:
                Fn = F + a*h
                if len(set(y[tr_val]))<2: continue
                sc = roc_auc_score(y[tr_val], Fn[tr_val])
                if best is None or sc>best[0]: best=(sc,j,a,c,h)
        if best is None: break
        sc,j,a,c,h = best
        prev = roc_auc_score(y[tr_val], F[tr_val]) if len(set(y[tr_val]))>1 else 0.5
        if sc <= prev + 1e-4: break                                # early stop on the inner split
        F = F + a*h; used.append(j); stages.append(dict(stage=s+1, phasor=names[j], alpha=a, inner_auc=round(sc,4)))
    return F, stages

rows=[]; allstages=[]
for cat in daily.columns:
    if cat not in rel.index: continue
    start = max(np.searchsorted(DAYS, rel.loc[cat,"reliable_from"].date()), int(np.argmax(valid)))
    x = daily[cat].to_numpy(float)[start:]
    if len(x) < 8*365: continue
    idx = np.arange(start, start+len(x)); n=len(x); cut=int(n*0.8); k=int(cut*0.75)
    y = peaks_labels(x); th = TH[idx]; nat = TH[start]            # the category's birth sky
    names, Z = phasors(th, nat)
    tr_fit = np.arange(k); tr_val = np.arange(k,cut); te = np.arange(cut,n)
    F, stages = boost(Z, names, y, tr_fit, tr_val)
    au = roc_auc_score(y[te], F[te]) if len(set(y[te]))>1 else np.nan
    # reference on the SAME rows: the calendar (day-of-year, weekday) + memory (trailing 7/30d means)
    doy = np.array([d.timetuple().tm_yday for d in DAYS[idx]],float); dow=np.array([d.weekday() for d in DAYS[idx]],float)
    Xc = np.stack([np.cos(TAU*doy/365.25),np.sin(TAU*doy/365.25),np.cos(2*TAU*doy/365.25),np.sin(2*TAU*doy/365.25),np.cos(TAU*dow/7),np.sin(TAU*dow/7)],1)
    Xm = np.nan_to_num(np.stack([pd.Series(x).rolling(7,min_periods=1).mean().shift(1).to_numpy(), pd.Series(x).rolling(30,min_periods=1).mean().shift(1).to_numpy()],1))
    def ref(X):
        if len(set(y[:cut]))<2 or len(set(y[te]))<2: return np.nan
        m = LogisticRegression(max_iter=2000, C=0.1).fit(X[:cut], y[:cut]); return roc_auc_score(y[te], m.decision_function(X[te]))
    a_cal, a_mem = ref(Xc), ref(Xm)
    # temporal-half check (ArtaModel §8): refit the chosen stages on the FIRST half of train, score the SECOND half
    h1 = np.arange(cut//2); h2 = np.arange(cut//2, cut)
    Fh, _ = boost(Z, names, y, h1[:int(len(h1)*0.75)], h1[int(len(h1)*0.75):], n_stages=len(stages) or 1)
    a_half = roc_auc_score(y[h2], Fh[h2]) if len(set(y[h2]))>1 else np.nan
    # circular-shift null
    rng = np.random.RandomState(3); nulls=[]
    for _ in range(2):
        sh = rng.randint(400, n-400); yn = np.roll(y, sh); nulls.append(roc_auc_score(yn[te], F[te]) if len(set(yn[te]))>1 else np.nan)
    rows.append(dict(category=cat, days=n, stages=len(stages), picks="|".join(s["phasor"] for s in stages), auc_artamodel=au, auc_calendar=a_cal, auc_memory=a_mem,
                     auc_temporal_half=a_half, null=float(np.nanmean(nulls))))
    allstages.append({"category":cat, "stages":stages})
    print(f"  {cat:<16} stages {len(stages)} · artamodel {au:.3f} · calendar {a_cal:.3f} · memory {a_mem:.3f} · temporal-half {a_half:.3f} · null {np.nanmean(nulls):.3f} · {'/'.join(s['phasor'] for s in stages[:3])}", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{D}/daily_artamodel.csv", index=False); json.dump(allstages, open(f"{D}/daily_artamodel_stages.json","w"), indent=1)
print(f"\n{len(df)} categories · ArtaModel's method (difference phasors, split+boost on inner split), per-category AUC averaged:")
for k_ in ("auc_artamodel","auc_calendar","auc_memory","auc_temporal_half","null"):
    v=df[k_].dropna(); print(f"  {k_:<18} mean {v.mean():.4f} · median {v.median():.4f} · >0.5 in {(v>0.5).mean()*100:.0f}%")
from collections import Counter
c = Counter(p for r in rows for p in r["picks"].split("|") if p)
print("  most-picked phasors:", c.most_common(8))
print(f"  stages chosen: mean {df.stages.mean():.2f} · zero stages (inner split refused every phasor) in {(df.stages==0).mean()*100:.0f}%")
json.dump({k_: round(float(df[k_].dropna().mean()),4) for k_ in ("auc_artamodel","auc_calendar","auc_memory","auc_temporal_half","null")} | {"mean_stages": round(float(df.stages.mean()),2), "top_picks": c.most_common(8)},
          open(f"{D}/daily_artamodel_summary.json","w"), indent=1)
