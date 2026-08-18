#!/usr/bin/env python3
"""The pie task with REAL charts — PyJHora (Vedic) and iztro (Zi Wei Dou Shu) features.

Base = the damped 10-year log-trend (the honest non-astrological ceiling: KL 0.0793 held out).
Question: does any chart feature set, entering as a shrunk log-space correction chosen on rolling
inner walls, lower the KL to the true pie? Feature families, all functions of the year's chart:

  vedic_lon      sin/cos of the 9 grahas' sidereal longitudes (+ Lagna)
  vedic_pairs    sin/cos of graha pair separations (36 pairs) — the aspect/yoga geometry
  vedic_rasi     one-hot rasi (sign) of Moon, Jupiter, Saturn — 12 each
  vedic_nak      one-hot nakshatra of the Moon (27) — what Vimshottari is seeded from
  panchanga      tithi, vaara, yoga, karana as cyclic coordinates
  ziwei_life     one-hot major stars in the life palace, soul star, five-elements class
  ziwei_all      counts of each major star across the 12 palaces (which palace holds which star)

For each family: log p = log trend + lam * (log softmax(B·f(t)) - log clim), B ridge-fitted on the
fit side of each wall, lam in {0, .25, .5, 1} chosen on the walls. Then ONE shot at 1992-2025.

  python3 analysis/arxivtopics/competition/pie_engines.py
"""
import os, sys, json, itertools, csv
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import arxiv_fit as af

names, Yv, labels, future = af.load_lunar()
tv = af.META["topic_valid"]; years = np.array([int(y) for y in labels]); J, n = Yv.shape
usable = np.where(tv.sum(0) >= 20)[0]; T = np.arange(usable[0], usable[-1] + 1)
S = np.clip(Yv, 0, None) * tv; S = S / np.maximum(S.sum(0, keepdims=True), 1e-12)
cut = int(len(T)*0.8); TR, TE = T[:cut], T[cut:]
EPS = 1e-6
def kl(P, Q):
    Pn = np.clip(P, EPS, None); Pn /= Pn.sum(0, keepdims=True); Qn = np.clip(Q, 0, None); Qn /= Qn.sum(0, keepdims=True)
    return float(np.mean((Qn*(np.log(np.maximum(Qn,1e-12))-np.log(Pn))).sum(0)))
def softmax_cols(Z): Z = Z - Z.max(0, keepdims=True); P = np.exp(Z); return P/P.sum(0, keepdims=True)

CH = os.path.expanduser("~/.artaquest-dev/artacomp/scidist3/charts")
V = pd.read_csv(f"{CH}/vedic.csv").set_index("year"); Z = {z["year"]: z for z in json.load(open(f"{CH}/ziwei.json")) if "error" not in z}
GR = ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu"]
TAU = 2*np.pi
def yr(idx): return years[idx]
def f_vedic_lon(idx):
    y = yr(idx); L = np.deg2rad(V.loc[y, [f"{g}_sid" for g in GR]].to_numpy(float)); A = np.deg2rad(V.loc[y, "asc"].to_numpy(float))
    return np.concatenate([np.cos(L), np.sin(L), np.cos(A)[:,None], np.sin(A)[:,None]], 1)
def f_vedic_pairs(idx):
    y = yr(idx); L = np.deg2rad(V.loc[y, [f"{g}_sid" for g in GR]].to_numpy(float)); C = []
    for i in range(9):
        for k in range(i+1, 9): d = L[:,i]-L[:,k]; C += [np.cos(d), np.sin(d)]
    return np.stack(C, 1)
def onehot(v, K): O = np.zeros((len(v), K)); O[np.arange(len(v)), np.clip(v.astype(int), 0, K-1)] = 1; return O
def f_vedic_rasi(idx):
    y = yr(idx); return np.concatenate([onehot(np.floor(V.loc[y, f"{g}_sid"].to_numpy(float)/30) % 12, 12) for g in ("Moon","Jupiter","Saturn")], 1)
def f_vedic_nak(idx):
    y = yr(idx); return onehot(np.floor(V.loc[y, "Moon_sid"].to_numpy(float)/(360/27)) % 27, 27)
def f_panchanga(idx):
    y = yr(idx); C = []
    for col, K in (("tithi",30),("vaara",7),("yoga",27),("karana",60),("nakshatra",27)):
        v = V.loc[y, col].to_numpy(float); C += [np.cos(TAU*v/K), np.sin(TAU*v/K)]
    return np.stack(C, 1)
STARS = sorted({s for z in Z.values() for p in z["palaces"] for s in p["major"]})
FIVE = sorted({z["five"] for z in Z.values()}); SOUL = sorted({z["soul"] for z in Z.values()})
def f_ziwei_life(idx):
    out = []
    for y in yr(idx):
        z = Z[int(y)]; life = z["palaces"][0]["major"]
        out.append([1.0*(s in life) for s in STARS] + [1.0*(z["five"]==f) for f in FIVE] + [1.0*(z["soul"]==s) for s in SOUL])
    return np.asarray(out, float)
def f_ziwei_all(idx):
    out = []
    for y in yr(idx):
        z = Z[int(y)]; row = []
        for s in STARS:
            pos = [pi for pi, p in enumerate(z["palaces"]) if s in p["major"]]
            row += [np.cos(TAU*pos[0]/12) if pos else 0.0, np.sin(TAU*pos[0]/12) if pos else 0.0]
        out.append(row)
    return np.asarray(out, float)
FAM = {"vedic_lon": f_vedic_lon, "vedic_pairs": f_vedic_pairs, "vedic_rasi": f_vedic_rasi, "vedic_nak": f_vedic_nak,
       "panchanga": f_panchanga, "ziwei_life": f_ziwei_life, "ziwei_all": f_ziwei_all}

def m_trend(fit_idx, pred_idx, K=10, phi=0.8):
    src = fit_idx[-K:]; L = np.log(np.clip(S[:, src], EPS, None)); x = np.arange(len(src), dtype=float); xm = x.mean()
    slope = ((x-xm)[None,:]*(L-L.mean(1,keepdims=True))).sum(1)/((x-xm)**2).sum(); last = L[:,-1]
    return softmax_cols(np.stack([last + slope*phi*(1-phi**h)/(1-phi) for h in range(1, len(pred_idx)+1)], 1))
def m_clim(fit_idx, pred_idx):
    c = S[:, fit_idx].mean(1); c /= c.sum(); return np.repeat(c[:,None], len(pred_idx), 1)
def m_sky(fit_idx, pred_idx, fn, lam=0.1, iters=1500, lr=0.5):
    Ftr, Fte = fn(fit_idx), fn(pred_idx); mu, sd = Ftr.mean(0), Ftr.std(0)+1e-9; Ftr=(Ftr-mu)/sd; Fte=(Fte-mu)/sd
    Ftr = np.hstack([np.ones((len(fit_idx),1)), Ftr]); Fte = np.hstack([np.ones((len(pred_idx),1)), Fte])
    Str = S[:, fit_idx]; clim = Str.mean(1); clim /= clim.sum()
    B = np.zeros((J, Ftr.shape[1])); B[:,0] = np.log(np.maximum(clim, EPS))
    for _ in range(iters):
        P = softmax_cols(B @ Ftr.T); G = (P-Str) @ Ftr/len(fit_idx); G[:,1:] += lam*B[:,1:]; B -= lr*G
    return softmax_cols(B @ Fte.T)

WALLS = [1960, 1968, 1976, 1984]; H = 8
def ws(w): return TR[years[TR] < w], TR[(years[TR] >= w) & (years[TR] < w+H)]
print("— per family: sky alone on walls, and best shrink lam onto the trend base:", flush=True)
print(f"  {'family':<12}{'alone':>8}{'lam':>6}{'trend+lam*sky':>15}   (trend alone {np.mean([kl(m_trend(*ws(w)), S[:, ws(w)[1]]) for w in WALLS]):.4f})", flush=True)
choice = {}
for fam, fn in FAM.items():
    alone = []; per_lam = {l: [] for l in (0.0, 0.25, 0.5, 1.0)}
    for w in WALLS:
        fit, jud = ws(w); base = m_trend(fit, jud); clim = m_clim(fit, jud); Pk = m_sky(fit, jud, fn)
        alone.append(kl(Pk, S[:, jud]))
        for l in per_lam:
            P = softmax_cols(np.log(np.clip(base,EPS,None)) + l*(np.log(np.clip(Pk,EPS,None)) - np.log(np.clip(clim,EPS,None))))
            per_lam[l].append(kl(P, S[:, jud]))
    m = {l: float(np.mean(v)) for l, v in per_lam.items()}; lb = min(m, key=m.get)
    choice[fam] = lb
    print(f"  {fam:<12}{np.mean(alone):>8.4f}{lb:>6}{m[lb]:>15.4f}", flush=True)
print("\n— held-out 1992-2025, with each family's wall-chosen lam:", flush=True)
base = m_trend(TR, TE); clim = m_clim(TR, TE); out = {"trend (base)": kl(base, S[:, TE]), "climatology": kl(clim, S[:, TE])}
Lall = np.log(np.clip(base, EPS, None)); any_on = False
for fam, fn in FAM.items():
    Pk = m_sky(TR, TE, fn); out[f"{fam} alone"] = kl(Pk, S[:, TE])
    if choice[fam] > 0:
        any_on = True; Lall = Lall + choice[fam]*(np.log(np.clip(Pk,EPS,None)) - np.log(np.clip(clim,EPS,None)))
        out[f"trend + {choice[fam]}*{fam}"] = kl(softmax_cols(np.log(np.clip(base,EPS,None)) + choice[fam]*(np.log(np.clip(Pk,EPS,None))-np.log(np.clip(clim,EPS,None)))), S[:, TE])
out["trend + all wall-chosen families"] = kl(softmax_cols(Lall), S[:, TE]) if any_on else out["trend (base)"]
for k, v in out.items(): print(f"  {k:<40} {v:.4f}", flush=True)
json.dump({"walls_lam": choice, "held_out": {k: round(v,4) for k,v in out.items()}}, open(os.path.expanduser("~/.artaquest-dev/artacomp/scidist3/pie_engines.json"),"w"), indent=1)
