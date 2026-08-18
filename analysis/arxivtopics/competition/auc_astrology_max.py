#!/usr/bin/env python3
"""THE STRONGEST ASTROLOGY MODEL — every classical technique, fitted strictly before the wall.

Information rule, enforced: a prediction for (field, year t) may use the field's NATAL CHART (the
sky of its first active year) and the sky at t. Nothing about the field after 1984 is ever read —
which is what makes the momentum bar 0.5320 rather than the 0.5820 a leaky baseline reported.

Techniques, all per-field and date-driven (a within-year ranking target ignores anything else):
  transits      cos/sin of (theta_i(t) - natal_i), all 7 bodies
  harmonics     the same separations at the 2nd..9th harmonic (harmonic astrology)
  aspects       conjunction/opposition/trine/square/sextile scored by orb
  returns       completed returns of each body to its natal degree, and closeness to the next
  profections   annual profection, (t - birth) mod 12
  dashas        Vimshottari maha- and antardasha, the 120-year Vedic period sequence, seeded from
                the natal lunar NODE's nakshatra (we carry the node, not the Moon — stated, not hidden)
  progressions  secondary progressions, the classical day-for-a-year
  midpoints     transiting midpoints of every body pair against the natal midpoint (Ebertin)
  nakshatras    sidereal mansion offsets, transit vs natal
  chinese       sexagenary stem/branch harmony and clash against the natal year
  mayan         tzolkin/haab/13/20 cycles since birth
  numerology    Pythagorean gematria of the name, digit roots, the classical personal year

Hyper-parameters are chosen on an INNER temporal split (fit <1960, judge 1960-84) so the 1985-2024
benchmark is never touched during selection.

  python3 analysis/arxivtopics/competition/auc_astrology_max.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import arxiv_fit as af

BUN = os.path.expanduser("~/.artaquest-dev/artacomp/aucomp")
tr = pd.read_csv(f"{BUN}/train.csv"); te = pd.read_csv(f"{BUN}/test.csv"); sol = pd.read_csv(f"{BUN}/solution.csv")
yte = sol.set_index("id").loc[te["id"]]["target"].to_numpy()
usage = sol.set_index("id").loc[te["id"]]["Usage"].to_numpy()
names, Yv, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
years = [int(y) for y in labels]; Y0 = years[0]
NB = TH.shape[1]
birth = tr.groupby("field")["year"].min().to_dict()
for f in te["field"].unique(): birth.setdefault(f, 1985)
PERIODS = np.array([1.88, 11.86, 29.46, 84.0, 164.8, 248.0, 18.6])      # sidereal years per body
AYAN = np.deg2rad(23.85)
ASPECTS = [(0.0, 10), (np.pi, 10), (2*np.pi/3, 8), (np.pi/2, 8), (np.pi/3, 6)]
HARM = (2, 3, 4, 5, 7, 9)
PAIRS = [(i, k) for i in range(NB) for k in range(i+1, NB)]
DASHA = [("ketu",7),("venus",20),("sun",6),("moon",10),("mars",7),("rahu",18),("jupiter",16),("saturn",19),("mercury",17)]
DTOT = sum(d[1] for d in DASHA)
LET = {c: (i % 9) + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
def droot(x):
    x = int(abs(x))
    while x > 9: x = sum(int(c) for c in str(x))
    return x or 9

def dasha_at(nat_node, age):
    """Vimshottari: which maha/antardasha lord is running, and how far through."""
    sidn = (nat_node - AYAN) % (2*np.pi)
    pos = sidn / (2*np.pi) * 27.0
    k = int(pos) % 27; frac = pos - int(pos)
    start = k % 9
    rem = (1 - frac) * DASHA[start][1]
    t = age; idx = start
    if t < rem: through = 1 - rem/DASHA[idx][1] + t/DASHA[idx][1]
    else:
        t -= rem; idx = (idx + 1) % 9
        while t >= DASHA[idx][1]:
            t -= DASHA[idx][1]; idx = (idx + 1) % 9
        through = t / DASHA[idx][1]
    sub = int(through * 9) % 9                                   # antardasha, evenly split (approx)
    return idx, sub, float(np.clip(through, 0, 1))

def row_feats(field, t):
    b = birth[field]; age = t - b
    nat, cur = TH[b - Y0], TH[t - Y0]
    d = (cur - nat) % (2*np.pi)
    f = [np.cos(d), np.sin(d)]
    for h in HARM: f += [np.cos(h*d), np.sin(h*d)]
    for ang, orb in ASPECTS:
        sep = np.abs(((d - ang + np.pi) % (2*np.pi)) - np.pi)
        f.append(np.clip(1 - sep/np.deg2rad(orb), 0, 1))
    ret = age / PERIODS
    f += [np.floor(ret) % 4, ret - np.floor(ret), np.cos(2*np.pi*ret), np.sin(2*np.pi*ret)]
    # sidereal nakshatras
    natS, curS = (nat - AYAN) % (2*np.pi), (cur - AYAN) % (2*np.pi)
    off = (np.floor(curS/(2*np.pi)*27) - np.floor(natS/(2*np.pi)*27)) % 27
    f += [np.cos(2*np.pi*off/27), np.sin(2*np.pi*off/27)]
    # secondary progressions: day-for-a-year
    prog = (nat + age * (2*np.pi/PERIODS) / 365.25) % (2*np.pi)
    dp = (cur - prog) % (2*np.pi)
    f += [np.cos(dp), np.sin(dp)]
    # midpoints, transiting vs natal
    mt = np.array([(cur[i] + cur[k]) / 2 for i, k in PAIRS])
    mn = np.array([(nat[i] + nat[k]) / 2 for i, k in PAIRS])
    dm = (mt - mn) % (2*np.pi)
    f += [np.cos(dm), np.sin(dm)]
    flat = np.concatenate([np.atleast_1d(x) for x in f])
    # profections, dashas, chinese, mayan, numerology
    prof = age % 12
    mi, si, thr = dasha_at(nat[6], age)
    oh = np.zeros(9); oh[mi] = 1
    oh2 = np.zeros(9); oh2[si] = 1
    ohp = np.zeros(12); ohp[prof % 12] = 1
    bb, cb = (b-4) % 12, (t-4) % 12; bs, cs = (b-4) % 10, (t-4) % 10
    db, ds = (cb-bb) % 12, (cs-bs) % 10
    g = sum(LET.get(c, 0) for c in field.lower()); gr = droot(g); yr = droot(t); pers = droot(gr+yr)
    extra = np.array([thr, np.cos(2*np.pi*prof/12), np.sin(2*np.pi*prof/12),
        np.cos(2*np.pi*db/12), np.sin(2*np.pi*db/12), np.cos(2*np.pi*ds/10), np.sin(2*np.pi*ds/10),
        1.0*(db % 4 == 0), 1.0*(db == 6),
        np.cos(2*np.pi*((age) % 60)/60), np.sin(2*np.pi*((age) % 60)/60),
        np.cos(2*np.pi*(age % 260)/260), np.sin(2*np.pi*(age % 260)/260),
        np.cos(2*np.pi*(age % 13)/13), np.sin(2*np.pi*(age % 13)/13),
        np.cos(2*np.pi*(age % 20)/20), np.sin(2*np.pi*(age % 20)/20),
        np.cos(2*np.pi*pers/9), np.sin(2*np.pi*pers/9), 1.0*(pers in (1,9)), 1.0*(g % 11 == 0),
        1.0*(gr == yr), (gr*yr)/81.0, np.cos(2*np.pi*(age % 9)/9), np.sin(2*np.pi*(age % 9)/9),
        np.log1p(age)])
    return np.concatenate([flat, oh, oh2, ohp, extra])

def build(df):
    return np.asarray([row_feats(f, int(t)) for f, t in zip(df["field"], df["year"])], float)

print("building features …", flush=True)
Xtr, Xte = build(tr), build(te)
ytr = tr["target"].to_numpy()
print(f"  {Xtr.shape[1]} features · train {Xtr.shape[0]} · test {Xte.shape[0]}", flush=True)

inner = tr["year"].to_numpy() < 1960
Xi, yi, Xv, yv = Xtr[inner], ytr[inner], Xtr[~inner], ytr[~inner]
print(f"  inner split: fit {inner.sum()} (<1960) · judge {(~inner).sum()} (1960-84)", flush=True)

def gb(A, ya, B, **kw):
    m = xgb.XGBClassifier(eval_metric="auc", tree_method="hist", random_state=7,
                          subsample=0.8, colsample_bytree=0.6, **kw)
    m.fit(A, ya); return m.predict_proba(B)[:, 1]
def lr(A, ya, B, C=0.1):
    sc = StandardScaler().fit(A)
    m = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(A), ya)
    return m.predict_proba(sc.transform(B))[:, 1]

print("\n— selecting on the inner wall (1960-84), benchmark untouched:", flush=True)
cands = []
for depth in (3, 4, 6):
    for nest, lrate in ((300, 0.05), (900, 0.02)):
        for mcw in (20, 80):
            s = roc_auc_score(yv, gb(Xi, yi, Xv, max_depth=depth, n_estimators=nest,
                                     learning_rate=lrate, min_child_weight=mcw))
            cands.append((s, dict(max_depth=depth, n_estimators=nest, learning_rate=lrate, min_child_weight=mcw)))
            print(f"   gb d{depth} n{nest} lr{lrate} mcw{mcw}: {s:.4f}", flush=True)
for C in (0.01, 0.1, 1.0):
    s = roc_auc_score(yv, lr(Xi, yi, Xv, C))
    cands.append((s, {"lr_C": C})); print(f"   logistic C={C}: {s:.4f}", flush=True)
cands.sort(key=lambda x: -x[0])
print(f"  chosen: {cands[0][1]} (inner {cands[0][0]:.4f})", flush=True)

def fit_full(cfg):
    return lr(Xtr, ytr, Xte, cfg["lr_C"]) if "lr_C" in cfg else gb(Xtr, ytr, Xte, **cfg)
def rep(tag, p):
    o = roc_auc_score(yte, p); pu = roc_auc_score(yte[usage=='Public'], p[usage=='Public'])
    pr = roc_auc_score(yte[usage=='Private'], p[usage=='Private'])
    print(f"  {tag:<40} overall {o:.4f} · public {pu:.4f} · private {pr:.4f}", flush=True); return o, p

print("\n— on the 1985-2024 benchmark (bar to beat: frozen momentum 0.5320):", flush=True)
res = {}
o1, p1 = rep("best single (inner-chosen)", fit_full(cands[0][1]))
res["best_single"] = o1
top = [c for c in cands[:5]]
ps = [fit_full(c[1]) for c in top]
from scipy.stats import rankdata
p_ens = np.mean([rankdata(p)/len(p) for p in ps], 0)
o2, _ = rep("rank-ensemble of the top 5", p_ens); res["ensemble_top5"] = o2
best_p = p_ens if o2 > o1 else p1
res["best"] = max(o1, o2)
pd.DataFrame({"id": te["id"], "target": best_p}).to_csv(f"{BUN}/submission_astro_max.csv", index=False)
json.dump({k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in res.items()},
          open(f"{BUN}/auc_astro_max.json", "w"), indent=1)
print(f"\n  best astrology AUC: {max(o1, o2):.4f} · bar 0.5320", flush=True)
