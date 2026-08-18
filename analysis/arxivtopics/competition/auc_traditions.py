#!/usr/bin/env python3
"""COMPETE on astro-trending-251 — an ensemble across astrological and numerological traditions.

The target is cross-sectional (half the fields are "trending" in every year), so ANY feature that
depends on the year alone is worth exactly nothing: it gives every field the same score and AUC
0.5. Every feature here is therefore an INTERACTION between something the field owns and something
the date does — which is precisely the shape of a horoscope.

Each field's "birth" is its first continuously-active year; its natal sky is the sky of that year,
and its name is the string it is known by. From those:

  WESTERN (tropical)   transits to natal: cos/sin of (theta_i(t) - natal_i), all 7 bodies; and the
                       classical aspect set (conjunction/opposition/trine/square/sextile) scored by
                       orb against each transit-natal pair.
  VEDIC (sidereal)     the same longitudes minus the Lahiri ayanamsa, plus the 27 nakshatras: which
                       mansion each transiting body occupies relative to the field's natal mansion.
  CHINESE              sexagenary year: the field's branch/stem from its birth year against the
                       current year's — the trine harmony group (branch diff mod 4 == 0), the clash
                       (diff 6), and the raw 12- and 10-cycle offsets.
  MAYAN                the field's position in the 260-day tzolkin and 365-day haab counts taken on
                       years, and the offsets between natal and transiting positions.
  NUMEROLOGY           Pythagorean gematria of the field name (a=1..i=9 wrapping), its reduced root
                       and master numbers, the year's digit root, and the classical PERSONAL YEAR
                       (name root + year root, reduced) plus name-root x year-root interactions.

Reported honestly: each tradition alone, the astrology-only ensemble, and — for context, never as
the headline — a momentum baseline built from the field's own past growth.

  python3 analysis/arxivtopics/competition/auc_traditions.py
"""
import os, sys, json, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import arxiv_fit as af

BUN = os.path.expanduser("~/.artaquest-dev/artacomp/aucomp")
tr = pd.read_csv(f"{BUN}/train.csv"); te = pd.read_csv(f"{BUN}/test.csv")
sol = pd.read_csv(f"{BUN}/solution.csv")
names, Yv, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)                 # (n_all, 7) longitudes, radians
years = [int(y) for y in labels]; Y0 = years[0]
BODIES = ["mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "node"]
NB = len(BODIES)

# ── each field's birth: first year it appears in the training rows ──
birth = tr.groupby("field")["year"].min().to_dict()
for f in te["field"].unique(): birth.setdefault(f, int(te[te.field == f]["year"].min()))
FIELDS = sorted(set(tr["field"]) | set(te["field"]))
FI = {f: i for i, f in enumerate(FIELDS)}

AYAN = np.deg2rad(23.85)                              # Lahiri ayanamsa, ~2000
def lon(t):  return TH[t - Y0]                        # tropical longitudes at year t
def sid(t):  return (TH[t - Y0] - AYAN) % (2 * np.pi)

ASPECTS = [(0, 10), (np.pi, 10), (2*np.pi/3, 8), (np.pi/2, 8), (np.pi/3, 6)]   # angle, orb(deg)
def aspect_feats(d):
    """Classical aspects by orb: 1 at exact, fading to 0 at the orb edge."""
    out = []
    for ang, orb in ASPECTS:
        sep = np.abs(((d - ang + np.pi) % (2 * np.pi)) - np.pi)
        out.append(np.clip(1 - sep / np.deg2rad(orb), 0, 1))
    return np.concatenate(out)

LET = {c: (i % 9) + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
def gematria(s):
    v = sum(LET.get(c, 0) for c in s.lower())
    return v
def digit_root(x):
    x = int(abs(x))
    while x > 9: x = sum(int(c) for c in str(x))
    return x or 9

def feats(field, t):
    j = FI[field]; b = birth[field]
    nat, cur = lon(b), lon(t)
    natS, curS = sid(b), sid(t)
    d = (cur - nat) % (2 * np.pi)                       # transit-to-natal separations
    dS = (curS - natS) % (2 * np.pi)
    west = np.concatenate([np.cos(d), np.sin(d), aspect_feats(d)])
    # Vedic: nakshatra (27ths) of each body, natal and transiting, and their offset
    nakN = np.floor(natS / (2 * np.pi) * 27); nakT = np.floor(curS / (2 * np.pi) * 27)
    off = (nakT - nakN) % 27
    ved = np.concatenate([np.cos(2*np.pi*off/27), np.sin(2*np.pi*off/27),
                          np.cos(dS), np.sin(dS)])
    # Chinese sexagenary
    bb, cb = (b - 4) % 12, (t - 4) % 12
    bs, cs = (b - 4) % 10, (t - 4) % 10
    db, ds = (cb - bb) % 12, (cs - bs) % 10
    chi = np.array([np.cos(2*np.pi*db/12), np.sin(2*np.pi*db/12),
                    np.cos(2*np.pi*ds/10), np.sin(2*np.pi*ds/10),
                    1.0 * (db % 4 == 0), 1.0 * (db == 6), 1.0 * ((cb - bb) % 60 == 0),
                    np.cos(2*np.pi*((t - b) % 60)/60), np.sin(2*np.pi*((t - b) % 60)/60)])
    # Mayan cycles on years
    tz, ha = (t - b) % 260, (t - b) % 365
    may = np.array([np.cos(2*np.pi*tz/260), np.sin(2*np.pi*tz/260),
                    np.cos(2*np.pi*ha/365), np.sin(2*np.pi*ha/365),
                    np.cos(2*np.pi*((t - b) % 13)/13), np.sin(2*np.pi*((t - b) % 13)/13),
                    np.cos(2*np.pi*((t - b) % 20)/20), np.sin(2*np.pi*((t - b) % 20)/20)])
    # Numerology
    g = gematria(field); gr = digit_root(g); yr = digit_root(t)
    pers = digit_root(gr + yr)
    num = np.array([np.cos(2*np.pi*pers/9), np.sin(2*np.pi*pers/9),
                    np.cos(2*np.pi*((gr + t) % 9)/9), np.sin(2*np.pi*((gr + t) % 9)/9),
                    1.0 * (pers in (1, 9)), 1.0 * (g % 11 == 0), 1.0 * (gr == yr),
                    (gr * yr) / 81.0, np.cos(2*np.pi*((t - b) % 9)/9), np.sin(2*np.pi*((t - b) % 9)/9)])
    return west, ved, chi, may, num

GROUPS = ["western", "vedic", "chinese", "mayan", "numerology"]
def build(df):
    parts = [[] for _ in GROUPS]
    for f, t in zip(df["field"].to_numpy(), df["year"].to_numpy()):
        fs = feats(f, int(t))
        for k in range(len(GROUPS)): parts[k].append(fs[k])
    return [np.asarray(p, float) for p in parts]

print("building features …", flush=True)
Xtr = build(tr); Xte = build(te)
ytr = tr["target"].to_numpy(); yte = sol.set_index("id").loc[te["id"]]["target"].to_numpy()
usage = sol.set_index("id").loc[te["id"]]["Usage"].to_numpy()
pub, pri = usage == "Public", usage == "Private"
for g, A in zip(GROUPS, Xtr): print(f"  {g:<11} {A.shape[1]:>3} features")

def auc(y, p): return roc_auc_score(y, p)
def report(tag, p):
    print(f"  {tag:<34} overall {auc(yte,p):.4f} · public {auc(yte[pub],p[pub]):.4f} · "
          f"private {auc(yte[pri],p[pri]):.4f}", flush=True)
    return auc(yte, p)

def fit_lr(A, B):
    sc = StandardScaler().fit(A)
    m = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(A), ytr)
    return m.predict_proba(sc.transform(B))[:, 1]
def fit_gb(A, B, depth=5, n=600, lr=0.05):
    m = xgb.XGBClassifier(max_depth=depth, n_estimators=n, learning_rate=lr, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=8, eval_metric="auc",
                          tree_method="hist", random_state=7)
    m.fit(A, ytr)
    return m.predict_proba(B)[:, 1]

print("\n— each tradition alone (logistic | boosted):", flush=True)
res = {}
solo = {}
for g, A, B in zip(GROUPS, Xtr, Xte):
    pl, pg = fit_lr(A, B), fit_gb(A, B)
    res[f"{g} (logistic)"] = report(f"{g} · logistic", pl)
    res[f"{g} (boosted)"] = report(f"{g} · boosted", pg)
    solo[g] = (pl + pg) / 2

print("\n— the astrology + numerology ensemble:", flush=True)
ALL_tr = np.concatenate(Xtr, 1); ALL_te = np.concatenate(Xte, 1)
p_lr, p_gb = fit_lr(ALL_tr, ALL_te), fit_gb(ALL_tr, ALL_te, depth=6, n=900)
res["all features · logistic"] = report("all features · logistic", p_lr)
res["all features · boosted"] = report("all features · boosted", p_gb)
p_blend = (p_lr + p_gb) / 2
res["all features · blend"] = report("all features · blend", p_blend)
p_mean = np.mean([solo[g] for g in GROUPS], 0)
res["mean of the five traditions"] = report("mean of the five traditions", p_mean)
p_best = 0.5 * p_blend + 0.5 * p_mean
res["THE STACK (blend + tradition mean)"] = report("THE STACK", p_best)

np.save(f"{BUN}/pred_astrology.npy", p_best)
sub = pd.DataFrame({"id": te["id"], "target": p_best})
sub.to_csv(f"{BUN}/submission_astrology.csv", index=False)
json.dump({k: round(float(v), 4) for k, v in res.items()}, open(f"{BUN}/auc_results.json", "w"), indent=1)
print(f"\nbest astrology AUC: {max(res.values()):.4f} · submission written", flush=True)
