#!/usr/bin/env python3
"""INDEPENDENT per-category trending classification on DAILY arXiv series, by peak detection.

Each category is its own experiment — its own series, its own peaks, its own model, its own
temporal split (last 20% of its reliable days), its own AUC. Nothing pooled across categories,
nothing shared. At the end the AUCs are averaged.

  series     daily submissions, smoothed with a centred 7-day mean (weekly rhythm removed)
  peaks      local maxima of the smoothed series that exceed the trailing 90-day median by
             >= max(20%, 1 sd of the trailing 90-day residual) — a "trending" event
  label(t)   1 if a peak occurs within the next 7 days of t   (balanced-ish, stated per category)
  features   the SKY on day t from PyJHora: Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn,
             Rahu (sidereal Lahiri) as sin/cos + first 3 harmonics; the Moon's nakshatra & tithi;
             pair separations of the four fastest bodies (the classical daily transits) — plus
             transit-to-natal for each body against the category's birth day (its first reliable
             day). Nothing about the series' own past enters the model.
  model      ridge logistic; C chosen on an inner temporal split inside train
  controls   (a) day-of-week and day-of-year cyclic features ONLY — the calendar; (b) the
             series' own trailing 7/30-day means — memory, reported for scale, never as astrology.

  python3 analysis/arxivtopics/competition/daily_trend.py
"""
import os, sys, json, csv, datetime as dt
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
D = os.path.expanduser("~/.artaquest-dev/artacomp/daily")
daily = pd.read_csv(f"{D}/daily.csv", parse_dates=["date"]).set_index("date")
rel = pd.read_csv(f"{D}/reliable_from.csv", parse_dates=["reliable_from"]).set_index("category")
days = daily.index.to_pydatetime(); DAYS = np.array([d.date() for d in days])
TAU = 2*np.pi

# ── the daily sky, once, for the whole span (PyJHora, sidereal Lahiri) ──────────────────────
# the daily sky from KERYKEION / Swiss Ephemeris (sidereal Lahiri), one engine, bodies NAMED — replaces the
# PyJHora table after a cross-check showed PyJHora's index order was not Sun,Moon,Mars,Mercury,...: indices
# 0-8 are Sun,Moon,Mercury,Venus,Sun(dup),Jupiter,Saturn,Uranus,Neptune. Mars and Rahu were never in it.
E = np.load(f"{D}/ephemeris_ker_1991_2026.npz"); e0 = dt.date.fromisoformat(str(E["d0"])); EB = list(E["bodies"])
off = np.array([(d - e0).days for d in DAYS]); valid_eph = (off >= 0) & (off < E["lon"].shape[0])
SEL = [EB.index(b) for b in ("sun","moon","mercury","venus","mars","jupiter","saturn","true_node")]
LON = np.zeros((len(DAYS), 8)); LON[valid_eph] = E["lon"][off[valid_eph]][:, SEL]
NAK = np.floor(LON[:,1]/(360/27)); TITHI = np.floor(((LON[:,1]-LON[:,0]) % 360)/12)      # tithi = 12° Moon-Sun steps
L = np.deg2rad(LON)
BOD = ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Rahu"]
FAST = [0,1,2,3,4]

def sky_feats(natal_idx):
    C = []
    for h in (1,2,3):
        for i in range(8): C += [np.cos(h*L[:,i]), np.sin(h*L[:,i])]
    for i in FAST:
        for k in FAST:
            if i < k: d = L[:,i]-L[:,k]; C += [np.cos(d), np.sin(d)]
    C += [np.cos(TAU*TITHI/30), np.sin(TAU*TITHI/30), np.cos(TAU*NAK/27), np.sin(TAU*NAK/27)]
    nat = L[natal_idx]
    for i in range(8):
        d = L[:,i]-nat[i]; C += [np.cos(d), np.sin(d), np.cos(2*d)]
    return np.stack(C, 1)
def cal_feats():
    doy = np.array([d.timetuple().tm_yday for d in DAYS], float); dow = np.array([d.weekday() for d in DAYS], float)
    return np.stack([np.cos(TAU*doy/365.25), np.sin(TAU*doy/365.25), np.cos(2*TAU*doy/365.25), np.sin(2*TAU*doy/365.25),
                     np.cos(TAU*dow/7), np.sin(TAU*dow/7)], 1)
CAL = cal_feats()

def peaks_and_labels(x):
    """x: daily counts. returns smoothed series, peak mask, label(t)=peak within next 7 days."""
    s = pd.Series(x).rolling(7, center=True, min_periods=1).mean().to_numpy()
    med = pd.Series(s).rolling(90, min_periods=30).median().shift(1).to_numpy()
    resid_sd = pd.Series(s - np.nan_to_num(med, nan=s.mean())).rolling(90, min_periods=30).std().shift(1).to_numpy()
    thr = np.nan_to_num(med, nan=np.inf) + np.maximum(0.2*np.nan_to_num(med, nan=0), np.nan_to_num(resid_sd, nan=np.inf))
    n = len(s); peak = np.zeros(n, bool)
    for i in range(3, n-3):
        if s[i] >= s[i-3:i+4].max() and s[i] > thr[i]: peak[i] = True
    lab = np.zeros(n, bool)
    for i in np.where(peak)[0]:
        lab[max(0, i-7):i] = True
    return s, peak, lab

def fit_auc(X, y, tr, te, Cs=(0.003, 0.03, 0.3)):
    """C chosen on the last 25% of the train span (temporal), then refit on all train, scored on test."""
    if len(set(y[te])) < 2 or len(set(y[tr])) < 2: return np.nan, None
    k = int(len(tr)*0.75); a, b = tr[:k], tr[k:]
    best = None
    for C in Cs:
        if len(set(y[b])) < 2: best = (0, C); break
        sc = StandardScaler().fit(X[a]); m = LogisticRegression(max_iter=2000, C=C).fit(sc.transform(X[a]), y[a])
        s = roc_auc_score(y[b], m.decision_function(sc.transform(X[b])))
        if best is None or s > best[0]: best = (s, C)
    sc = StandardScaler().fit(X[tr]); m = LogisticRegression(max_iter=2000, C=best[1]).fit(sc.transform(X[tr]), y[tr])
    return roc_auc_score(y[te], m.decision_function(sc.transform(X[te]))), best[1]

rows = []
for cat in daily.columns:
    if cat not in rel.index: continue
    start = max(np.searchsorted(DAYS, rel.loc[cat, "reliable_from"].date()), int(np.argmax(valid_eph)))
    x = daily[cat].to_numpy(float)[start:]
    if len(x) < 8*365: continue
    s, peak, lab = peaks_and_labels(x)
    idx = np.arange(start, start+len(x)); cut = int(len(x)*0.8)
    tr, te = np.arange(cut), np.arange(cut, len(x))
    y = lab.astype(int)
    Xs = sky_feats(start)[idx]; Xc = CAL[idx]
    mem = np.stack([pd.Series(x).rolling(7, min_periods=1).mean().shift(1).to_numpy(),
                    pd.Series(x).rolling(30, min_periods=1).mean().shift(1).to_numpy()], 1); mem = np.nan_to_num(mem)
    a_sky, C1 = fit_auc(Xs, y, tr, te); a_cal, _ = fit_auc(Xc, y, tr, te); a_mem, _ = fit_auc(mem, y, tr, te)
    a_skycal, _ = fit_auc(np.hstack([Xs, Xc]), y, tr, te)
    rows.append(dict(category=cat, days=len(x), peaks=int(peak.sum()), pos_rate=round(float(y.mean()),3),
                     test_pos=round(float(y[te].mean()),3), auc_sky=a_sky, auc_calendar=a_cal, auc_sky_plus_cal=a_skycal, auc_memory=a_mem, C=C1))
    print(f"  {cat:<18} days {len(x):>6} peaks {int(peak.sum()):>4} pos {y.mean():.2f} · sky {a_sky:.3f} · calendar {a_cal:.3f} · sky+cal {a_skycal:.3f} · memory {a_mem:.3f}", flush=True)
df = pd.DataFrame(rows).sort_values("auc_sky", ascending=False)
df.to_csv(f"{D}/daily_trend_results.csv", index=False)
print(f"\n{len(df)} categories, each its own model and its own test span (last 20% of its reliable days):")
for k in ("auc_sky","auc_calendar","auc_sky_plus_cal","auc_memory"):
    v = df[k].dropna(); print(f"  {k:<18} mean {v.mean():.4f} · median {v.median():.4f} · > 0.5 in {(v>0.5).mean()*100:.0f}% of categories")
json.dump({"n": int(len(df)), **{k: round(float(df[k].dropna().mean()),4) for k in ("auc_sky","auc_calendar","auc_sky_plus_cal","auc_memory")}},
          open(f"{D}/daily_trend_summary.json","w"), indent=1)
