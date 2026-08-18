#!/usr/bin/env python3
"""THE PHASOR MODEL OF RECORD on the DAILY series, one fit per category (operator 2026-08-18).

    y(t) = | b + SUM_i a_i * exp( i * (theta_i(t) - p_i) ) |^2

fitted to each category's daily submission count on the sqrt scale (so the model's amplitude IS
the envelope), in closed form via the exact expansion of the square into
  1 · cos/sin theta_i (transits) · cos/sin(theta_i - theta_k) (aspects)   — 1 + 2B + B(B-1) features
followed by the exact projection back onto b, a_i, p_i (the campaign's stage-2 split, np.roots).
Bodies: the daily PyJHora sky — Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Rahu (sidereal).
Because this is daily, the fast bodies finally move: the Moon completes 13 cycles a year, Mercury
and Venus their own — the phasor can hear them.

Per category, independently: fit on the first 80% of its reliable days, forecast the last 20% from
the sky alone, and score the forecast as a peak detector — AUC of the forecast level against the
same "peak within 7 days" labels used before. Also the forecast's own change (rise) as the score,
and the classical variant with only the slow bodies, and a Sun-only phasor (the season) as control.
Averaged over categories at the end.

  python3 analysis/arxivtopics/competition/daily_phasor.py
"""
import os, sys, json, datetime as dt
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
D = os.path.expanduser("~/.artaquest-dev/artacomp/daily")
daily = pd.read_csv(f"{D}/daily.csv", parse_dates=["date"]).set_index("date")
rel = pd.read_csv(f"{D}/reliable_from.csv", parse_dates=["reliable_from"]).set_index("category")
DAYS = np.array([d.date() for d in daily.index.to_pydatetime()])
E = np.load(f"{D}/ephemeris_ker_1991_2026.npz"); e0 = dt.date.fromisoformat(str(E["d0"])); EB = list(E["bodies"])
off = np.array([(d - e0).days for d in DAYS]); valid = (off >= 0) & (off < E["lon"].shape[0])
SEL = [EB.index(b) for b in ("sun","moon","mercury","venus","mars","jupiter","saturn","true_node")]
LON = np.zeros((len(DAYS), 8)); LON[valid] = E["lon"][off[valid]][:, SEL]
TH = np.deg2rad(LON)
BOD = ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","TrueNode"]   # kerykeion, sidereal Lahiri
TAU = 2*np.pi

def design(th):
    """The exact expansion of |b + sum a_i e^{i(theta_i - p_i)}|^2 : constant, transits, aspects."""
    B = th.shape[1]; C = [np.ones(len(th))]
    for i in range(B): C += [np.cos(th[:,i]), np.sin(th[:,i])]
    for i in range(B):
        for k in range(i+1, B): d = th[:,i]-th[:,k]; C += [np.cos(d), np.sin(d)]
    return np.stack(C, 1)
def fit_phasor(th_tr, y_tr, th_all, ridge=1e-3, w=None):
    """Closed-form ridge on the sqrt scale, then exact projection to (b, a_i, p_i); returns forecast + params."""
    X = design(th_tr); Xa = design(th_all); s = np.sqrt(np.maximum(y_tr, 0))
    W = np.ones(len(s)) if w is None else w
    R = np.eye(X.shape[1]); R[0,0] = 0
    c = np.linalg.solve(X.T @ (X*W[:,None]) + ridge*R, X.T @ (W*s))
    B = th_tr.shape[1]
    alpha, beta = c[1:1+2*B:2], c[2:2+2*B:2]
    p = np.arctan2(beta, alpha); M = np.sqrt(alpha**2 + beta**2)          # M_i = 2 b a_i
    C0 = max(c[0], 1e-9)                                                  # b^2 + sum a_i^2
    # exact split: b^2 solves b^4 - C0 b^2 + sum(M_i^2)/4 = 0
    disc = C0**2 - (M**2).sum()
    if disc < 0: M = M*np.sqrt(0.5)*C0/np.sqrt((M**2).sum()); disc = 0.0
    b2 = (C0 + np.sqrt(disc))/2; b = np.sqrt(max(b2, 1e-12)); a = M/(2*b)
    z = b + (a[None,:]*np.exp(1j*(th_all - p[None,:]))).sum(1)
    return np.abs(z)**2, dict(b=float(b), a=a.tolist(), p=np.rad2deg(p).tolist())

def peaks_labels(x):
    s = pd.Series(x).rolling(7, center=True, min_periods=1).mean().to_numpy()
    med = pd.Series(s).rolling(90, min_periods=30).median().shift(1).to_numpy()
    sd = pd.Series(s - np.nan_to_num(med, nan=s.mean())).rolling(90, min_periods=30).std().shift(1).to_numpy()
    thr = np.nan_to_num(med, nan=np.inf) + np.maximum(0.2*np.nan_to_num(med, nan=0), np.nan_to_num(sd, nan=np.inf))
    n = len(s); peak = np.zeros(n, bool)
    for i in range(3, n-3):
        if s[i] >= s[i-3:i+4].max() and s[i] > thr[i]: peak[i] = True
    lab = np.zeros(n, bool)
    for i in np.where(peak)[0]: lab[max(0,i-7):i] = True
    return lab.astype(int), peak

def auc(y, s):
    return roc_auc_score(y, s) if len(set(y)) > 1 else np.nan
rows = []; params = {}
SLOW = [4,5,6,7]           # Mars, Jupiter, Saturn, Rahu
for cat in daily.columns:
    if cat not in rel.index: continue
    start = max(np.searchsorted(DAYS, rel.loc[cat,"reliable_from"].date()), int(np.argmax(valid)))
    x = daily[cat].to_numpy(float)[start:]
    if len(x) < 8*365: continue
    idx = np.arange(start, start+len(x)); cut = int(len(x)*0.8); te = np.arange(cut, len(x))
    y, _ = peaks_labels(x)
    # detrend the level the phasor cannot carry: fit on the sqrt series minus its 365-day trailing mean? NO -
    # the model of record fits the raw sqrt level; we keep it faithful and let the anchored constant b hold the level.
    res = {"category": cat, "days": len(x), "test_pos": round(float(y[te].mean()),3)}
    # b carries the level: the trailing 365-day mean (known at t, causal), the arrows carry the timing.
    lvl = pd.Series(x).rolling(365, min_periods=60).mean().shift(1).bfill().to_numpy()
    xr = x / np.maximum(lvl, 1e-9)                       # ratio to own trailing level: ~1, peaks > 1
    for tag, bods in (("all8", list(range(8))), ("slow4", SLOW), ("sun", [0]), ("moon", [1]), ("fast4", [0,1,2,3])):
        th = TH[idx][:, bods]
        yhat, prm = fit_phasor(th[:cut], xr[:cut], th)   # phasor on the RATIO: |b + arrows|^2 ~ x/level
        rise = np.diff(yhat, prepend=yhat[0])
        res[f"auc_level_{tag}"] = auc(y[te], yhat[te]); res[f"auc_rise_{tag}"] = auc(y[te], rise[te])
        if tag == "all8": params[cat] = prm
    # the model of record forecasts x as level * phasor(ratio); score its sqrt-R2 on held-out
    yh8, _ = fit_phasor(TH[idx][:, :8][:cut], xr[:cut], TH[idx][:, :8]); fc = lvl * yh8
    s_te = np.sqrt(x[te]); res["r2_sqrt_all8"] = float(1 - ((s_te - np.sqrt(np.maximum(fc[te],0)))**2).sum()/((s_te - np.sqrt(lvl[te]))**2).sum())
    res["r2_note"] = "vs trailing-level baseline"
    rows.append(res)
    print(f"  {cat:<16} level: all8 {res['auc_level_all8']:.3f} slow4 {res['auc_level_slow4']:.3f} sun {res['auc_level_sun']:.3f} moon {res['auc_level_moon']:.3f} · rise all8 {res['auc_rise_all8']:.3f} · R2 {res['r2_sqrt_all8']:+.3f}", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{D}/daily_phasor_results.csv", index=False)
json.dump(params, open(f"{D}/daily_phasor_params.json","w"))
print(f"\n{len(df)} categories, the phasor model of record fitted independently on each, averaged:")
for k in [c for c in df.columns if c.startswith("auc_") or c == "r2_sqrt_all8"]:
    v = df[k].dropna(); print(f"  {k:<20} mean {v.mean():.4f} · median {v.median():.4f}")
json.dump({k: round(float(df[k].dropna().mean()),4) for k in df.columns if k.startswith("auc_") or k == "r2_sqrt_all8"}, open(f"{D}/daily_phasor_summary.json","w"), indent=1)
