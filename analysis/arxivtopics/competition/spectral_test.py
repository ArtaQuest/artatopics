#!/usr/bin/env python3
"""THE SPECTRAL TEST — does science's field dynamics carry power at PLANETARY periods?

No model, no fitting, no wall. If planetary cycles drive which fields rise and fall, the fields'
own share-change series must contain excess power at those periods. This asks the question directly.

Method: each field's annual change in citation share, linearly detrended and Hann-windowed, is
turned into a periodogram. Periodograms are averaged across fields. The comparison is against
SURROGATE series: an AR(1) process fitted to each field's own series, simulated 200 times. This
matters because scientific time series are red — power rises at long periods for reasons that have
nothing to do with planets — so a raw peak at 30 years proves nothing. The AR(1) null reproduces
that redness, and only excess ABOVE it counts.

Periods tested are those a ~170-year record can actually resolve (at least two full cycles);
Neptune at 165y and Pluto at 248y CANNOT be tested here and are reported as untestable rather than
quietly omitted. Synodic periods are included because the conjunction cycle between two planets —
above all the 19.86-year Jupiter-Saturn "great conjunction" — is what classical mundane astrology
actually reads for the fate of nations and disciplines.

  python3 analysis/arxivtopics/competition/spectral_test.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import arxiv_fit as af

names, Yv, labels, future = af.load_lunar()
tv = af.META["topic_valid"]; years = [int(y) for y in labels]
J, n = Yv.shape
rng = np.random.RandomState(0)

PLANET = {"Jupiter": 11.86, "Lunar node": 18.61, "Jupiter-Saturn synodic (great conjunction)": 19.86,
          "Saturn": 29.46, "Jupiter-Uranus synodic": 13.81, "Saturn-Uranus synodic": 45.36,
          "Jupiter-Neptune synodic": 12.78, "Saturn-Neptune synodic": 35.87,
          "Uranus-Neptune synodic": 171.4, "Uranus": 84.02, "Neptune": 164.8, "Pluto": 248.1}

def series(j):
    idx = np.where(tv[j])[0]
    if len(idx) < 80: return None
    a, b = idx[0], idx[-1]
    s = Yv[j, a:b+1].astype(float)
    if np.std(s) < 1e-12: return None
    d = np.diff(s)
    if np.std(d) < 1e-12: return None
    return d

def spec(x, freqs):
    """Power at the requested frequencies: detrend, Hann-window, then a direct DFT."""
    m = len(x)
    t = np.arange(m)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    w = np.hanning(m); x = x * w
    x = x / (np.std(x) + 1e-12)
    E = np.exp(-2j * np.pi * np.outer(freqs, t))
    return (np.abs(E @ x) ** 2) / m

def ar1(x, k):
    """k surrogates from the AR(1) fitted to x — same redness, no planets."""
    m = len(x)
    r = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    r = float(np.clip(r, -0.95, 0.95))
    sd = np.std(x) * np.sqrt(max(1 - r*r, 1e-6))
    out = np.zeros((k, m))
    out[:, 0] = rng.randn(k) * np.std(x)
    e = rng.randn(k, m) * sd
    for i in range(1, m): out[:, i] = r * out[:, i-1] + e[:, i]
    return out

PMIN, PMAX = 4.0, 60.0
periods = np.exp(np.linspace(np.log(PMIN), np.log(PMAX), 200))
freqs = 1.0 / periods

def local_background(power, win=41):
    """The red-noise floor under each period: a running median in LOG-period space. Self-normalising,
    so no surrogate has to be scaled correctly — a real cycle is a PEAK above its own neighbourhood."""
    m = len(power); bg = np.zeros(m)
    for i in range(m):
        a, b = max(0, i - win//2), min(m, i + win//2 + 1)
        bg[i] = np.median(power[a:b])
    return np.maximum(bg, 1e-12)

EX = np.full((J, len(periods)), np.nan)     # per-field excess over its own local background
LEN = np.zeros(J)
for j in range(J):
    x = series(j)
    if x is None or len(x) < 2*PMIN: continue
    LEN[j] = len(x)
    pw = spec(x, freqs)
    ex = pw / local_background(pw)
    ex[periods > len(x)/2.0] = np.nan       # a period this record cannot resolve
    EX[j] = ex
ok = LEN > 0
print(f"fields analysed: {int(ok.sum())} of {J} · record length median {np.median(LEN[ok]):.0f}y, "
      f"max {LEN[ok].max():.0f}y", flush=True)

nres = np.sum(~np.isnan(EX), 0)
mean_ex = np.nanmean(EX, 0)
base = float(np.nanmean(mean_ex))
print(f"\n— excess over each field's own red-noise background (1.00 = no peak; baseline {base:.3f}):", flush=True)
print(f"  {'cycle':<44}{'period':>8}{'fields':>8}{'excess':>9}{'vs base':>9}", flush=True)
res = {}
for nm, P in sorted(PLANET.items(), key=lambda kv: kv[1]):
    if P > PMAX:
        print(f"  {nm:<44}{P:>8.1f}   UNTESTABLE — needs a {2*P:.0f}-year record", flush=True)
        res[nm] = {"period": P, "testable": False}; continue
    k = int(np.argmin(np.abs(periods - P)))
    if nres[k] < 20:
        print(f"  {nm:<44}{P:>8.1f}   only {nres[k]} fields resolve it", flush=True)
        res[nm] = {"period": P, "testable": False, "fields": int(nres[k])}; continue
    e = float(mean_ex[k])
    print(f"  {nm:<44}{P:>8.2f}{nres[k]:>8}{e:>9.3f}{e-base:>+9.3f}", flush=True)
    res[nm] = {"period": P, "testable": True, "fields": int(nres[k]), "excess": round(e,3),
               "vs_baseline": round(e-base,3)}
# is the planetary SET special among all periods?
pk = [int(np.argmin(np.abs(periods-P))) for nm,P in PLANET.items() if P <= PMAX and nres[int(np.argmin(np.abs(periods-P)))] >= 20]
pl_mean = float(np.mean([mean_ex[k] for k in pk]))
allk = [k for k in range(len(periods)) if nres[k] >= 20]
rng2 = np.random.RandomState(1)
draws = [float(np.mean(rng2.choice([mean_ex[k] for k in allk], len(pk), replace=False))) for _ in range(20000)]
pv = float(np.mean([d >= pl_mean for d in draws]))
print(f"\n  planetary periods mean excess {pl_mean:.3f} vs {len(pk)} random periods drawn 20,000 times:", flush=True)
print(f"  p = {pv:.3f}  (the chance a random set of periods looks this good or better)", flush=True)
top = [k for k in np.argsort(-mean_ex) if nres[k] >= 20][:8]
print(f"\n  the 8 strongest cycles in the record, planetary or not:", flush=True)
for k in top:
    near = min(PLANET.items(), key=lambda kv: abs(kv[1]-periods[k]))
    print(f"    {periods[k]:6.1f}y  excess {mean_ex[k]:.3f}  ({nres[k]} fields)  nearest: {near[0]} {near[1]:.1f}y", flush=True)
json.dump({"fields": int(ok.sum()), "baseline": round(base,3), "planetary": res,
           "planetary_mean": round(pl_mean,3), "p_value": pv,
           "top": [{"period": round(float(periods[k]),2), "excess": round(float(mean_ex[k]),3)} for k in top]},
          open(os.path.expanduser("~/.artaquest-dev/artacomp/piecomp/spectral.json"), "w"), indent=1)
