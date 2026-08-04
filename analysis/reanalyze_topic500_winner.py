#!/usr/bin/env python3
"""RE-ANALYSE every atlas topic with the TOPIC-500 BASELINE MODEL — the global-phase sinc, fit by
gradient descent (operator directive 2026-07-07, the FINAL simplification: no permutation; since
2026-07-10 the fit is PYTORCH — one batched, GPU-accelerated pass over every topic, mode="atlas":
the plain train-objective classification fit, best of the 12 sign-centre starts by train loss).

Each topic's SEASON and analysis are re-derived from the competition's ONE fixed model class, so the
/skills page shows EXACTLY the model the board is built on:

    y = SUM_i w_i * sinc( f_i * (x_i - p) )  +  b        (i over the 11 bodies)

  fit by 12 gradient starts (one at each 30 deg sign centre, keep the best — topic500_reference_solution.py).
  The fitted global PHASE p -> the topic's zodiac SIGN (christmas -> Scorpio, halloween -> Virgo,
  super-bowl -> Capricorn, swimming -> Gemini). The per-body FREQUENCIES f_i are reported per topic.

    season   = { sign, peak=p, month, freqs (11 per-body), r2 }
    fitted   = the model's fit over the full CLEAN series (2004-01 .. 2025-06)
    forecast = the model's 12-month forecast over the dropped recency year (2025-07 .. 2026-06)
    shares   = an 11-body variance decomposition (each body's Var(w_i*feat_i)/Var(y), summing to R^2) —
               keeps /cycles coherent; the Sun's share is the ANNUAL season, the slow outer bodies the trend.

The sidereal originals (sid_sign/sid_ref/…) are preserved; records are stamped class_model="sinc-gd".

  python3 analysis/reanalyze_topic500_winner.py        # updates analysis/_topics_weekly.json in place
"""
import importlib.util as _u, json, math, os, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)


def _load(path, name):
    s = _u.spec_from_file_location(name, path); m = _u.module_from_spec(s); s.loader.exec_module(m); return m


tf = _load(os.path.join(HERE, "trends_fit.py"), "tf")
R5 = _load(os.path.join(HERE, "topic500_reference_solution.py"), "r5")   # the global-phase sinc — the competition's model

SIGNS = list(tf.SIGNS)
BODIES = list(tf.BODIES)                                   # sun,mercury,…,node  (11, the HKEY order)
HKEY = [f"{b}:1" for b in BODIES]
NB = len(BODIES)
SUN_I = BODIES.index("sun")
PERIOD = {b: float(tf.PERIOD_YEARS[b]) for b in BODIES}
TREND_THRESH = 0.40                                       # annual_frac (detrended calendar-season strength) below this ⇒ trend-led
#   (no real annual season → the zodiac sign is indicative only). Above it → season-led (lead with the sign). christmas 0.99,
#   halloween 0.99, super-bowl 0.82, tax 0.92, swimming 0.69 → season; python 0.31, cybersecurity 0.12, ramadan 0.08 → trend.


def circ_dist(a, b):
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def season_month(peak, sun_by_month):
    """The CALENDAR month (0-11) whose mean sidereal Sun longitude is closest to the fitted phase — so the
       plain-English month never contradicts the sign (sidereal Gemini ≈ June, Virgo ≈ October)."""
    return int(min(range(12), key=lambda mm: circ_dist(sun_by_month[mm], peak)))


def trend_direction(y):
    """The net direction of the interest over the series — 'rising' / 'falling' / 'steady' — from the first
       vs last quarter's robust median, relative to the series' own scale."""
    y = np.asarray(y, float); nn = len(y); q = max(3, nn // 4)
    lo, hi = float(np.median(y[:q])), float(np.median(y[-q:]))
    rel = (hi - lo) / (float(np.median(y)) + 1e-9)
    return "rising" if rel > 0.20 else "falling" if rel < -0.20 else "steady"


def seasonality(y, moy):
    """The topic's CALENDAR-SEASON STRENGTH ∈ [0,1] — the fraction of its DETRENDED variance that is
       month-of-year periodic. Remove a centred 13-month rolling median (the slow trend), bin the residual
       anomaly by calendar month, and take the variance the 12 month-means explain over the anomaly's
       total. High for a real annual season (christmas 0.99, tax 0.92), low for a secular trend (cyber 0.12).
       Direct from the series (no fit), so it is the honest 'how seasonal is this topic' the split needs."""
    y = np.asarray(y, float); moy = np.asarray(moy, int)
    trend = pd.Series(y).rolling(13, center=True, min_periods=1).median().to_numpy()
    detr = y - trend
    sm = np.array([detr[moy == m].mean() if (moy == m).any() else 0.0 for m in range(12)])
    recon = sm[moy]
    sst = float(np.sum((detr - detr.mean()) ** 2))
    return float(np.sum((recon - detr.mean()) ** 2) / sst) if sst > 1e-9 else 0.0


def decompose(y, par, sky_m):
    """Per-body variance decomposition of the sinc-GD fit: flatten each body's feature to its MEAN (keeping
       the fitted weight) and recompute R^2; the DROP is that body's contribution v_b = Var(w_b*feat_b)/Var(y)
       ≥ 0. The 11 v_b are normalised to sum EXACTLY to R^2. Returns (shares{HKEY->v}, r2_full) — keeps
       /cycles coherent (each body's share of the explained variance)."""
    y = np.asarray(y, float); n = len(y)
    w = par[:NB]; f = par[NB:2 * NB]; p = par[2 * NB]; b = par[2 * NB + 1]
    feat = np.sinc(np.deg2rad(R5.wrap(sky_m - p)) * f[None, :])       # n×NB
    pred = feat @ w + b
    sst = float(np.sum((y - y.mean()) ** 2)) or 1.0
    r2_full = 1.0 - float(np.sum((y - pred) ** 2)) / sst
    v = np.empty(NB)
    for i in range(NB):
        pred_i = pred - w[i] * (feat[:, i] - feat[:, i].mean())        # body i flattened to its mean feature
        v[i] = r2_full - (1.0 - float(np.sum((y - pred_i) ** 2)) / sst)
    v = np.clip(v, 0.0, None); tot = float(v.sum())
    if tot <= 1e-12:                                                   # degenerate (flat / unfit) — tiny period-spread
        wr = np.array([PERIOD[bd] for bd in BODIES]); wr = wr / wr.sum()
        shares = {HKEY[i]: float(1e-4 * wr[i]) for i in range(NB)}
        return shares, max(0.0, r2_full)
    scale = max(0.0, r2_full) / tot                                   # renormalise so the shares sum to R^2
    shares = {HKEY[i]: float(v[i] * scale) for i in range(NB)}
    return shares, max(0.0, r2_full)


def main():
    t0 = time.time()
    path = "analysis/_topics_weekly.json"
    reg = json.load(open(path))
    lon = tf.ephemeris()
    sun = np.asarray(lon["sun"], float)
    sky = np.stack([np.round(np.asarray(lon[b], float)).astype(float) % 360 for b in BODIES], axis=1)  # n×11 float
    grid = pd.DatetimeIndex(tf.GRID)
    clean_end = len(grid) - 12                             # [0, clean_end) = the full clean series 2004-01…2025-06
    moy_clean = np.array([d.month - 1 for d in grid])[:clean_end]      # month-of-year (0-11) — for the seasonality strength
    # Circular-mean sidereal Sun longitude per calendar month — the wheel the fitted phase lives on, used to
    # name the peak MONTH so it can never contradict the sign (sidereal Gemini ≈ June, Virgo ≈ October).
    sun_by_month = []
    for mm in range(12):
        ang = [math.radians(float(sun[i])) for i in range(len(grid)) if grid[i].month == mm + 1]
        sun_by_month.append(math.degrees(math.atan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))) % 360.0)

    # ── gather every weekly topic's clean series, then FIT them all in parallel ──
    jobs = []                                             # (key, y_clean, m)
    for key, r in reg.items():
        if r.get("res") != "weekly":
            continue
        _, y = tf.load_y(r.get("label", key))             # the clean series [0, clean_end)
        if y is None or len(y) < 36:
            continue
        m = min(len(y), clean_end)
        jobs.append((key, np.asarray(y[:m], float), m))
    print(f"fitting the sinc-GD model on {len(jobs)} topics (torch, batched, device {R5._device()},"
          " mode=atlas)…", flush=True)
    Ys = [np.clip(np.round(y, 0), 0, 100).astype(float) for (_k, y, _m) in jobs]
    Xs = [sky[:m] for (_k, _y, m) in jobs]
    pars = R5.fit_many(Ys, Xs, NB, mode="atlas", progress=True)
    fits = []
    for yc, X, par in zip(Ys, Xs, pars):
        pred = R5.predict(par, X, NB)
        ss = float(np.sum((yc - yc.mean()) ** 2))
        fits.append((par, 1.0 - float(np.sum((yc - pred) ** 2)) / ss if ss > 1e-9 else 0.0))
    print(f"  fit {len(jobs)}/{len(jobs)} · {time.time()-t0:.0f}s", flush=True)

    n = 0; hist = {}; hist_season = {}; n_trend = 0
    for (key, y, m), (par, r2v) in zip(jobs, fits):
        r = reg[key]
        par = np.asarray(par, float)
        peak = R5.phase_of(par, NB); sign = SIGNS[int(peak // 30) % 12]
        freqs = R5.freqs_of(par, NB)
        weights = R5.weights_of(par, NB)                  # per-body weight w_i (>= 0 by the constrained fit)
        shares, r2_full = decompose(y, par, sky[:m])
        annual_frac = seasonality(y, moy_clean[:m])       # detrended calendar-season strength (direct from the series)
        is_trend = annual_frac < TREND_THRESH             # weak calendar season ⇒ trend-led (phase indicative only)
        # the model's in-sample FIT over the clean months + a 12-month FORECAST beyond it (the dropped year).
        fit = np.clip(R5.predict(par, sky[:m], NB), 0.0, 100.0)
        fut = sky[m:m + 12]
        forecast = np.clip(R5.predict(par, fut, NB), 0.0, 100.0) if len(fut) else np.array([])
        tdir = trend_direction(y)
        season = {"sign": sign, "peak": round(float(peak), 1), "month": season_month(peak, sun_by_month),
                  "freqs": {HKEY[i]: round(float(freqs[i]), 4) for i in range(NB)}, "r2": round(float(r2v), 4)}

        if "sid_sign" not in r:                           # preserve the sidereal originals once
            r["sid_sign"], r["sid_ref"] = r.get("sign"), r.get("ref")
            r["sid_period"], r["sid_frequency"] = r.get("period"), r.get("frequency")
        if "sid_period_curve" not in r:
            r["sid_period_curve"] = r.get("period_curve")
        if "sid_fitted" not in r:
            r["sid_fitted"] = r.get("fitted")
        r["sign"] = sign
        r["ref"] = round(float(peak), 1)
        r["season"] = season
        r["freqs"] = {HKEY[i]: round(float(freqs[i]), 4) for i in range(NB)}   # per-body frequency f_i (top-level, for export)
        r["weights"] = {HKEY[i]: round(float(weights[i]), 4) for i in range(NB)}   # per-body weight w_i (>= 0 — GATE: non-negative)
        r["fitted"] = [round(float(v), 1) for v in fit]
        r["forecast"] = [round(float(v), 1) for v in forecast]
        r["shares"] = shares
        r["r2"] = round(float(r2v), 4)
        r["annual_frac"] = round(float(annual_frac), 4)
        r["strength"] = round(float(min(1.0, max(0.0, r2v))), 3)
        r["reject_h0"] = bool(not is_trend)               # a real annual season (lead with the sign)
        r["trend"] = bool(is_trend)                       # trend-led (H0 holds), not season-led
        r["trend_dir"] = tdir
        # DOMINANT RHYTHM — 1 year when the annual season leads, else NO clean sinusoid period (a secular trend).
        r["period"] = None if is_trend else 1.0
        r["frequency"] = None if is_trend else 1.0
        r["period_curve"] = []
        r["period_prom"] = None
        r["rep"] = round(float(min(1.0, max(0.0, r2v))), 3)
        r["f1"] = round(float(r2v), 4)
        r["class_model"] = "sinc-gd"
        hist[sign] = hist.get(sign, 0) + 1                # sign distribution over ALL topics
        if not is_trend:
            hist_season[sign] = hist_season.get(sign, 0) + 1
        n_trend += int(is_trend)
        n += 1

    json.dump(reg, open(path, "w"), indent=0)
    n_season = n - n_trend
    def entropy(h, tot):
        return -sum((c / tot) * math.log2(c / tot) for c in h.values() if c) if tot else 0.0
    ent_all = entropy(hist, n)
    ent_season = entropy(hist_season, max(1, n_season))
    populated = sum(1 for s in SIGNS if hist.get(s, 0) > 0)
    print(f"\n{path}: re-analysed {n} topics with the sinc-GD baseline ({time.time()-t0:.0f}s)")
    print(f"  {n_season} SEASON-led (annual_frac ≥ {TREND_THRESH}) · {n_trend} TREND-led")
    print(f"sign DISTRIBUTION (all {n}): {populated}/12 signs · entropy {ent_all:.3f} bits (need ≥3.0)")
    print("  " + "  ".join(f"{s[:3]}:{hist.get(s, 0)}" for s in SIGNS))
    print(f"sign distribution (season-led {n_season}): entropy {ent_season:.3f} bits")
    print("  " + "  ".join(f"{s[:3]}:{hist_season.get(s, 0)}" for s in SIGNS))
    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for probe in ("christmas", "halloween", "super bowl", "swimming", "cybersecurity", "tax", "python"):
        rr = reg.get(probe) or next((reg[k] for k in reg if reg[k].get("label", "").lower() == probe), None) \
            or next((reg[k] for k in reg if probe in reg[k].get("label", "").lower()), None)
        if rr and rr.get("season"):
            flag = "SEASON→" + rr["sign"] if rr.get("reject_h0") else "trend (indicative)"
            print(f"  {probe:13s} phase {rr['season']['peak']:5.0f}° {MONTHS[rr['season']['month']]:3s} {rr['sign']:11s} {flag:20s} annual_frac={rr['annual_frac']} r2={rr['r2']}")


if __name__ == "__main__":
    main()
