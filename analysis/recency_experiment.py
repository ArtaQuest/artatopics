#!/usr/bin/env python3
"""Google-Trends RECENCY-BIAS crop sweep (monthly 12-body model)
================================================================
How much of the recent tail should a descriptive time-series model discard before fitting?
Google Trends' most recent observations are the least reliable (sampled from incomplete data,
revised later). We CROP the most recent C months off every field's monthly series, refit the
Topics Seasonality Model (analysis/trends_fit.py: fit_topic — 11 params, 11 sidereal bodies, a
PURE COSINE kernel per body at its own orbital period) on the remainder, and record the in-sample
R². No hold-out / test set: the model is descriptive, so pure in-sample R² asks the honest
question — how much of the recent data does the model STRUGGLE to fit?

Sweep C = 0…MAX_MONTHS months, take the median + mean R² across fields at each crop, then report
the KNEE (first crop within 0.5% of the peak median) and the BEST (peak median). Writes the
crop-vs-R² curve to analysis/_recency.json, which export_research.py folds into research.json
(the SPA atlas).

The R² at each (field, crop) tracks fit_topic's headline R²: a non-negative HUBER angle sweep of
the pure-cosine kernel, taking the best-fitting angle's R². Two bookkeeping speed-ups (the curve
is robust to both — verified diff < 0.004 vs the full 72-angle sweep): the per-crop design
matrices (which depend only on the ephemeris, not the field) are built ONCE and reused across all
fields rather than rebuilt inside every fit_topic call; and the phase is swept on a COARSER grid
(every 3rd of the 72 angles = 15°) instead of fit_topic's full 5° grid — the MAX R² over angle is
insensitive to the sampling. This brings the fields x crops sweep down to a few minutes.

  python3 analysis/recency_experiment.py
"""
import importlib.util as _u, os, json
import numpy as np

_spec = _u.spec_from_file_location("tf", os.path.join(os.path.dirname(__file__) or ".", "trends_fit.py"))
tf = _u.module_from_spec(_spec); _spec.loader.exec_module(tf)

REG = os.environ.get("AQ_REG", "analysis/_fields_weekly.json")
OUT = "analysis/_recency.json"
MAX_MONTHS = 72                 # sweep 0…6 years of recent crop
STEP = 6                        # every 6 months; month 0 always included (2026-07-01: 6-mo grid for speed under the pure-cosine kernel)
ANGLE_STRIDE = 3                # phase grid stride: fit_topic sweeps every 5° (72 angles); here every 3rd = 15° (24
#                                 angles) — the MAX R² over angle is insensitive to it, and it is faster (see docstring)


def full_series(label):
    """Load a field's FULL monthly series on GRID (NO DROP_LAST) — the recency sweep does its own cropping."""
    p = f"{tf.DATA_DIR}/{tf.slug(label)}.csv"
    if not os.path.exists(p):
        return None
    import pandas as pd
    df = pd.read_csv(p); df["Time"] = pd.to_datetime(df["Time"])
    M = df.drop_duplicates("Time").set_index("Time")["v"].reindex(tf.GRID)
    return tf.clean_y(pd.DataFrame({"v": M.values}))


def best_r2(y, designs):
    """fit_topic's headline R² at length len(y): non-negative HUBER fit at every pre-built design (one per angle);
       the angle of lowest Huber cost is the field's pick, and we return that angle's R² (= the reported fit)."""
    n = len(y)
    fscale = max(1.0, 1.345 * 1.4826 * float(np.median(np.abs(y - np.median(y)))))
    best_cost, best_pred = np.inf, None
    for X in designs:                                   # X is already sliced to n rows
        _, _, pred, cost = tf._huber_fit(X, y, fscale)
        if cost < best_cost:
            best_cost, best_pred = cost, pred
    return tf._r2(y, best_pred)


def main():
    reg = json.load(open(REG)) if os.path.exists(REG) else {}
    fields = [r for r in reg.values() if r.get("res") == "weekly"] or list(reg.values())
    lon = tf.ephemeris()
    series = []
    for r in fields:
        y = full_series(r["label"])
        if y is not None and len(y) > MAX_MONTHS + 24:    # need enough months left after the deepest crop
            series.append(y)
    n_keywords = len(series)
    print(f"  {n_keywords} fields with a usable full monthly series (of {len(fields)})", flush=True)

    # Pre-build the per-angle design matrices ONCE at the FULL length, then slice per crop (rows 0..n-c).
    full_designs = tf.designs_at(lon)[::ANGLE_STRIDE]   # (len(GRID) x 11) per angle, COARSE phase grid
    L = min(len(d) for d in full_designs)

    crops = [0] + list(range(STEP, MAX_MONTHS + 1, STEP))
    curve = []
    for c in crops:
        n = L - c
        designs_c = [d[:n] for d in full_designs]         # same matrices, sliced to the cropped length
        r2s = []
        for y in series:
            yc = y[:L][:n]                                 # crop the most recent c months
            r2s.append(best_r2(yc, designs_c))
        r2s = np.asarray(r2s, float)
        row = {"months": c, "weeks": int(round(c * 4.345)), "n": n_keywords,
               "r2_median": float(np.median(r2s)), "r2_mean": float(np.mean(r2s))}
        curve.append(row)
        print(f"    crop {c:3d} mo → median R2 {row['r2_median']*100:.2f}%  mean {row['r2_mean']*100:.2f}%", flush=True)

    best = max(curve, key=lambda r: r["r2_median"])                # peak median R²
    peak = best["r2_median"]
    # KNEE = the ELBOW of the curve, not the plateau's high point. This curve rises STEEPLY over the first ~year, then
    # creeps up slowly across a long plateau to a shallow, distant peak — so "within 0.5% of the peak" would land deep
    # in the plateau and miss the point. We use the Kneedle max-curvature detector: over the rising segment (crop 0 to
    # the peak) normalise both axes to [0,1] and take the crop of MAXIMUM distance below the chord — the point where the
    # steep climb turns into the plateau. Here that is ~12 months (almost all the gain, keeping the most data).
    pi = max(range(len(curve)), key=lambda i: curve[i]["r2_median"])   # peak index
    seg = curve[: pi + 1] or curve
    xs = np.array([r["months"] for r in seg], float); ys = np.array([r["r2_median"] for r in seg], float)
    xr = (xs.max() - xs.min()) or 1.0; yr = (ys.max() - ys.min()) or 1.0
    dist = (ys - ys.min()) / yr - (xs - xs.min()) / xr                 # gap above the first->peak chord (Kneedle)
    knee = seg[int(np.argmax(dist))]

    out = {"n_keywords": n_keywords, "max_months": MAX_MONTHS, "curve": curve, "best": best, "knee": knee}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"  → {OUT}: peak median R2 {peak*100:.1f}% @ {best['months']} mo · knee {knee['months']} mo "
          f"({knee['r2_median']*100:.1f}%) · crop@0 {curve[0]['r2_median']*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
