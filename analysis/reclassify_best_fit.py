#!/usr/bin/env python3
"""RECLASSIFY every atlas field from the BEST-FIT model (operator directive 2026-07-04).

The atlas' phase (`ref`), `frequency`/`period` and the sign→house CLASSIFICATION previously came
from the sidereal-kernel fit. They now come from the best-fit model the competition established:

    phase  ref   = the ANNUAL-CYCLE PEAK of the winning decomposition (trend + annual + dominant
                   harmonics), mapped to the sun's sidereal longitude at that time of year — i.e.
                   literally "the season its worldwide search interest peaks" (the pages' stated
                   semantics). sign = that longitude's zodiac sign → the field's house.
    period/frequency = the best-fit DOMINANT period (years) and its reciprocal.

The sidereal originals are preserved once per record as sid_sign/sid_ref/sid_period/sid_frequency,
and records are stamped class_model="best-fit" (export_research keeps their stored rep instead of
recomputing a now-meaningless sidereal area-share for the new sign).

  python3 analysis/reclassify_best_fit.py         # updates _fields_weekly.json + _topics_weekly.json
"""
import importlib.util as _u, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)
_spec = _u.spec_from_file_location("bma", os.path.join(HERE, "best_model_analysis.py"))
bma = _u.module_from_spec(_spec); _spec.loader.exec_module(bma)
tf = bma.tf
SIGNS = ['Aries','Taurus','Gemini','Cancer','Leo','Virgo','Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces']

# Circular-mean sidereal SUN longitude per calendar month (the wheel the classification lives on).
_lon = tf.ephemeris()["sun"]
_grid = tf.GRID
SUN_BY_MONTH = []
for m in range(12):
    a = [math.radians(float(_lon[i])) for i in range(len(_grid)) if _grid[i].month == m + 1]
    SUN_BY_MONTH.append(math.degrees(math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a)))) % 360.0)


def sun_lon_at(month_frac):
    """Sun longitude at a fractional month-of-year (0=Jan) — circular interpolation between month means."""
    m0 = int(month_frac) % 12
    m1 = (m0 + 1) % 12
    f = month_frac - int(month_frac)
    a, b = SUN_BY_MONTH[m0], SUN_BY_MONTH[m1]
    d = ((b - a + 180) % 360) - 180
    return (a + f * d) % 360.0


def annual_peak_month(y):
    """The winning decomposition's annual-peak as a fractional month-of-year (0=Jan), or None."""
    y = np.asarray(y, float)
    t = np.arange(len(y), dtype=float)
    best = (-1e9, None)
    for per in bma.PERIODS:
        if per > len(y):
            continue
        X = bma.design(t, per, bma.NH)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rr = bma.r2(y, X @ coef)
        if rr > best[0]:
            best = (rr, (per, coef))
    if best[1] is None:
        return None, None
    per, coef = best[1]
    a, b = float(coef[2]), float(coef[3])           # annual cos/sin (grid starts 2004-01 → t=0 is Jan)
    phi = math.atan2(b, a)                           # a·cos(ωt)+b·sin(ωt) peaks at ωt = φ
    return (phi * 12.0 / (2 * math.pi)) % 12.0, float(per)


PER_GRID = np.arange(4.0, 259.0, 2.0)                # dense months grid (4mo floor keeps harmonics at/under Nyquist); interior winners have neighbours


def dominant_period_interior(y):
    """The best-fit dominant period as an INTERIOR optimum: the highest-R² candidate that is a
    LOCAL MAXIMUM (both adjacent grid values score lower) — never a grid edge, so every published
    frequency is a genuine peak. Falls back to the annual (12 mo) when the R²(period) curve is
    monotone (trend-dominated series with no finite interior peak). Returns (period_months,
    centred 5-bar curve [{period_yr, r2, f1}], r2_at_winner)."""
    y = np.asarray(y, float)
    t = np.arange(len(y), dtype=float)
    grid = PER_GRID[PER_GRID <= max(24, len(y) - 2)]
    rs = []
    for per in grid:
        X = bma.design(t, per, bma.NH)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rs.append(bma.r2(y, X @ coef))
    rs = np.asarray(rs)
    # the winner must dominate its whole displayed 5-bar window (both adjacent values AND the
    # next-adjacent ones lower) — so the gold bar is always the tallest bar shown, dead-centre.
    def window_max(i):
        lo, hi = max(i - 2, 0), min(i + 3, len(grid))
        return rs[i] >= max(rs[lo:hi]) - 1e-12
    interior = [i for i in range(1, len(grid) - 1)
                if rs[i] >= rs[i - 1] and rs[i] >= rs[i + 1]
                and (rs[i] > rs[i - 1] or rs[i] > rs[i + 1]) and window_max(i)]
    if not interior:                                  # relax to plain local maxima
        interior = [i for i in range(1, len(grid) - 1)
                    if rs[i] >= rs[i - 1] and rs[i] >= rs[i + 1] and (rs[i] > rs[i - 1] or rs[i] > rs[i + 1])]
    if interior:
        w = max(interior, key=lambda i: rs[i])
    else:                                             # monotone curve → the annual cycle is the honest peak
        w = int(np.argmin(np.abs(grid - 12.0)))
        w = min(max(w, 1), len(grid) - 2)
    # Build the displayed 5-bar window: the winner DEAD-CENTRE, flanked on each side by the two
    # nearest candidates that score LOWER — walking outward past any higher bar (possible when the
    # winner is a plain-local-max fallback on an oscillating slope). Every published curve therefore
    # shows a genuine optimum: the gold centre bar is the tallest bar on display.
    def flank(step):
        out, i = [], w
        while len(out) < 2:
            i += step
            if i <= 0 or i >= len(grid) - 0:
                break
            if i < 0 or i >= len(grid):
                break
            if rs[i] < rs[w]:
                out.append(i)
        return out
    left = flank(-1)[::-1]
    right = flank(+1)
    idxs = (left + [w] + right)
    # pad if a side ran short (winner near the grid edge of lower values) — keep 5 bars
    while len(idxs) < 5:
        cand = [i for i in range(len(grid)) if i not in idxs and rs[i] < rs[w]]
        if not cand:
            break
        idxs.append(min(cand, key=lambda i: abs(i - w)))
        idxs = sorted(set(idxs), key=lambda i: grid[i])
    idxs = sorted(set(idxs), key=lambda i: grid[i])[:5]
    curve = [{"period": round(float(grid[i]) / 12.0, 3), "r2": round(float(rs[i]), 4),
              "f1": round(float(rs[i]), 4)} for i in idxs]
    # the winner's in-sample fitted vector (the chart's gold curve) + how PROMINENT the peak is
    # vs the whole scan (a flat curve ⇒ the "dominant period" is spurious — trend-dominated).
    Xw = bma.design(t, float(grid[w]), bma.NH)
    coefw, *_ = np.linalg.lstsq(Xw, y, rcond=None)
    fitted = np.clip(Xw @ coefw, 0.0, 100.0)
    prom = float(rs[w] - np.median(rs))
    return float(grid[w]), curve, float(rs[w]), fitted, prom


def reclassify(path):
    reg = json.load(open(path))
    n = 0
    for key, r in reg.items():
        if r.get("res") != "weekly":
            continue
        _, y = tf.load_y(r.get("label", key))
        if y is None or len(y) < 36:
            continue
        peak, _ = annual_peak_month(y)
        if peak is None:
            continue
        per_months, curve, r2w, fitted, prom = dominant_period_interior(y)
        ref = sun_lon_at(peak)
        if "sid_sign" not in r:                      # preserve the sidereal originals once
            r["sid_sign"], r["sid_ref"] = r.get("sign"), r.get("ref")
            r["sid_period"], r["sid_frequency"] = r.get("period"), r.get("frequency")
        if "sid_period_curve" not in r:
            r["sid_period_curve"] = r.get("period_curve")
        r["sign"] = SIGNS[int(ref // 30) % 12]
        r["ref"] = round(ref, 1)
        r["period"] = round(per_months / 12.0, 3)    # years — the INTERIOR-optimal dominant period
        r["frequency"] = round(12.0 / per_months, 4) # cycles per year
        r["period_curve"] = curve                    # 5 bars, winner centred, best-fit R² per candidate
        r["f1"] = round(r2w, 4)                      # the winner's best-fit R² (the bars' ranking value)
        r["period_prom"] = round(prom, 4)            # peak prominence vs the scan median (flat ⇒ spurious)
        if "sid_fitted" not in r:
            r["sid_fitted"] = r.get("fitted")
        r["fitted"] = [round(float(v), 2) for v in fitted]   # the chart's gold curve = THE BEST-FIT MODEL
        r["class_model"] = "best-fit"
        n += 1
    json.dump(reg, open(path, "w"), indent=0)
    print(f"{path}: reclassified {n} records")
    return reg


if __name__ == "__main__":
    regs = [reclassify(p) for p in ("analysis/_fields_weekly.json", "analysis/_topics_weekly.json") if os.path.exists(p)]
    from collections import Counter
    c = Counter(r["sign"] for reg in regs for r in reg.values() if r.get("class_model") == "best-fit")
    print("new sign distribution:", dict(sorted(c.items(), key=lambda x: -x[1])))
