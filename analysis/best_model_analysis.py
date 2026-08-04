#!/usr/bin/env python3
"""Best-model analysis of EVERY atlas field's Google-Trends series.

The sky/sidereal fit (trends_fit.py) is the platform's celestial-seasonality descriptor (mean in-sample
R² ~0.67). The skills-500 competition duel established the model that ACTUALLY fits these monthly series
best: a per-field HARMONIC decomposition — a linear trend plus the annual (12-month) cycle plus the
field's own dominant long period (found by search) — which is what makes the series so predictable. This
script fits that best model to all ~1500 atlas fields' full monthly series and records, per field:

    bestR2      in-sample R² of trend + annual + dominant-period harmonics (how periodic the series is)
    periodBest  the dominant period in months (the search winner; the true seasonal wavelength)
    annualFrac  fraction of the explained variance carried by the annual (12-month) cycle
    trendFrac   fraction carried by the linear trend

Writes analysis/_best_model.json {key: {bestR2, periodBest, annualFrac, trendFrac}} and merges these
four fields into artaquest-web/src/data/research.json (additive — the celestial fit is untouched), so
every field page can show the best achievable fit alongside its celestial reading.

  python3 analysis/best_model_analysis.py            # analyse + write _best_model.json + merge research.json
"""
import importlib.util as _u, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)
_spec = _u.spec_from_file_location("tf", os.path.join(HERE, "trends_fit.py"))
tf = _u.module_from_spec(_spec); _spec.loader.exec_module(tf)

# The DEPLOYED model: the skills-500 competition-winning ensemble (per-field holdout protocol).
# gen_annual imports the bundle's `decode` module, which loads a competition ephemeris at import
# time — irrelevant here (the atlas has real dates) — so stub it before the import.
import types
_stub = types.ModuleType("decode")
_stub.BODIES = ["sun", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "chiron", "node"]
_stub.decode = None
sys.modules["decode"] = _stub
sys.path.insert(0, os.path.join(HERE, "skills500_winning_solution"))
import gen_annual as GA          # the winning Model-B machinery (members/OOF/NNLS)

ALLFIT = "analysis/_allfit.json"
RESEARCH = "artaquest-web/src/data/research.json"
OUT = "analysis/_best_model.json"
PAPERM = "analysis/_paper_model.json"                      # paper #17 rung-(g) holdout scores (paper_model_analysis.py)
PERIODS = np.concatenate([np.arange(6, 60, 3), np.arange(60, 260, 4)]).astype(float)  # dominant-period search grid (months)
NH = 2                                                     # harmonics of the dominant period


def design(t, period, nh, annual=True):
    """trend + annual(12) + nh harmonics of `period`, as a least-squares design matrix over month index t."""
    cols = [np.ones_like(t), t]                            # intercept + linear trend
    if annual:
        cols += [np.cos(2 * np.pi * t / 12.0), np.sin(2 * np.pi * t / 12.0)]
    for h in range(1, nh + 1):
        cols += [np.cos(2 * np.pi * h * t / period), np.sin(2 * np.pi * h * t / period)]
    return np.column_stack(cols)


def r2(y, yh):
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - yh) ** 2)) / ss if ss > 1e-9 else 0.0


def fit_best(y):
    """Fit trend+annual+dominant-period harmonics; return (bestR2, periodBest, annualFrac, trendFrac)."""
    y = np.asarray(y, float)
    n = len(y)
    t = np.arange(n, dtype=float)
    best = (-1e9, PERIODS[0], None)
    for per in PERIODS:
        if per > n:                                        # can't identify a period longer than the record
            continue
        X = design(t, per, NH)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rr = r2(y, X @ coef)
        if rr > best[0]:
            best = (rr, per, (X, coef))
    bestR2, periodBest, fit = best
    # variance-share of the annual + trend terms (drop-to-mean R² attribution on the winning design)
    annualFrac = trendFrac = 0.0
    if fit is not None:
        X, coef = fit
        full = r2(y, X @ coef)
        # trend cols are 1 (linear); annual cols are 2,3 (cos/sin 12)
        for name, idxs in (("trend", [1]), ("annual", [2, 3])):
            Xd = X.copy()
            for j in idxs:
                Xd[:, j] = Xd[:, j].mean()                 # flatten this term
            drop = full - r2(y, Xd @ coef)
            if name == "trend": trendFrac = max(0.0, drop)
            else: annualFrac = max(0.0, drop)
        tot = full if full > 1e-9 else 1.0
        annualFrac = round(min(1.0, annualFrac / tot), 4)
        trendFrac = round(min(1.0, trendFrac / tot), 4)
    return round(max(0.0, bestR2), 4), round(float(periodBest), 1), annualFrac, trendFrac


def winning_model_holdout(y, seed):
    """Score the COMPETITION-WINNING model on this field under the competition's own protocol:
    hold out a seeded random 20% of the months (the same 206-known/52-held geometry as the live
    contest), fit the winning ensemble (interpolator/smoother library + fixed-annual + dominant-
    period harmonic members, OOF NNLS stack — fast config) on the rest, and return the holdout R².
    This is THE deployed number: directly comparable to the public leaderboard (~0.903)."""
    y = np.asarray(y, float)
    n = len(y)
    x = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    k = max(4, int(round(n * 0.20)))
    hold = np.zeros(n, bool)
    hold[rng.choice(n, k, replace=False)] = True
    xt, yt = x[~hold], y[~hold]
    xq, yq = x[hold], y[hold]
    dom = GA.dominant_period(xt, yt, seed=seed)
    _, names = GA.members(xt, yt, dom)
    P, Y = GA.oof(xt, yt, names, dom, 6, seed + 1)          # 6 OOF repeats — the fast config
    w = GA.fit_nnls(P, Y)
    cc, _ = GA.members(xt, yt, dom)
    pred = np.zeros(len(xq))
    for wi, nm in zip(w, names):
        pred += wi * np.asarray(cc[nm](xq), float)
    pred = np.clip(pred, 0.0, 100.0)
    ss = float(np.sum((yq - yq.mean()) ** 2))
    return round(1.0 - float(np.sum((yq - pred) ** 2)) / ss, 4) if ss > 1e-9 else 0.0


def main():
    reg = json.load(open(ALLFIT))
    out = {}
    done = skip = 0
    for key, rec in reg.items():
        label = rec.get("label", key)
        _, y = tf.load_y(label)
        if y is None or len(y) < 36:
            skip += 1
            continue
        b, per, af, tfrac = fit_best(y)
        import zlib
        win = winning_model_holdout(y, zlib.crc32(key.encode()) % (2**31))
        out[key] = {"bestR2": b, "periodBest": per, "annualFrac": af, "trendFrac": tfrac, "winR2": win}
        done += 1
        if done % 200 == 0:
            print(f"  {done} fields analysed…", flush=True)
    json.dump(out, open(OUT, "w"), indent=0)
    wins = [v["winR2"] for v in out.values()]
    print(f"  DEPLOYED winning-model holdout R²: mean {np.mean(wins):.3f}  median {np.median(wins):.3f} "
          f"(directly comparable to the competition's ~0.903)")
    r2s = [v["bestR2"] for v in out.values()]
    print(f"analysed {done} fields ({skip} lacked a usable series) → {OUT}")
    print(f"  best-model R²: mean {np.mean(r2s):.3f}  median {np.median(r2s):.3f}  "
          f"(vs the celestial fit's ~0.67 mean) · {sum(x > 0.8 for x in r2s)} fields > 0.8")

    # ── merge into research.json (additive) ────────────────────────────────────
    # Also re-adds what export_research.py drops on rebuild: sidSign (the sidereal original, from the
    # registry's sid_sign) and paperR2 (the paper #17 rung-(g) holdout R², from _paper_model.json).
    if os.path.exists(RESEARCH):
        research = json.load(open(RESEARCH))
        paperm = json.load(open(PAPERM)) if os.path.exists(PAPERM) else {}
        merged = 0
        for tp in research.get("topics", []):
            b = out.get(tp["key"])
            if b:
                tp.update(bestR2=b["bestR2"], periodBest=b["periodBest"],
                          annualFrac=b["annualFrac"], trendFrac=b["trendFrac"], winR2=b.get("winR2"))
                merged += 1
            rec = reg.get(tp["key"]) or {}
            if rec.get("sid_sign"):
                tp["sidSign"] = rec["sid_sign"]
            pm = paperm.get(tp["key"], {}).get("paperR2")
            if pm is not None:
                tp["paperR2"] = pm
        json.dump(research, open(RESEARCH, "w"), separators=(",", ":"))
        print(f"  merged best-model fields into {merged} / {len(research.get('topics', []))} research.json topics (+ sidSign + paperR2)")


if __name__ == "__main__":
    main()
