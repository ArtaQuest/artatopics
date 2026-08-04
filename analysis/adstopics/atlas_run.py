#!/usr/bin/env python3
"""adstopics — the CANONICAL ATLAS run under THE MODEL OF RECORD (operator spec, 2026-07-14):

    y_hat(t) = F(t) = Σ_i w_i · sinc( f_i · wrap(x_i(t) − p) )        (12 bodies: synodic moon +
                                                                       the platform 11)
    - 24 weights + 1 phase = 25 parameters, ALL positive (w projected >= 0, f in (0,FMAX] by
      sigmoid, p mod 360); NO intercept, NO trend, NO level term
    - EVERY fit runs 12 gradient descents, the phase initialised at each sign centre
      (15°, 45°, ..., 345°); the best validation checkpoint wins
    - two-stage pruning (12 lamps -> the 4 highest train-contribution survivors, refit) — the
      generalization gain established in the experiment tournament
    protocol: recency year excluded ENTIRELY; 70/15/15 time split; FUTURE test R² reported per
    topic; the final atlas parameters come from a full-clean-window refit AFTER metrics freeze

Per topic the atlas records: phase p -> sidereal sign, the 4 surviving lamps (+κ, w), level stats,
r2_test (future), r2_val, in-sample R², seasonality gate (season-led if r2_test > 0), and the raw
peak month. Runs the whole vocabulary in gated tiers (full gate / relaxed / excluded-flat).

  python3 analysis/adstopics/atlas_run.py
→ analysis/adstopics/atlas.json + atlas_topics.csv
"""
import importlib.util as u, json, os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

tf = _load("analysis/trends_fit.py", "tf")
r5 = _load("analysis/topic500_reference_solution.py", "r5")
ex = _load("analysis/adstopics/experiments.py", "ex")
co = _load("analysis/adstopics/combo_experiments.py", "co")

# protocol switches (defaults = the operator's named model of record)
KERNEL = os.environ.get("AQ_ATLAS_KERNEL", "gauss")           # OPERATOR 2026-07-14: gaussian is the model of record
FIXED_F = float(os.environ.get("AQ_ATLAS_FIXEDF", "1.0")) or None   # frozen width by default
INTERCEPT = os.environ.get("AQ_ATLAS_INTERCEPT", "1") == "1"
CLIP = os.environ.get("AQ_ATLAS_CLIP", "1") == "1"


def _fit(Ys, X, dev, body_mask=None):
    return co.fit_vm(Ys, X, dev, body_mask=body_mask, kernel=KERNEL,
                     fixed_f=FIXED_F, intercept=INTERCEPT)


def _kvals(par, X, a, i):
    z = np.deg2rad((X[:a] - par["p"][i] + 180.0) % 360.0 - 180.0)
    if KERNEL == "vonmises":
        return np.exp(par["kappa"][i][None, :] * (np.cos(z) - 1.0))
    if KERNEL == "cos":
        return np.cos(z)
    if KERNEL == "gauss":
        return np.exp(-(z * par["kappa"][i][None, :]) ** 2)
    return np.sinc(z * par["kappa"][i][None, :])

SIGNS = r5.SIGNS
BODIES12 = list(tf.BODIES)                      # 11 bodies — moon removed (operator 2026-07-15)
NBX = ex.NBX
CHUNK = 250


def load_all():
    grid = pd.DatetimeIndex(tf.GRID)
    i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
    iend = len(grid) - tf.DROP_LAST
    lon = tf.ephemeris()
    X = np.column_stack([np.asarray(lon[b], float)[i0:iend] for b in tf.BODIES])   # 11 bodies, NO moon
    months = [d.strftime("%Y-%m") for d in grid[i0:iend]]
    vocab = json.load(open("analysis/adstopics/vocabulary.json"))
    _bl = set(json.load(open("analysis/adstopics/blacklist.json")).get("excluded_topics", []))
    vocab = {k: v for k, v in vocab.items() if k not in _bl}
    out = []
    nn = iend - i0
    a, b = ex.split3(nn)
    for t in sorted(vocab):
        p = f"analysis/adstopics/series/{tf.slug(t)}.csv"
        if not os.path.exists(p): continue
        df = pd.read_csv(p); df["Time"] = pd.to_datetime(df["Time"])
        v = pd.to_numeric(df.drop_duplicates("Time").set_index("Time")["v"].reindex(grid[i0:iend]), errors="coerce")
        if v.notna().sum() < nn * 0.5:
            out.append((t, None, "sparse")); continue
        y = v.interpolate(limit_direction="both").to_numpy(float)
        if not np.isfinite(y).all() or float(y.max()) <= 0:
            out.append((t, None, "flat")); continue
        tier = "full" if (y.max() >= 10 and y[:a].std() >= 1.0 and y[b:].std() >= 1.0) else "relaxed"
        out.append((t, y, tier))
    return out, X, months, vocab


def main():
    data, X, months, vocab = load_all()
    usable = [(t, y, tier) for t, y, tier in data if y is not None]
    skipped = [(t, tier) for t, y, tier in data if y is None]
    print(f"[atlas] usable {len(usable)} · skipped {len(skipped)} · months {len(months)}")
    dev = r5._device()
    n = len(usable[0][1]); a, b = ex.split3(n)

    def lvl(y, upto):
        return float(np.median(y[max(0, upto - 12):upto]))

    rows = []
    for lo in range(0, len(usable), CHUNK):
        batch = usable[lo:lo + CHUNK]
        Ys = [y for _, y, _ in batch]
        resid = [y for y in Ys]                                # SPEC: the model fits the raw series
        # stage 1: all 12 lamps
        _, par1 = _fit(resid, X, dev)
        # contributions on train -> prune to 4
        C = np.zeros((len(Ys), NBX))
        for i in range(len(Ys)):
            C[i] = (par1["w"][i][None, :] * _kvals(par1, X, a, i)).std(0)
        mask = np.zeros_like(C)
        idx = np.argsort(-C, axis=1)[:, :4]
        for i in range(C.shape[0]):
            mask[i, idx[i]] = 1.0
        # stage 2: pruned refit (train, val checkpoint) -> metrics
        pred2, par2 = _fit(resid, X, dev, body_mask=mask)
        for i, (t, y, tier) in enumerate(batch):
            Lb = lvl(y, b)                                     # kept as a descriptive stat only
            pr = np.clip(pred2[i], 0.0, 100.0) if CLIP else pred2[i]
            sstv = max(((y[a:b] - y[a:b].mean()) ** 2).sum(), 1e-9)
            sst = max(((y[b:] - y[b:].mean()) ** 2).sum(), 1e-9)
            ssti = max(((y - y.mean()) ** 2).sum(), 1e-9)
            r2v = 1 - ((y[a:b] - pr[a:b]) ** 2).sum() / sstv
            r2t = 1 - ((y[b:] - pr[b:]) ** 2).sum() / sst
            r2i = 1 - ((y - pr) ** 2).sum() / ssti
            phase = float(par2["p"][i])
            lamps = [BODIES12[j] for j in idx[i]]
            mo = pd.to_datetime(pd.Series(months)).dt.month.to_numpy()
            mm = [y[mo == m].mean() if (mo == m).any() else -1 for m in range(1, 13)]
            test_std = float(y[b:].std())
            annual = 0.0
            det = y - pd.Series(y).rolling(13, center=True, min_periods=1).median().to_numpy()
            moy = pd.to_datetime(pd.Series(months)).dt.month.to_numpy()
            sm = np.array([det[moy == m].mean() if (moy == m).any() else 0.0 for m in range(1, 13)])
            sstd = max(((det - det.mean()) ** 2).sum(), 1e-9)
            annual = float(((sm[moy - 1] - det.mean()) ** 2).sum() / sstd)
            rows.append({
                "topic": t, "tier": tier, "sign": SIGNS[int(phase // 30) % 12],
                "phase": round(phase, 2),
                "lamps": ",".join(lamps),
                "w": ",".join(f"{par2['w'][i][j]:.3f}" for j in idx[i]),
                "kappa": ",".join(f"{par2['kappa'][i][j]:.2f}" for j in idx[i]),
                "level_now": round(Lb, 2), "r2_val": round(float(r2v), 4),
                "r2_test": round(float(r2t), 4), "r2_insample": round(float(r2i), 4),
                "season_led": bool(r2t > 0 and tier == "full" and test_std >= 1.0),
                "test_std": round(test_std, 2), "annual_frac": round(annual, 4),
                "raw_peak_month": int(np.argmax(mm)) + 1,
                "n_paths": vocab[t]["n_paths"],
            })
        print(f"  atlas {min(lo + CHUNK, len(usable))}/{len(usable)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("analysis/adstopics/atlas_topics.csv", index=False)
    sled = df[df["season_led"]]
    summary = {
        "topics": len(df), "skipped": len(skipped), "months": len(months),
        "window": [months[0], months[-1]],
        "season_led": int(len(sled)),
        "median_r2_test": float(df["r2_test"].median()),
        "sign_dist_all": {s: int((df["sign"] == s).sum()) for s in SIGNS},
        "sign_dist_season_led": {s: int((sled["sign"] == s).sum()) for s in SIGNS},
        "lamp_survival_all": {bd: int(df["lamps"].str.contains(bd).sum()) for bd in BODIES12},
        "lamp_survival_season_led": {bd: int(sled["lamps"].str.contains(bd).sum()) for bd in BODIES12},
        "mechanism": f"furnace kernel={KERNEL} fixed_f={FIXED_F} intercept={INTERCEPT} clip={CLIP}; 12 bodies, positive params, 12 sign-centre GD inits, prune-4 refit",
    }
    json.dump(summary, open("analysis/adstopics/atlas.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
