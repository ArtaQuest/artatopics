#!/usr/bin/env python3
"""Tune the ONE global kernel hyperparameter PERIOD_SCALE by MAX MEAN IN-SAMPLE R^2 over ALL collected
weekly fields (DROP_LAST fixed at 52). The kernel width for each body is sinc(dist / (PERIOD_SCALE * period_b)):
a small scale keeps the fast Moon narrow (it aliases at weekly → contributes nothing), a large scale widens
every kernel (the Moon gets captured, but the slow bodies over-broaden and the trend fits weaken). The data
picks the balance. The sweep uses a fast non-negative least-squares fit (the scale optimum matches the final
Huber fit's); scale is passed EXPLICITLY into the design (never via a module global — that aliasing bug made
an earlier version look scale-invariant). Writes analysis/_period_scale.json -> {scale, drop}.

  python3 analysis/tune_period.py
"""
import importlib.util as u, json, os, glob, statistics as st
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")

def L(p):
    s = u.spec_from_file_location(p.split("/")[-1][:-3], p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

wf = L("analysis/weekly_fit.py"); TF = wf.tf
BODIES = TF.BODIES; PERIOD = TF.PERIOD_YEARS
REFS = np.arange(0, 360, 5)                       # 5° sweep — fast, and the scale optimum is robust to it
DROP = 52
SCALES = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def designs(lon, scale, n):
    out = []
    for r in REFS:
        cols = [np.sinc(np.deg2rad((lon[b][:n] - r + 180) % 360 - 180) / (scale * PERIOD[b])) for b in BODIES]
        out.append(np.column_stack(cols + [np.ones(n)]))
    return out


def best_r2(y, des):
    b = -1.0
    for X in des:
        b = max(b, float(LinearRegression(positive=True).fit(X[:, :-1], y).score(X[:, :-1], y)))
    return b


def main():
    LON = TF.ephemeris(); lon = TF.eff_lon(LON)            # synodic Moon
    lon = {b: np.asarray(lon[b], float) for b in BODIES}
    series = {}
    for p in sorted(glob.glob("analysis/data_weekly/*.csv")):
        y = TF.clean_y(pd.read_csv(p))
        if y is not None and len(y) > DROP + 60:
            series[os.path.basename(p)[:-4]] = y[:-DROP]
    n = len(next(iter(series.values())))
    lon = {b: v[:n] for b, v in lon.items()}
    print(f"[tune] {len(series)} weekly fields · scales {SCALES[0]}–{SCALES[-1]} · max mean in-sample R²", flush=True)
    curve = {}
    for sc in SCALES:
        des = designs(lon, sc, n)
        r2s = [best_r2(y, des) for y in series.values()]
        curve[sc] = float(st.mean(r2s))
        fm = (f"  (full-moon Moon-capable)" if "full-moon" in series else "")
        print(f"  scale {sc:5.1f}: mean R² {curve[sc]*100:5.2f}%   median {st.median(r2s)*100:5.2f}%", flush=True)
    best = max(curve, key=curve.get)
    json.dump({"scale": best, "drop": DROP, "mean_r2": round(curve[best], 4), "n_fields": len(series),
               "tuned": "max mean in-sample R² over all collected fields"}, open("analysis/_period_scale.json", "w"))
    print(f"[tune] BEST PERIOD_SCALE={best}  (mean in-sample R² {curve[best]*100:.2f}% over {len(series)} fields) → analysis/_period_scale.json", flush=True)


if __name__ == "__main__":
    main()
