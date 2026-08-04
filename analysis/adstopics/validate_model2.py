#!/usr/bin/env python3
"""adstopics — model-2 validation harness: on N already-fetched topics, compare OUT-OF-SAMPLE
(untouched test tail) R² of (a) the old topic-500 atlas fit, (b) MODEL 2, (c) the harmonic+trend
closed-form baseline — the paper's model-improvement evidence.
  python3 analysis/adstopics/validate_model2.py 300
"""
import importlib.util as u, json, os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
tf = _load("analysis/trends_fit.py", "tf")
r5 = _load("analysis/topic500_reference_solution.py", "r5")
m2 = _load("analysis/adstopics/model2.py", "m2")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
lon = tf.ephemeris()
grid = pd.DatetimeIndex(tf.GRID)
i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
iend = len(grid) - tf.DROP_LAST
X = np.column_stack([np.asarray(lon[b], float)[i0:iend] for b in tf.BODIES])

vocab = sorted(json.load(open("analysis/adstopics/vocabulary.json")))
Ys, names = [], []
for t in vocab:
    p = f"analysis/adstopics/series/{tf.slug(t)}.csv"
    if not os.path.exists(p): continue
    df = pd.read_csv(p); df["Time"] = pd.to_datetime(df["Time"])
    v = pd.to_numeric(df.drop_duplicates("Time").set_index("Time")["v"].reindex(grid[i0:iend]), errors="coerce")
    if v.notna().sum() < (iend - i0) * 0.5: continue
    y = v.interpolate(limit_direction="both").to_numpy(float)
    if not np.isfinite(y).all() or float(np.nanmax(y)) <= 0: continue
    Ys.append(y); names.append(t)
    if len(Ys) >= N: break
print(f"[validate] {len(Ys)} topics")

# MODEL 2 (includes harmonic baseline per topic)
res2 = m2.fit_many2(Ys, [X] * len(Ys), progress=True)
# OLD atlas fit, scored on the same untouched test tail
n = len(Ys[0]); a, b = m2.split3(n)
par_old = r5.fit_many([y[:b][r5.canon_order(X[:b])] if False else y[:b] for y in Ys], X[:b], mode="atlas", progress=True)
r2_old = []
for i, y in enumerate(Ys):
    pred = r5.predict(par_old[i], X)
    yo = y; sst = float(np.sum((yo[b:] - yo[b:].mean()) ** 2))
    r2_old.append(1.0 - float(np.sum((yo[b:] - pred[b:]) ** 2)) / sst if sst > 1e-9 else 0.0)
out = pd.DataFrame({
    "topic": names,
    "oos_model2": [r["r2_test"] for r in res2],
    "oos_old_atlas": np.round(r2_old, 4),
    "oos_harmonic": [r["r2_test_harmonic"] for r in res2],
    "decay": [r["decay"] for r in res2],
})
out.to_csv("analysis/adstopics/model2_validation.csv", index=False)
print(out[["oos_model2", "oos_old_atlas", "oos_harmonic"]].describe().round(3))
print("mean:", out[["oos_model2", "oos_old_atlas", "oos_harmonic"]].mean().round(4).to_dict())
print("median:", out[["oos_model2", "oos_old_atlas", "oos_harmonic"]].median().round(4).to_dict())
print("decay distribution:", out["decay"].value_counts().to_dict())
