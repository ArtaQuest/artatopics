#!/usr/bin/env python3
"""THE ATLAS, RESCORED PER TOPIC (operator 2026-08-16).

Each field is scored on its own: over that field's benchmark years, does the feature rank the years
it GAINED slice above the years it lost slice? One AUC per field, then averaged across fields. This
is the timing claim astrology actually makes, and it changes what can score:

  * the natal promise, and every per-field constant, becomes EXACTLY 0.5 — it cannot order a
    field's own years. The persistence effect that dominated every earlier result is gone.
  * the calendar comes BACK. A feature that is identical for all fields in a year still varies
    across one field's years, so the transiting-only features stop being controls and become the
    quantity to watch: whatever they score is what any transit feature can earn for free.

Direction (the sign of the 2-param logistic's slope) is still fitted on the 30 years before the
wall and applied unchanged to 1997-2024.

  python3 analysis/arxivtopics/competition/atlas_per_topic.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd, importlib.util
from sklearn.metrics import roc_auc_score
BUN = os.path.expanduser("~/.artaquest-dev/artacomp/piecomp")
spec = importlib.util.spec_from_file_location("atlas", os.path.join(HERE, "feature_atlas.py"))
src = open(spec.origin).read().split("print(f\"catalogue:")[0]
ns = {"__file__": spec.origin}; exec(compile(src, spec.origin, "exec"), ns)
CAT, Y, YR, FLD, FIT, TEST, AGE = (ns[k] for k in ("CAT","Y","YR","FLD","FIT","TEST","AGE"))
print(f"catalogue: {len(CAT)} features", flush=True)

def tauc(mask, s, y=None):
    """One AUC per FIELD over its own years, then averaged. Fields lacking both classes are skipped."""
    yy = Y if y is None else y
    out = []
    for f in np.unique(FLD[mask]):
        m = mask & (FLD == f)
        if m.sum() < 6 or len(set(yy[m])) < 2: continue
        out.append(roc_auc_score(yy[m], s[m]))
    return float(np.mean(out)), len(out)

rows = []
for nm, trad, expl, v in CAT:
    if np.std(v[FIT]) < 1e-12: continue
    z = (v - v[FIT].mean()) / (v[FIT].std() + 1e-12)
    c = np.corrcoef(z[FIT], Y[FIT])[0, 1]
    sgn = 1.0 if (c >= 0 or np.isnan(c)) else -1.0
    a_tr, _ = tauc(FIT, sgn*z); a_te, nf = tauc(TEST, sgn*z)
    rows.append(dict(feature=nm, tradition=trad, explanation=expl,
                     direction=("higher favours gaining slice" if sgn > 0 else "lower favours gaining slice"),
                     train_auc=round(a_tr,4), test_auc=round(a_te,4), fields=nf))
df = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
# noise floor: permute each field's labels among ITS OWN years, refit direction, same pipeline
rng = np.random.RandomState(0); null = []
for _ in range(200):
    yp = Y.copy()
    for f in np.unique(FLD):
        m = FLD == f; yp[m] = rng.permutation(yp[m])
    v = CAT[rng.randint(len(CAT))][3]
    z = (v - v[FIT].mean())/(v[FIT].std()+1e-12)
    c = np.corrcoef(z[FIT], yp[FIT])[0,1]
    s = 1.0 if (c >= 0 or np.isnan(c)) else -1.0
    null.append(tauc(TEST, s*z, y=yp)[0])
lo, hi = float(np.percentile(null,2.5)), float(np.percentile(null,97.5))
df["above_noise"] = df.test_auc > hi
df.to_csv(f"{BUN}/feature_atlas_per_topic.csv", index=False)
print(f"\nnoise floor from 200 within-FIELD label permutations: 95% band {lo:.4f}-{hi:.4f}", flush=True)
print(f"above it: {int(df.above_noise.sum())} of {len(df)} (chance ~{0.025*len(df):.0f})", flush=True)
print("\n— top 20 per-topic averaged AUC:", flush=True)
for _, r in df.head(20).iterrows():
    print(f"  {r.test_auc:.4f} (train {r.train_auc:.4f})  {r.feature:<38} {r.tradition}", flush=True)
print("\n— the calendar features (identical for every field in a year — now NOT controls):", flush=True)
for _, r in df[df.tradition == "CONTROL"].iterrows():
    print(f"  {r.test_auc:.4f}  {r.feature}", flush=True)
print("\n— by tradition:", flush=True)
g = df[df.tradition != "CONTROL"].groupby("tradition").agg(n=("test_auc","size"), best=("test_auc","max"), median=("test_auc","median"))
print(g.sort_values("best", ascending=False).to_string(), flush=True)
json.dump({"n": len(df), "null_band":[round(lo,4),round(hi,4)], "above": int(df.above_noise.sum()),
           "expected": round(0.025*len(df),1),
           "calendar": df[df.tradition=="CONTROL"][["feature","test_auc"]].to_dict("records"),
           "top": df.head(25)[["feature","tradition","test_auc","train_auc"]].to_dict("records")},
          open(f"{BUN}/atlas_per_topic_summary.json","w"), indent=1)
