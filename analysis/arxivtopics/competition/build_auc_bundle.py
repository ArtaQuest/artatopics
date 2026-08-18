#!/usr/bin/env python3
"""Kaggle competition bundle — "which fields trend next year?", scored by AUC.

TARGET — cross-sectional, balanced in EVERY year, and leak-free:
    g_j(t)     = (works_j(t+1) - works_j(t)) / max(works_j(t), 1)      relative growth
    label(j,t) = 1 if g_j(t) > median over all fields alive at t of g(t)
Half the fields are 1 in every single year, by construction. Two consequences make this the right
benchmark for this question:
  * A pure calendar signal scores EXACTLY 0.5. Any model of the date alone gives every field the
    same number in a given year, and AUC is computed within the pooled test set where each year
    contributes both classes — so "science grew in the 1990s" earns nothing. The campaign kept
    finding that its accuracy was a slow clock; this target removes the clock from the board.
  * What CAN score is per-field structure that the date modulates differently for each field —
    which is exactly the claim an astrological or numerological model makes.
The threshold is a within-year median of the same quantity being predicted, so it is never
published and never available as a feature; competitors get only (field, year) plus train history.

WALL — the latest 20% of rows are the benchmark (operator 2026-08-16): future predictive accuracy.

  python3 analysis/arxivtopics/competition/build_auc_bundle.py
"""
import os, sys, json, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import arxiv_fit as af

names, Yv, labels, future = af.load_lunar()
n = Yv.shape[1]
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
_w = pd.read_csv(os.path.join(REPO, "analysis/citations/rail_works_yearly.csv"))
_c = pd.read_csv(os.path.join(REPO, "analysis/citations/citations_received_yearly.csv"))
_w = _w.set_index("subfield_id").loc[_c["subfield_id"]].reset_index()
ycols = [c for c in _w.columns if c[:1].isdigit()][:n]
W = _w[ycols].to_numpy(float)
J = W.shape[0]
years = [int(y) for y in labels]
OUT = os.path.expanduser("~/.artaquest-dev/artacomp/aucomp"); os.makedirs(OUT, exist_ok=True)

# alive span per field: continuously non-zero to the end, label needs t+1
alive = np.zeros((J, n), bool)
for j in range(J):
    w = W[j]; nz = np.ones(n, bool)
    for i in range(n - 2, -1, -1): nz[i] = (w[i] > 0) & nz[i + 1]
    t0 = int(nz.argmax())
    if n - 1 - t0 < 30: continue
    alive[j, t0:n - 1] = True

G = np.full((J, n), np.nan)
for j in range(J):
    ts = np.where(alive[j])[0]
    if len(ts) == 0: continue
    G[j, ts] = (W[j, ts + 1] - W[j, ts]) / np.maximum(W[j, ts], 1)

rows = []
for t in range(n - 1):
    js = np.where(alive[:, t])[0]
    if len(js) < 20: continue                      # a within-year median needs a real cross-section
    g = G[js, t]; med = float(np.median(g))
    for j, gv in zip(js, g):
        rows.append((int(j), years[t], int(W[j, t]), int(gv > med)))
ally = np.array([r[1] for r in rows])
WALL_Y = int(np.quantile(ally, 0.80))
print(f"rows {len(rows)} · fields {len(set(r[0] for r in rows))} · years {ally.min()}..{ally.max()}")
print(f"latest-20% wall: train < {WALL_Y}, benchmark {WALL_Y}..{ally.max()} "
      f"({(ally >= WALL_Y).mean()*100:.1f}% of rows)")

tr = [r for r in rows if r[1] < WALL_Y]
te = [r for r in rows if r[1] >= WALL_Y]
print(f"train {len(tr)} (balance {np.mean([r[3] for r in tr]):.3f}) · "
      f"test {len(te)} (balance {np.mean([r[3] for r in te]):.3f})")

with open(f"{OUT}/train.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "field", "year", "works", "target"])
    for i, r in enumerate(tr): w.writerow([f"tr_{i}", names[r[0]], r[1], r[2], r[3]])
with open(f"{OUT}/test.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "field", "year"])
    for i, r in enumerate(te): w.writerow([f"te_{i}", names[r[0]], r[1]])
with open(f"{OUT}/sample_submission.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "target"])
    for i in range(len(te)): w.writerow([f"te_{i}", 0.5])
teyears = np.array([r[1] for r in te])
cut = int(np.quantile(teyears, 0.35))
with open(f"{OUT}/solution.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "target", "Usage"])
    for i, r in enumerate(te): w.writerow([f"te_{i}", r[3], "Public" if r[1] < cut else "Private"])
npub = int((teyears < cut).sum())
print(f"solution: public {npub} (years <{cut}) · private {len(te) - npub}")
json.dump({"rows": len(rows), "train": len(tr), "test": len(te), "wall": WALL_Y,
           "public_cut": cut, "public": npub, "private": len(te) - npub,
           "train_balance": float(np.mean([r[3] for r in tr])),
           "test_balance": float(np.mean([r[3] for r in te]))},
          open(f"{OUT}/stats.json", "w"), indent=1)
print("bundle written to", OUT)
