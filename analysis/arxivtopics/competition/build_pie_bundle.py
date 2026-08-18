#!/usr/bin/env python3
"""THE PIE BENCHMARK (operator 2026-08-16): will this field hold a BIGGER SLICE next year?

    share_j(t) = field j's share of that year's total       (Yv: citations, sums to 100 each year)
    label(j,t) = 1 if share_j(t+1) > share_j(t)

Trending is relative BY CONSTRUCTION — a field only gains slice by outgrowing the field as a whole,
so the pie is a zero-sum frame and the year cancels out: a model of the date alone scores exactly
0.5. Balance falls out at ~0.507 with no thresholds, no medians, nothing tuned. This also gives the
deployed phasor a like-for-like test, because share is precisely the quantity it forecasts.

Wall: the latest 20% of rows are the benchmark, train is everything before it.

  python3 analysis/arxivtopics/competition/build_pie_bundle.py
"""
import os, sys, json, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import arxiv_fit as af

names, Yv, labels, future = af.load_lunar()
n = Yv.shape[1]; years = [int(y) for y in labels]
tv = af.META["topic_valid"]
J = len(names)
OUT = os.path.expanduser("~/.artaquest-dev/artacomp/piecomp"); os.makedirs(OUT, exist_ok=True)

rows = []
for t in range(n - 1):
    js = np.where(tv[:, t] & tv[:, t + 1])[0]
    if len(js) < 20: continue
    for j in js:
        rows.append((int(j), years[t], float(Yv[j, t]), int(Yv[j, t + 1] > Yv[j, t])))
ally = np.array([r[1] for r in rows])
WALL_Y = int(np.quantile(ally, 0.80))
tr = [r for r in rows if r[1] < WALL_Y]; te = [r for r in rows if r[1] >= WALL_Y]
print(f"rows {len(rows)} · years {ally.min()}..{ally.max()} · wall {WALL_Y} "
      f"({(ally >= WALL_Y).mean()*100:.1f}% held out)")
print(f"train {len(tr)} (balance {np.mean([r[3] for r in tr]):.3f}) · "
      f"test {len(te)} (balance {np.mean([r[3] for r in te]):.3f})")

with open(f"{OUT}/train.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "field", "year", "share", "target"])
    for i, r in enumerate(tr): w.writerow([f"tr_{i}", names[r[0]], r[1], round(r[2], 6), r[3]])
with open(f"{OUT}/test.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "field", "year"])
    for i, r in enumerate(te): w.writerow([f"te_{i}", names[r[0]], r[1]])
with open(f"{OUT}/sample_submission.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "target"])
    for i in range(len(te)): w.writerow([f"te_{i}", 0.5])
teyears = np.array([r[1] for r in te]); cut = int(np.quantile(teyears, 0.35))
with open(f"{OUT}/solution.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "target", "Usage"])
    for i, r in enumerate(te): w.writerow([f"te_{i}", r[3], "Public" if r[1] < cut else "Private"])
json.dump({"rows": len(rows), "train": len(tr), "test": len(te), "wall": WALL_Y, "public_cut": cut,
           "train_balance": float(np.mean([r[3] for r in tr])),
           "test_balance": float(np.mean([r[3] for r in te]))}, open(f"{OUT}/stats.json", "w"), indent=1)
print("bundle written to", OUT)
