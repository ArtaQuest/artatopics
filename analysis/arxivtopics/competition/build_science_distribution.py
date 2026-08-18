#!/usr/bin/env python3
"""SCIENCE-DISTRIBUTION — the dataset: how the world's attention divides across 251 research fields,
year by year, 1858-2025, and whether each field's slice grows.

Files:
  shares.csv      field x year matrix of each field's share of that year's citations (the pie)
  ephemeris.csv   the seven slow bodies' ecliptic longitudes per year, 1700-2055 (tropical) —
                  the ONLY input a model is allowed at prediction time, besides the date itself
  train.csv       (id, field, year, share, target) for every year before the wall
  test.csv        (id, field, year) for the last 20% of years — no share, no label
  solution.csv    (id, target, Usage) held back
  README.md       the task, the metric, the rules

Target: target = 1 if the field's share is LARGER next year than this year. Trending is relative by
construction — a field only gains slice by outgrowing the field as a whole — so the pie is zero-sum
and the calendar cannot help a within-topic ranking.

Metric: mAUC — one ROC-AUC per FIELD across that field's own test years, averaged over fields. A
per-field constant scores exactly 0.5, so the model must time each field, not rank fields.

Wall: the last 20% of years are held out, so every training year precedes every test year.

  python3 analysis/arxivtopics/competition/build_science_distribution.py
"""
import os, sys, json, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import arxiv_fit as af

names, Yv, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
tv = af.META["topic_valid"]
years = np.array([int(y) for y in labels]); J, n = Yv.shape
BODY = ["mars","jupiter","saturn","uranus","neptune","pluto","node"]
OUT = os.path.expanduser("~/.artaquest-dev/artacomp/scidist"); os.makedirs(OUT, exist_ok=True)

usable = np.where(tv.sum(0) >= 20)[0]
T = np.arange(usable[0], usable[-1])                      # label needs t+1
S = np.clip(Yv, 0, None) * tv
cut = int(len(T)*0.8); WALL_Y = int(years[T[cut]])
print(f"years {years[T[0]]}..{years[T[-1]]} · wall {WALL_Y} · train <{WALL_Y}, test {WALL_Y}..{years[T[-1]]}")

rows = []
for t in T:
    js = np.where(tv[:, t] & tv[:, t+1])[0]
    if len(js) < 20: continue
    for j in js:
        rows.append((int(j), int(years[t]), float(S[j, t]), int(S[j, t+1] > S[j, t])))
tr = [r for r in rows if r[1] < WALL_Y]; te = [r for r in rows if r[1] >= WALL_Y]
print(f"rows {len(rows)} · train {len(tr)} (balance {np.mean([r[3] for r in tr]):.3f}) · "
      f"test {len(te)} (balance {np.mean([r[3] for r in te]):.3f})")

with open(f"{OUT}/train.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["id","field","year","share","target"])
    for i,r in enumerate(tr): w.writerow([f"tr_{i}", names[r[0]], r[1], round(r[2],6), r[3]])
with open(f"{OUT}/test.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["id","field","year"])
    for i,r in enumerate(te): w.writerow([f"te_{i}", names[r[0]], r[1]])
with open(f"{OUT}/sample_submission.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["id","target"])
    for i in range(len(te)): w.writerow([f"te_{i}", 0.5])
tey = np.array([r[1] for r in te]); pcut = int(np.quantile(tey, 0.35))
with open(f"{OUT}/solution.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["id","target","Usage"])
    for i,r in enumerate(te): w.writerow([f"te_{i}", r[3], "Public" if r[1] < pcut else "Private"])
# the full pie, train years only (the test span is withheld entirely)
with open(f"{OUT}/shares.csv","w",newline="") as f:
    w=csv.writer(f); ycols=[int(years[t]) for t in T if years[t] < WALL_Y]
    w.writerow(["field"]+ycols)
    for j,nm in enumerate(names):
        w.writerow([nm]+[round(float(S[j, list(years).index(y)]),6) for y in ycols])
with open(f"{OUT}/ephemeris.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["year"]+[f"{b}_lon_deg" for b in BODY])
    allyr = list(range(1700, 2056))
    for y in allyr:
        k = y - 1700
        if k >= TH.shape[0]: break
        w.writerow([y]+[round(float(np.rad2deg(TH[k,i]) % 360),4) for i in range(7)])
README = f"""# Science-Distribution: does the sky time what science studies?

How the world's attention divides across **251 research fields**, 1858-2025, and whether each
field's slice of the pie grows.

## The task
For each (field, year) in `test.csv`, predict the probability that the field holds a **larger share
of citations next year than this year**. Trending is relative by construction: a field only gains
slice by outgrowing the field as a whole.

## The rule that makes this a test of astrology
At prediction time you may use **the field's identity and the date, and nothing else**. `shares.csv`
stops at the wall, so the field's recent state in a test year is not available. Every feature you
build is therefore a function of a birth chart and a transiting sky.

## The metric: mAUC
One ROC-AUC per **field**, across that field's own test years, averaged over fields. This is
deliberate. A per-field constant — "this field usually grows" — scores exactly 0.5, because it
cannot order a field's own years. The model has to say *when*.

## The wall
The last 20% of years ({WALL_Y}-{int(years[T[-1]])}) are held out. Every training year precedes every
test year. Choose hyper-parameters on the training span only; the wall is not a validation set.

## Files
| file | what |
|---|---|
| `train.csv` | id, field, year, share, target — every year before {WALL_Y} |
| `test.csv` | id, field, year — the held-out span, no share and no label |
| `shares.csv` | the full pie, fields x years, train years only |
| `ephemeris.csv` | ecliptic longitude of Mars, Jupiter, Saturn, Uranus, Neptune, Pluto and the lunar node, 1700-2055 |
| `sample_submission.csv` | id, target |

## What is already known
Published baselines on this exact split, all selected on train and scored on the held-out span:
a per-field constant scores **0.5000** by construction; a purely calendar feature (a transiting
position, identical for every field in a year) scores **0.5275**; and the best of 3,683 classical
astrological and numerological features selected by train performance scores **0.5061**, with
corr(train, held) = **+0.024** across the whole catalogue. Beating 0.5275 with a model selected
honestly on train is an open problem.

Data derived from OpenAlex (CC0). Ephemeris computed from standard orbital elements.
No causal claims are made.
"""
open(f"{OUT}/README.md","w").write(README)
json.dump({"wall": WALL_Y, "train_rows": len(tr), "test_rows": len(te), "fields": J,
           "first_year": int(years[T[0]]), "last_year": int(years[T[-1]]),
           "train_balance": float(np.mean([r[3] for r in tr])),
           "test_balance": float(np.mean([r[3] for r in te]))},
          open(f"{OUT}/stats.json","w"), indent=1)
print("written to", OUT)
