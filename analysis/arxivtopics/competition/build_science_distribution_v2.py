#!/usr/bin/env python3
"""SCIENCE-DISTRIBUTION v2 — shuffled, with the calendar hidden (operator 2026-08-18).

What changed from v1 and why:
  * Rows are SHUFFLED with a fixed seed; row order carries nothing.
  * There is NO year column, no age, no founding year, no share. A row is (id, field, sky) only.
  * The sky is given RELATIVE: for each of the seven bodies, the transiting longitude minus that
    field's NATAL longitude (the sky at its founding), folded to [0, 360). "Saturn stands 90 degrees
    past where it stood at your founding" — a horoscope reading, not a date.
  * The natal sky itself is NOT shipped. So a row cannot be dated by reading Neptune+Pluto off an
    absolute longitude, and cannot be aged by counting returns from a known founding.

What is still true, stated plainly rather than hidden: seven relative angles from seven bodies with
seven different periods still constrain the elapsed time since founding, and a determined competitor
who guesses a field's founding year could in principle recover the date. The calendar shortcut is
made EXPENSIVE, not impossible — the ephemeris is public and orbital mechanics is invertible. What
this design does guarantee is that no cheap column (year, age, order) leaks it, so any model that
scores must have gone through the sky.

Metric mAUC (per-field AUC averaged): a per-field constant scores 0.5. Wall unchanged (last 20% of
years); the split is by DATE even though the date is hidden, so train strictly precedes test.

  python3 analysis/arxivtopics/competition/build_science_distribution_v2.py
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
BODY_ALL = ["mars","jupiter","saturn","uranus","neptune","pluto","node"]
SHIP = [0, 1, 2, 6]                    # mars, jupiter, saturn, node — see README for why
BODY = [BODY_ALL[i] for i in SHIP]
OUT = os.path.expanduser("~/.artaquest-dev/artacomp/scidist2"); os.makedirs(OUT, exist_ok=True)
rng = np.random.RandomState(20260818)

usable = np.where(tv.sum(0) >= 20)[0]
T = np.arange(usable[0], usable[-1])
S = np.clip(Yv, 0, None) * tv
cut = int(len(T)*0.8); WALL_Y = int(years[T[cut]])
start = tv.argmax(1)                                    # each field's first active index = founding
rows = []
for t in T:
    js = np.where(tv[:, t] & tv[:, t+1])[0]
    if len(js) < 20: continue
    for j in js:
        rel = (np.rad2deg(TH[t, SHIP] - TH[start[j], SHIP]) % 360.0)
        rows.append((int(j), int(years[t]), rel, int(S[j, t+1] > S[j, t])))
tr = [r for r in rows if r[1] < WALL_Y]; te = [r for r in rows if r[1] >= WALL_Y]
rng.shuffle(tr); rng.shuffle(te)
print(f"wall {WALL_Y} · train {len(tr)} (balance {np.mean([r[3] for r in tr]):.3f}) · "
      f"test {len(te)} (balance {np.mean([r[3] for r in te]):.3f}) · shuffled")

# opaque ids that carry no order
# opaque, collision-free ids: a random permutation of a large range, so no id encodes order or split
pool = rng.permutation(10**7)[:len(tr)+len(te)] + 10**7
tr_ids = [f"r{v}" for v in pool[:len(tr)]]; te_ids = [f"r{v}" for v in pool[len(tr):]]
assert len(set(tr_ids)|set(te_ids)) == len(tr)+len(te)
hdr = ["id","field"] + [f"{b}_rel_deg" for b in BODY]
with open(f"{OUT}/train.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(hdr+["target"])
    for i,r in zip(tr_ids,tr): w.writerow([i, names[r[0]]] + [round(float(x),3) for x in r[2]] + [r[3]])
with open(f"{OUT}/test.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(hdr)
    for i,r in zip(te_ids,te): w.writerow([i, names[r[0]]] + [round(float(x),3) for x in r[2]])
with open(f"{OUT}/sample_submission.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["id","target"])
    for i in te_ids: w.writerow([i, 0.5])
tey = np.array([r[1] for r in te]); pcut = int(np.quantile(tey, 0.35))
with open(f"{OUT}/solution.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["id","target","Usage"])
    for i,r in zip(te_ids,te): w.writerow([i, r[3], "Public" if r[1] < pcut else "Private"])
# private key: id -> (field, year), NEVER shipped — for the host's own audits
with open(f"{OUT}/PRIVATE_key.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["id","field","year","split"])
    for i,r in zip(tr_ids,tr): w.writerow([i, names[r[0]], r[1], "train"])
    for i,r in zip(te_ids,te): w.writerow([i, names[r[0]], r[1], "test"])
README = f"""# Science-Distribution v2: does the sky time what science studies?

Each row is one (field, moment) and asks: **does this field hold a larger share of the world's
citations next year than now?** 251 fields, {len(rows):,} rows, shuffled.

## What a row contains — and what it does not
`id, field, mars_rel_deg, jupiter_rel_deg, saturn_rel_deg, node_rel_deg [, target]`

Each angle is the **transiting body read against the field's own natal sky**: its longitude now
minus its longitude at the field's founding, folded to 0–360°. Saturn at 90° means Saturn stands a
quarter-turn past where it stood when the field was born. This is a horoscope reading — transit
against natal — and it is the only input.

There is **no year, no age, no founding date, no share, no natal chart, and no row order**: rows are
shuffled and ids are a random permutation.

## Why only Mars, Jupiter, Saturn and the lunar node
We measured how well the DATE can be recovered from the angles alone (random forest, held rows):

| bodies shipped | median date error | within 5 years |
|---|---|---|
| all seven | 0.3 y | 97% |
| Uranus + Neptune + Pluto alone | 0.3 y | 97% |
| **Mars + Jupiter + Saturn + node** | **10.3 y** | **28%** |
| Mars + Jupiter + Saturn | 16.7 y | 17% |

The outer planets ARE the calendar: seven bodies with seven different periods, read against a fixed
natal chart, form a clock, and Uranus/Neptune/Pluto alone date any row to within a year. Every
"signal" those bodies showed in the open v1 benchmark turned out to be the era, and the strongest of
them flipped sign across the wall. So v2 ships the classical TIMING planets — the ones a horoscope
actually reads for *when* — and withholds the three that merely tell you *which decade*. The date is
fuzzy to a decade at best. That is the point.

## The metric: mAUC
One ROC-AUC per **field** over that field's own rows, averaged over fields. A per-field constant
("this field usually grows") scores exactly 0.5. The model must say *when*.

## The split
Train and test are separated by **date** even though the date is hidden: every training row precedes
every test row in time (the last 20% of years are held out). Public/private is likewise temporal.

## Published baselines on the open v1 (same rows, calendar visible)
Per-field constant 0.5000 · best purely-calendar feature 0.5275 · best of 3,683 astrological and
numerological features selected by train performance 0.5061 (corr(train, held) = +0.024).
Beating 0.5 honestly is the open problem.

Data derived from OpenAlex (CC0). No causal claims are made.
"""
open(f"{OUT}/README.md","w").write(README)
json.dump({"wall": WALL_Y, "train_rows": len(tr), "test_rows": len(te), "fields": J, "shuffled": True,
           "seed": 20260818, "columns": hdr}, open(f"{OUT}/stats.json","w"), indent=1)
print("written to", OUT)
