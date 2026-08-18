#!/usr/bin/env python3
"""SCIENCE-DISTRIBUTION v3 — one row per year; the columns are the pie (operator 2026-08-18).

    train.csv   year, <251 field columns>      each row sums to 1: that year's citation distribution
    test.csv    year                           the last 20% of years — predict the whole row
    sample_submission.csv  year, <251 columns> uniform 1/251
    solution.csv (private)  the true rows

The input is the year and nothing else. Every training year precedes every test year.

  python3 analysis/arxivtopics/competition/build_science_distribution_v3.py
"""
import os, sys, json, csv, re, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import arxiv_fit as af

names, Yv, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
tv = af.META["topic_valid"]
years = np.array([int(y) for y in labels]); J, n = Yv.shape
BODY = ["mars","jupiter","saturn","uranus","neptune","pluto","node"]
OUT = os.path.expanduser("~/.artaquest-dev/artacomp/scidist3"); os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    if f.endswith(".csv"): os.remove(os.path.join(OUT, f))

def col(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")
COLS = [col(nm) for nm in names]
assert len(set(COLS)) == J, "field column names collide"

usable = np.where(tv.sum(0) >= 20)[0]
T = np.arange(usable[0], usable[-1] + 1)
S = np.clip(Yv, 0, None) * tv
S = S / np.maximum(S.sum(0, keepdims=True), 1e-12)               # each year sums to 1
cut = int(len(T) * 0.8); WALL_Y = int(years[T[cut]])
TR, TE = T[:cut], T[cut:]
print(f"years {years[T[0]]}..{years[T[-1]]} · train {years[TR[0]]}-{years[TR[-1]]} ({len(TR)} rows) · "
      f"TEST {years[TE[0]]}-{years[TE[-1]]} ({len(TE)} rows)")

def write_wide(path, idx, with_values=True):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year"] + (COLS if with_values else []))
        for t in idx:
            row = [int(years[t])]
            if with_values:
                v = S[:, t]; v = v / v.sum()
                row += [f"{x:.8f}" for x in v]
            w.writerow(row)
write_wide(f"{OUT}/train.csv", TR)
write_wide(f"{OUT}/test.csv", TE, with_values=False)
write_wide(f"{OUT}/solution.csv", TE)
with open(f"{OUT}/sample_submission.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["year"] + COLS)
    for t in TE: w.writerow([int(years[t])] + [f"{1.0/J:.8f}"] * J)
with open(f"{OUT}/ephemeris.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["year"] + [f"{b}_lon_deg" for b in BODY])
    for k, y in enumerate(range(1700, 1700 + TH.shape[0])):
        w.writerow([y] + [round(float(np.rad2deg(TH[k,i]) % 360), 4) for i in range(7)])
with open(f"{OUT}/fields.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["column", "field"])
    for c, nm in zip(COLS, names): w.writerow([c, nm])
# checks
tr = np.loadtxt(f"{OUT}/train.csv", delimiter=",", skiprows=1)
print(f"train shape {tr.shape} · row sums min {tr[:,1:].sum(1).min():.6f} max {tr[:,1:].sum(1).max():.6f}")
README = f"""# Science-Distribution v3: given the year, predict the pie

**Input: the year. Output: that year's distribution of the world's citations over 251 research fields.**

| file | shape | what |
|---|---|---|
| `train.csv` | {len(TR)} rows × 252 | `year` + 251 field columns; **each row sums to 1** — the pie, {years[TR[0]]}–{years[TR[-1]]} |
| `test.csv` | {len(TE)} rows × 1 | `year` only — the last 20% of years, {years[TE[0]]}–{years[TE[-1]]}; predict the whole row |
| `sample_submission.csv` | {len(TE)} × 252 | uniform 1/251 |
| `ephemeris.csv` | 1700–2055 | ecliptic longitudes of Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, lunar node |
| `fields.csv` | 251 | column name → full field name |

The year is the only input. A model that reaches the pie through the sky (`ephemeris.csv` is a function
of the year) is an astrological model; one that uses the year directly is a trend model. Both are
allowed — that is the question the benchmark asks. Every training year precedes every test year.

## Baselines on this split (train-selected, scored on the held-out years)
| model | cross-entropy (nats) | rank ρ |
|---|---|---|
| uniform | 5.5255 | — |
| the train-mean pie (climatology) | 5.1324 | 0.810 |
| sky softmax, selected on train | 5.3499 | 0.848 |
| carry-forward (has memory; reference only) | 4.8715 | 0.946 |

Data derived from OpenAlex (CC0). No causal claims are made.
"""
open(f"{OUT}/README.md", "w").write(README)
json.dump({"wall": WALL_Y, "train_years": [int(years[TR[0]]), int(years[TR[-1]])], "test_years": [int(years[TE[0]]), int(years[TE[-1]])],
           "fields": J, "columns": COLS}, open(f"{OUT}/stats.json", "w"), indent=1)
print("written to", OUT)
