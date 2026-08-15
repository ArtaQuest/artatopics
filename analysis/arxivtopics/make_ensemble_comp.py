#!/usr/bin/env python3
"""THE ASTRO-ENSEMBLE COMPETITION BUNDLE (operator 2026-08-15: "launch your own dataset and
competition and benchmark to find the best ensemble astrology stack").

Aligned with the Foundation's own competition machinery (AQ\\Competitions):
  public/train.csv           field,year,share,works — every field's history through 1995, from its
                             own emergence year (its pre-existence years are absent, not zero)
  public/test.csv            field,year — the 251x30 = 7,530 rows to predict (1996..2025)
  public/ephemeris.csv       year + seven sidereal longitudes, 1700..2055 — the ONLY feature
                             permitted at prediction time; it is known centuries ahead
  public/sample_submission.csv   trend,date,target — carry-forward, the honest do-nothing entry
  public/reference_solution.py   the same baseline as runnable, reproducible code (the default
                             code_url every submission must carry)
  public/RULES.md            the benchmark protocol, stated plainly
  solution/solution.php      the guarded holdout map {"field|year": share} — never web-served
  stats.json                 n_train / n_test / n_features / n_targets for dataset_stats()

Metric: the platform's 'r2' — per-field R^2 over the 30 hidden years against the HOLDOUT-mean
baseline, averaged across fields. Stricter than the campaign's train-mean skill: a flat guess at the
field's past level scores NEGATIVE wherever the field drifted.

  python3 analysis/arxivtopics/make_ensemble_comp.py [outdir]
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import arxiv_fit as af

OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.artaquest-dev/artacomp/bundle"))
names, Yv, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
n = Yv.shape[1]
WALL = n - 30
starts = af.META["topic_valid"].argmax(1)
years = [int(y) for y in labels]
years_ext = [int(y) for y in (labels + future)]

import re, unicodedata
def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")
SLUGS = [slug(nm) for nm in names]
assert len(set(SLUGS)) == len(SLUGS), "field slugs must be unique"

# works matrix aligned by subfield_id (the trending task's alignment, reused)
import pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_w = pd.read_csv(os.path.join(REPO, "analysis/citations/rail_works_yearly.csv"))
_c = pd.read_csv(os.path.join(REPO, "analysis/citations/citations_received_yearly.csv"))
_w = _w.set_index("subfield_id").loc[_c["subfield_id"]].reset_index()
W = _w[[c for c in _w.columns if c[:1].isdigit()][:n]].to_numpy(float)

os.makedirs(os.path.join(OUT, "public"), exist_ok=True)
os.makedirs(os.path.join(OUT, "solution"), exist_ok=True)

n_train = 0
with open(os.path.join(OUT, "public/train.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["field", "year", "share", "works"])
    for j, nm in enumerate(names):
        for t in range(int(starts[j]), WALL):
            w.writerow([SLUGS[j], years[t], round(float(Yv[j, t]), 6), int(W[j, t])]); n_train += 1

with open(os.path.join(OUT, "public/test.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["field", "year"])
    for j in range(len(names)):
        for t in range(WALL, n):
            w.writerow([SLUGS[j], years[t]])

with open(os.path.join(OUT, "public/ephemeris.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["year"] + [f"{b}_lon_deg" for b in af.BODIES])
    for t, yr in enumerate(years_ext):
        w.writerow([yr] + [round(float(np.degrees(TH[t, i]) % 360), 4) for i in range(TH.shape[1])])

# carry-forward sample submission (the do-nothing entry, and the bar to beat)
with open(os.path.join(OUT, "public/sample_submission.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["trend", "date", "target"])
    for j in range(len(names)):
        last = float(Yv[j, WALL - 1])
        for t in range(WALL, n):
            w.writerow([SLUGS[j], years[t], round(last, 6)])

sol = {f"{SLUGS[j]}|{years[t]}": round(float(Yv[j, t]), 6)
       for j in range(len(names)) for t in range(WALL, n)}
with open(os.path.join(OUT, "solution/solution.php"), "w") as f:
    f.write("<?php\nif ( ! defined( 'AQ_COMP_SOLUTION' ) ) { exit; }\nreturn " )
    f.write(json.dumps(sol).replace("{", "[", 1)[::-1].replace("}", "]", 1)[::-1]
            .replace('":', '" =>').replace(',"', ',"'))
    f.write(";\n")

with open(os.path.join(OUT, "stats.json"), "w") as f:
    json.dump({"n_train": n_train, "n_test": len(sol), "n_features": 1 + TH.shape[1], "n_targets": len(names)}, f)

with open(os.path.join(OUT, "public/reference_solution.py"), "w") as f:
    f.write('''#!/usr/bin/env python3
"""Reference solution — carry-forward. The do-nothing bar every model must beat.
Reads train.csv + test.csv beside it, writes submission.csv (trend,date,target)."""
import csv, collections
last = {}
for r in csv.DictReader(open("train.csv")):
    y = int(r["year"]); f = r["field"]
    if f not in last or y > last[f][0]: last[f] = (y, r["share"])
with open("submission.csv", "w", newline="") as out:
    w = csv.writer(out); w.writerow(["trend", "date", "target"])
    for r in csv.DictReader(open("test.csv")):
        w.writerow([r["field"], r["year"], last[r["field"]][1]])
print("submission.csv written")
''')

with open(os.path.join(OUT, "public/RULES.md"), "w") as f:
    f.write("""# The Astro-Ensemble Stack — 251 fields, 30 hidden years

Predict each research field's share of the world's citations for every year 1996-2025, given its
history to 1995 and the positions of seven celestial bodies (known for any year, past or future).

- **Score**: per-field R^2 over the 30 hidden years, averaged over the 251 fields.
- **The sky-only rule**: at prediction time a model may read the ephemeris and nothing else. The
  history is for FITTING. Ensembles of sky models are the point of this benchmark — stack anything,
  as long as every member forecasts from the ephemeris alone.
- **Every submission carries runnable public code** (code_url) — the platform's standing rule: a
  score nobody can reproduce is not a score.
- **The bar**: carry-forward (sample_submission.csv). A model below it has learned nothing.
- Reference numbers from the open benchmark (github.com/ArtaQuest/artatopics): carry-forward 0.73,
  the per-field receiver 0.80, on the pooled variant of this metric.
""")

sz = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(OUT) for fn in fns)
print(f"bundle at {OUT}: train {n_train} rows · test {len(sol)} rows · {sz//1024}KB")
print("PHP solution guard check:", open(os.path.join(OUT, "solution/solution.php")).read(120).replace("\n", " "))
