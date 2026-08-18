#!/usr/bin/env python3
"""DAILY arXiv series per category, from the Cornell metadata snapshot (operator 2026-08-18).

Why arXiv and not OpenAlex: OpenAlex stores year-precision dates as YYYY-01-01 for about half of
every era, and citations carry no date at all — "daily citation share" is not a quantity that
exists. arXiv records the exact submission timestamp of every paper since 1991, so daily
submission counts per category are the reliable daily record of what science is doing.

Per category (primary category of each paper, submission day = versions[0].created):
  daily.csv          date, <one column per category>: submissions that day
  reliable_from.csv  category, first day after which the series is continuously active
                     (no 30-day window with zero submissions from that day to the end)
Categories kept: those with >= 8 years of reliable daily record.

  python3 analysis/arxivtopics/competition/daily_build.py
"""
import os, json, gzip, csv, sys, collections, datetime as dt
import numpy as np
SRC = os.path.expanduser("~/.artaquest-dev/arxiv-daily/arxiv-metadata-oai-snapshot.json")
OUT = os.path.expanduser("~/.artaquest-dev/artacomp/daily"); os.makedirs(OUT, exist_ok=True)
MONTH = {m: i for i, m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1)}
def day_of(created):
    # "Mon, 2 Apr 2007 19:18:42 GMT"
    p = created.split(); return dt.date(int(p[3]), MONTH[p[2]], int(p[1]))
counts = collections.defaultdict(lambda: collections.Counter())
n = 0; bad = 0
with open(SRC) as f:
    for line in f:
        n += 1
        try:
            r = json.loads(line); cat = r["categories"].split()[0]; d = day_of(r["versions"][0]["created"])
        except Exception: bad += 1; continue
        counts[cat][d] += 1
        if n % 500000 == 0: print(f"  {n:,} papers", flush=True)
print(f"papers {n:,} · unparsable {bad} · categories {len(counts)}", flush=True)
d0 = min(min(c) for c in counts.values()); d1 = max(max(c) for c in counts.values())
days = [d0 + dt.timedelta(k) for k in range((d1 - d0).days + 1)]
DI = {d: i for i, d in enumerate(days)}
cats = sorted(counts)
M = np.zeros((len(days), len(cats)), np.int32)
for j, c in enumerate(cats):
    for d, k in counts[c].items(): M[DI[d], j] = k
# reliable-from: first day such that every 30-day window from there to the end has >= 1 submission
rel = {}
for j, c in enumerate(cats):
    x = M[:, j]; W = 30
    if x.sum() < 2000: continue
    csum = np.concatenate([[0], np.cumsum(x)])
    win = csum[W:] - csum[:-W]                     # win[i] = sum x[i:i+W]
    ok = win > 0
    # find the earliest i such that ok[i:] all true
    bad_idx = np.where(~ok)[0]
    start = 0 if len(bad_idx) == 0 else int(bad_idx[-1]) + 1
    if len(days) - start >= 8*365: rel[c] = days[start]
print(f"categories with >= 8 reliable years: {len(rel)}", flush=True)
keep = [c for c in cats if c in rel]
with open(f"{OUT}/daily.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["date"] + keep)
    for i, d in enumerate(days): w.writerow([d.isoformat()] + [int(M[i, cats.index(c)]) for c in keep])
with open(f"{OUT}/reliable_from.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["category", "reliable_from", "reliable_days", "total_submissions"])
    for c in keep: w.writerow([c, rel[c].isoformat(), (d1 - rel[c]).days, int(M[:, cats.index(c)].sum())])
print(f"daily.csv: {len(days)} days × {len(keep)} categories · {d0} → {d1}", flush=True)
