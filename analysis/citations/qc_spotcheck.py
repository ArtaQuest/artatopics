#!/usr/bin/env python3
"""QC SPOT-CHECK (operator 2026-07-22): our snapshot-derived quarterly cells vs the live API.

Draws N random (topic, year, quarter) cells from openalex_topic_quarter_cohort.csv.gz
(recent-ish years, q1..q4, works>=50) and asks the live OpenAlex API for the same cell:
  filter=primary_topic.id:T...,from_publication_date:...,to_publication_date:...
  &per-page=1&select=id&cited_by_count_sum=true
comparing meta.count vs our works and meta.cited_by_count_sum vs our cited_by_sum.

Differences are EXPECTED to be small but nonzero: the API moved past the 2026-06-26
snapshot, and q1..q4 cells exclude Jan-1-dated works (year-only precision) while the
API window for Q1 includes Jan 1 — so Q1 checks use from=Jan 02 to match our binning.
Keyless budget is ~1,000 list calls/day; default N=30 uses 30.

  python3 analysis/citations/qc_spotcheck.py [--n 30]
"""
import argparse, csv, gzip, json, random, subprocess, time

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
QSTART = {1: "01-02", 2: "04-01", 3: "07-01", 4: "10-01"}   # Q1 starts Jan 2: q0 holds Jan 1
QEND = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    rows = []
    with gzip.open(f"{HERE}/openalex_topic_quarter_cohort.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if (r["topic_id"] != "none" and r["quarter"] != "0"
                    and 1990 <= int(r["year"]) <= 2024 and int(r["works"]) >= 50):
                rows.append(r)
    random.seed(20260722)
    sample = random.sample(rows, min(args.n, len(rows)))
    print(f"[qc] {len(sample)} cells vs live API")
    ok = 0
    wd, cd = [], []
    for r in sample:
        t, y, q = r["topic_id"], r["year"], int(r["quarter"])
        url = (f"https://api.openalex.org/works?filter=primary_topic.id:T{t},"
               f"from_publication_date:{y}-{QSTART[q]},to_publication_date:{y}-{QEND[q]}"
               f"&per-page=1&select=id&cited_by_count_sum=true")
        try:
            m = json.loads(subprocess.check_output(["curl", "-sS", "-m", "30", url]))["meta"]
        except Exception as e:  # noqa: BLE001
            print(f"  T{t} {y}q{q}: API error {e}")
            time.sleep(2)
            continue
        w_ours, w_api = int(r["works"]), m["count"]
        c_ours, c_api = int(r["cited_by_sum"]), m.get("cited_by_count_sum", -1)
        dw = abs(w_ours - w_api) / max(w_api, 1)
        dc = abs(c_ours - c_api) / max(c_api, 1)
        wd.append(dw); cd.append(dc)
        flag = "OK " if dw < 0.03 and dc < 0.10 else "DIFF"
        if flag == "OK ":
            ok += 1
        print(f"  {flag} T{t} {y}q{q} [{r['topic'][:34]}]: works {w_ours} vs {w_api} "
              f"({dw * 100:.1f}%) · cites {c_ours} vs {c_api} ({dc * 100:.1f}%)")
        time.sleep(1.2)
    if wd:
        wd.sort(); cd.sort()
        print(f"[qc] {ok}/{len(wd)} OK · median dev works {wd[len(wd) // 2] * 100:.2f}% "
              f"cites {cd[len(cd) // 2] * 100:.2f}%")


if __name__ == "__main__":
    main()
