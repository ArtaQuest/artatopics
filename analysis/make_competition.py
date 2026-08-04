#!/usr/bin/env python3
"""
Build the FIRST ArtaQuest competition dataset — "sky-vs-search".

TASK. Given the 11 sidereal bodies' ecliptic longitudes (degrees) each month, predict a skill's
Google-Trends search interest that month.

FEATURES per datapoint (one month): the 11 bodies' longitudes from the shared monthly ephemeris —
    date, sun, mercury, venus, mars, jupiter, saturn, uranus, neptune, pluto, chiron, node
TARGET: the trend's monthly interest value (tf.load_y — already recency-trimmed by DROP_LAST=12
    months; it is the identity-stabilised value the site fits, which is fine as the target).

TRENDS: the axis=house SKILLS in analysis/_fields_weekly.json (a clean, bounded set — NOT all 1034).

SPLIT (by time): within each trend's already-recency-trimmed series, the LAST 24 months = TEST
    (holdout, target hidden); everything before = TRAIN.

EMITS (under wp-content/uploads/competitions/sky-vs-search/):
    train.csv               long: trend,date,sun,…,node,target        (all train rows, all trends)
    test.csv                long: trend,date,sun,…,node               (holdout rows, NO target)
    sample_submission.csv   trend,date,target(=0)                      (the submission template)
    solution.json           { "trend|date": true_target, … }          SERVER-ONLY (never in the DB,
                                                                        never git-committed)

Run from the repo root:  python3 analysis/make_competition.py
(or from anywhere — it chdirs to the repo root so trends_fit's relative paths resolve).
"""
import os, sys, csv, json

# trends_fit.py uses paths relative to the repo root (e.g. "analysis/data_monthly"), so anchor there.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import trends_fit as tf

SLUG = "sky-vs-search"
TEST_MONTHS = 24                      # last 2 years of each series = the hidden holdout
FIELDS_JSON = "analysis/_fields_weekly.json"
OUT_DIR = os.path.join(REPO, "wp-content", "uploads", "competitions", SLUG)

# Column order for the feature matrix — the 11 bodies, in tf.BODIES order.
BODIES = list(tf.BODIES)              # sun,mercury,venus,mars,jupiter,saturn,uranus,neptune,pluto,chiron,node
FEATURE_COLS = ["date"] + BODIES


def house_labels():
    """The axis=house skills from _fields_weekly.json, as their tf.load_y label. Deduped, sorted."""
    with open(FIELDS_JSON) as f:
        fields = json.load(f)
    labels = []
    seen = set()
    for key, rec in fields.items():
        if not isinstance(rec, dict) or rec.get("axis") != "house":
            continue
        label = rec.get("label") or key
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return sorted(labels)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Monthly ephemeris on tf.GRID → per-date longitudes, keyed by 'YYYY-MM-DD' (month start).
    lon = tf.ephemeris()
    missing = [b for b in BODIES if b not in lon]
    if missing:
        sys.exit(f"ephemeris missing bodies {missing} — cannot build features")
    grid = pd.DatetimeIndex(tf.GRID)
    date_str = {i: grid[i].strftime("%Y-%m-%d") for i in range(len(grid))}
    # date -> {body: longitude}
    feat_by_date = {}
    for i in range(len(grid)):
        feat_by_date[date_str[i]] = {b: float(lon[b][i]) for b in BODIES}

    labels = house_labels()
    print(f"[{len(labels)} axis=house skills] building {SLUG} — last {TEST_MONTHS} months = holdout")

    train_rows = []       # dicts: trend,date,<bodies>,target
    test_rows = []        # dicts: trend,date,<bodies>
    solution = {}         # "trend|date" -> true target
    n_targets = 0
    skipped = 0

    for label in labels:
        times, y = tf.load_y(label)
        if y is None or times is None or len(y) == 0:
            skipped += 1
            continue
        times = pd.DatetimeIndex(pd.to_datetime(times))
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n <= TEST_MONTHS + 1:      # need at least some train rows beyond the holdout
            skipped += 1
            continue
        trend = tf.slug(label)
        n_targets += 1
        split = n - TEST_MONTHS
        for j in range(n):
            d = times[j].strftime("%Y-%m-%d")
            feats = feat_by_date.get(d)
            if feats is None:
                # Series month not on the ephemeris grid (shouldn't happen — both are tf.GRID) — skip row.
                continue
            base = {"trend": trend, "date": d}
            for b in BODIES:
                base[b] = round(feats[b], 4)
            if j < split:
                base["target"] = round(float(y[j]), 6)
                train_rows.append(base)
            else:
                test_rows.append(base)                    # NO target in the public test set
                solution[f"{trend}|{d}"] = round(float(y[j]), 6)

    # ── write train.csv (long format, all trends) ─────────────────────────────
    train_path = os.path.join(OUT_DIR, "train.csv")
    with open(train_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trend"] + FEATURE_COLS + ["target"])
        for r in train_rows:
            w.writerow([r["trend"]] + [r[c] for c in FEATURE_COLS] + [r["target"]])

    # ── write test.csv (long format, NO target) ───────────────────────────────
    test_path = os.path.join(OUT_DIR, "test.csv")
    with open(test_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trend"] + FEATURE_COLS)
        for r in test_rows:
            w.writerow([r["trend"]] + [r[c] for c in FEATURE_COLS])

    # ── write sample_submission.csv (trend,date,target=0) ─────────────────────
    sample_path = os.path.join(OUT_DIR, "sample_submission.csv")
    with open(sample_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trend", "date", "target"])
        for r in test_rows:
            w.writerow([r["trend"], r["date"], 0])

    # ── write solution.json (SERVER-ONLY — the hidden holdout answers) ────────
    sol_path = os.path.join(OUT_DIR, "solution.json")
    with open(sol_path, "w") as f:
        json.dump(solution, f)

    stats = {
        "slug": SLUG,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "n_features": len(BODIES),     # 11 bodies
        "n_targets": n_targets,        # distinct trends
        "test_months": TEST_MONTHS,
        "skipped": skipped,
    }
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  trends used : {n_targets}  (skipped {skipped})")
    print(f"  n_train     : {len(train_rows)} rows")
    print(f"  n_test      : {len(test_rows)} rows (holdout)")
    print(f"  n_features  : {len(BODIES)}  n_targets: {n_targets}")
    print(f"  wrote       : {train_path}")
    print(f"                {test_path}")
    print(f"                {sample_path}")
    print(f"                {sol_path}  (server-only — do NOT commit)")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
