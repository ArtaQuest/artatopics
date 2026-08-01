#!/usr/bin/env python3
"""Sidereal (Lahiri) ephemeris tables for the arXiv fits — sampled at the MID-POINT of each period
(operator 2026-07-21): weekly rows are labelled by the week's Monday but the sky is computed at
Thursday 12:00 UT (start + 3.5 days); monthly rows are labelled YYYY-MM and computed at the 15th,
12:00 UT. Chiron needs analysis/_swisseph/seas_18.se1.

  python3 analysis/arxivtopics/build_ephemeris.py
"""
import os
import numpy as np, pandas as pd
import swisseph as swe

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
swe.set_ephe_path("analysis/_swisseph")
swe.set_sid_mode(swe.SIDM_LAHIRI)
FLAGS = swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED
BODIES = {"sun": swe.SUN, "moon": swe.MOON, "mercury": swe.MERCURY, "venus": swe.VENUS,
          "mars": swe.MARS, "jupiter": swe.JUPITER, "saturn": swe.SATURN, "uranus": swe.URANUS,
          "neptune": swe.NEPTUNE, "pluto": swe.PLUTO, "node": swe.MEAN_NODE, "chiron": swe.CHIRON}


def sky_row(ts):
    jd = swe.julday(ts.year, ts.month, ts.day, 12.0)
    row = {}
    for b, code in BODIES.items():
        pos = swe.calc_ut(jd, code, FLAGS)[0]
        row[f"{b}_lon"] = round(pos[0] % 360.0, 4)
        row[f"{b}_dist"] = round(max(pos[2], 1e-6), 6)
    return row


def main():
    # WEEKLY: Mondays 2005-01-03 .. 2026-12-28, sky at Monday + 3.5 days ≈ Thursday noon.
    mons = pd.date_range("2005-01-03", "2026-12-28", freq="7D")
    rows = []
    for m in mons:
        r = sky_row(m + pd.Timedelta(days=3))          # Thursday 12:00 UT = week mid-point
        r["Time"] = m.strftime("%Y-%m-%d")
        rows.append(r)
    dfw = pd.DataFrame(rows).set_index("Time")
    dfw.to_csv("analysis/arxivtopics/_ephemeris_weekly.csv")
    print(f"  weekly: {len(dfw)} rows (labelled Monday, sampled Thursday noon)", flush=True)

    # MONTHLY: 2005-01 .. 2026-12, sky at the 15th 12:00 UT.
    rows = []
    for ts in pd.date_range("2005-01-01", "2026-12-01", freq="MS") + pd.Timedelta(days=14):
        r = sky_row(ts)
        r["Time"] = ts.strftime("%Y-%m")
        rows.append(r)
    dfm = pd.DataFrame(rows).set_index("Time")
    dfm.to_csv("analysis/arxivtopics/_ephemeris_monthly.csv")
    print(f"  monthly: {len(dfm)} rows (sampled the 15th, noon)", flush=True)

    # LUNAR (operator 2026-07-21): the sky at each exact FULL-MOON instant (_full_moons.csv jd).
    fm = pd.read_csv("analysis/arxivtopics/_full_moons.csv")
    rows = []
    for _, r in fm.iterrows():
        jd = float(r["jd"])
        row = {}
        for b, code in BODIES.items():
            pos = swe.calc_ut(jd, code, FLAGS)[0]
            row[f"{b}_lon"] = round(pos[0] % 360.0, 4)
            row[f"{b}_dist"] = round(max(pos[2], 1e-6), 6)
        row["Time"] = r["utc"][:10]
        rows.append(row)
    dfl = pd.DataFrame(rows).set_index("Time")
    dfl.to_csv("analysis/arxivtopics/_ephemeris_lunar.csv")
    print(f"  lunar: {len(dfl)} rows (sampled at the full-moon instant)", flush=True)

    # QUARTERLY (operator 2026-07-22): one row per calendar quarter, sampled at the quarter's
    # exact MID-POINT (start + half the quarter's length), noon UT.
    rows = []
    for yr in range(1700, 2040):
        for q in range(4):
            q0 = pd.Timestamp(yr, 1 + 3 * q, 1)
            q1 = pd.Timestamp(yr + (q == 3), 1 + 3 * ((q + 1) % 4), 1)
            r = sky_row(q0 + (q1 - q0) / 2)
            r["Time"] = f"{yr}Q{q + 1}"
            rows.append(r)
    dfq = pd.DataFrame(rows).set_index("Time")
    dfq.to_csv("analysis/arxivtopics/_ephemeris_quarterly.csv")
    print(f"  quarterly: {len(dfq)} rows (sampled at each quarter's mid-point, noon)", flush=True)
    # YEARLY (operator 2026-07-24): one row per year, sampled at the year's MID-POINT (July 2 noon).
    # Extended to 2056 (operator 2026-07-24: 30-year headline forecast reaches 2025+30 = 2055).
    rows = []
    for yr in range(1700, 2057):
        r = sky_row(pd.Timestamp(yr, 7, 2))
        r["Time"] = str(yr)
        rows.append(r)
    pd.DataFrame(rows).set_index("Time").to_csv("analysis/arxivtopics/_ephemeris_yearly.csv")
    print(f"  yearly: {len(rows)} rows (mid-year, 1700..2056)", flush=True)
    print("EPHEM DONE", flush=True)


if __name__ == "__main__":
    main()
