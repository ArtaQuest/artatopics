#!/usr/bin/env python3
"""comp_ephemeris.py — sidereal (Lahiri) daily ephemeris for the ArtaAstro competition, using the
SAME bodies as topic-500 PLUS the Moon: sun, moon, mercury, venus, mars, jupiter, saturn, uranus,
neptune, pluto, chiron, node (the mean lunar node) — 12 bodies. Cast at 10:00 America/New_York.

Covers every day of the cleaned GDELT dataset (1979 → present, as far back as GDELT goes), so the
model can train on the full history. Writes out/ephemeris_comp.csv (date + the 12 longitudes, deg).
"""
import os, warnings, datetime as dt
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from kerykeion import AstrologicalSubject
import swisseph as swe

HERE = os.path.dirname(os.path.abspath(__file__)); AST = os.path.dirname(HERE)
BODIES = ["sun","moon","mercury","venus","mars","jupiter","saturn","uranus","neptune","pluto","chiron","node"]
KER = {"sun":"sun","moon":"moon","mercury":"mercury","venus":"venus","mars":"mars","jupiter":"jupiter",
       "saturn":"saturn","uranus":"uranus","neptune":"neptune","pluto":"pluto","chiron":"chiron"}
swe.set_sid_mode(swe.SIDM_LAHIRI, 0, 0)

dates = pd.read_csv(os.path.join(HERE, "world_events_daily_clean.csv"), parse_dates=["date"])["date"]
rows = []
for i, d in enumerate(dates):
    s = AstrologicalSubject("d", d.year, d.month, d.day, 10, 0, lng=-74.0060, lat=40.7128,
                            tz_str="America/New_York", zodiac_type="Sidereal", sidereal_mode="LAHIRI", online=False)
    vals = [round(float(getattr(s, KER[b]).abs_pos) % 360.0, 4) for b in BODIES if b != "node"]
    node = round(float(swe.calc_ut(s.julian_day, swe.MEAN_NODE, swe.FLG_SWIEPH | swe.FLG_SIDEREAL)[0][0]) % 360.0, 4)
    rows.append([d.strftime("%Y-%m-%d")] + vals + [node])
    if (i + 1) % 4000 == 0:
        print(f"  {d.date()} ({i+1}/{len(dates)})", flush=True)
out = os.path.join(AST, "out", "ephemeris_comp.csv")
pd.DataFrame(rows, columns=["date"] + BODIES).to_csv(out, index=False)
print(f"wrote {out} ({len(rows)} days, bodies: {BODIES})")
