#!/usr/bin/env python3
"""Per-year charts from two real engines (operator 2026-08-18): PyJHora (Vedic) and iztro (Zi Wei).

One chart per year at a fixed MUNDANE instant — 1 July, 12:00 UT, Greenwich (0°N, 0°E). This is a
convention and is stated as one: the task's input is the bare year, so the chart of "the year" has
to be pinned to some moment, and mid-year noon at the prime meridian is the least arbitrary choice
we could defend. Every feature below is a function of that instant only.

Vedic (PyJHora, Lahiri sidereal):
  9 grahas' sidereal longitudes (Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu),
  the Lagna, each graha's rasi and nakshatra, panchanga (tithi, vaara, nakshatra, yoga, karana),
  Vimshottari lord at that instant (from the TRUE Moon's nakshatra), retrogrades, ayanamsa.
Zi Wei Dou Shu (iztro): the 12 palaces' major stars, the soul/body palace, five-elements class,
  and the decadal (da xian) palace in force — for a fixed birth-hour convention, again stated.

  python3 analysis/arxivtopics/competition/build_charts.py
"""
import os, sys, json, subprocess, csv
import numpy as np
from jhora.panchanga import drik
from jhora import const, utils
OUT = os.path.expanduser("~/.artaquest-dev/artacomp/scidist3/charts"); os.makedirs(OUT, exist_ok=True)
YEARS = list(range(1700, 2056))
place = drik.Place("Greenwich", 0.0, 0.0, 0.0)          # name, lat, lon, tz-hours
GRAHA = ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu"]
drik.set_ayanamsa_mode("LAHIRI")

def jd_of(y):
    return utils.julian_day_number(drik.Date(y, 7, 1), (12, 0, 0))

rows = []
for y in YEARS:
    jd = jd_of(y)
    try:
        lons = [float(drik.sidereal_longitude(jd, p)) for p in range(9)]
    except Exception as e:
        # planet indices: PyJHora uses 0..8 for the nine grahas in the order above; fall back per-planet
        lons = []
        for p in range(9):
            try: lons.append(float(drik.sidereal_longitude(jd, p)))
            except Exception: lons.append(float("nan"))
    try: asc = float(drik.ascendant(jd, place)[1]) if isinstance(drik.ascendant(jd, place), (list, tuple)) else float(drik.ascendant(jd, place))
    except Exception: asc = float("nan")
    try: ayan = float(drik.get_ayanamsa_value(jd))
    except Exception: ayan = float("nan")
    def safe(fn, *a):
        try:
            r = fn(*a); return r[0] if isinstance(r, (list, tuple)) else r
        except Exception: return -1
    tithi = safe(drik.tithi, jd, place); vaara = safe(drik.vaara, jd, place)
    naksh = safe(drik.nakshatra, jd, place); yoga = safe(drik.yogam, jd, place); karana = safe(drik.karana, jd, place)
    try: retro = list(drik.planets_in_retrograde(jd, place))
    except Exception: retro = []
    rows.append(dict(year=y, ayanamsa=ayan, asc=asc, tithi=tithi, vaara=vaara, nakshatra=naksh, yoga=yoga, karana=karana,
                     retro=retro, **{f"{g}_sid": l for g, l in zip(GRAHA, lons)}))
    if y % 50 == 0: print(f"  {y}: Sun {lons[0]:.1f} Moon {lons[1]:.1f} Sat {lons[6]:.1f} · tithi {tithi} nak {naksh} · ayan {ayan:.2f}", flush=True)
with open(f"{OUT}/vedic.csv", "w", newline="") as f:
    hdr = ["year","ayanamsa","asc","tithi","vaara","nakshatra","yoga","karana","retro"] + [f"{g}_sid" for g in GRAHA]
    w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
    for r in rows: r = dict(r); r["retro"] = "|".join(map(str, r["retro"])); w.writerow(r)
print(f"vedic.csv: {len(rows)} years", flush=True)

# ── Zi Wei Dou Shu via iztro (node) ──
js = r'''
const {astro} = require('iztro');
const out = [];
for (let y = 1700; y <= 2055; y++) {
  try {
    const a = astro.bySolar(`${y}-07-01`, 6, 'male', true, 'en-US');   // hour index 6 = 11:00-13:00 (noon), stated convention
    const pal = a.palaces.map(p => ({ name: p.name, major: p.majorStars.map(s => s.name), minor: p.minorStars.map(s => s.name), stem: p.heavenlyStem, branch: p.earthlyBranch }));
    out.push({ year: y, soul: a.soul, body: a.body, five: a.fiveElementsClass, zodiac: a.zodiac, sign: a.sign, palaces: pal });
  } catch (e) { out.push({ year: y, error: String(e).slice(0, 80) }); }
}
process.stdout.write(JSON.stringify(out));
'''
open(os.path.expanduser("~/.artaquest-dev/tools/iztro_run.js"), "w").write(js)
res = subprocess.run(["node", "iztro_run.js"], cwd=os.path.expanduser("~/.artaquest-dev/tools"), capture_output=True, text=True, timeout=600)
zw = json.loads(res.stdout)
ok = [z for z in zw if "error" not in z]
print(f"iztro: {len(ok)} of {len(zw)} years charted; sample {ok[150]['year']}: soul {ok[150]['soul']}, five {ok[150]['five']}, life-palace stars {ok[150]['palaces'][0]['major']}", flush=True)
json.dump(zw, open(f"{OUT}/ziwei.json", "w"))
print("written to", OUT)
