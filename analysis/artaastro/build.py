#!/usr/bin/env python3
"""
build.py — ArtaAstro day-by-day sidereal (Lahiri) ephemeris + feature generator.

Engine: KERYKEION (Sidereal, sidereal_mode="LAHIRI") for Sun..Pluto — its output matches raw
Swiss Ephemeris to the decimal and covers the whole span. Rahu/Ketu (the lunar nodes) are taken
from the SAME underlying Swiss Ephemeris at Kerykeion's own Julian day (this Kerykeion build
returns None for its node fields). Eclipses come from pyswisseph's global eclipse finders.

Instant per day: 10:00 America/New_York (the project's Season-reset convention), DST-aware.

It produces, in one pass over the union span 1979-01-01 .. 2253-02-09:
  out/daily_intensity.csv     one row per day: date, intensity(0-100), raw, features, top trigger
                              -> the series the backtest scores against GDELT (sliced to 1979..today)
  out/astro-YYYYs.json        per-decade packed chart data over the PRODUCT range 2020-2253
  out/astro-notable.json      the highest-intensity days (with triggers + a short reading)
  out/manifest.json           span, chunk list, body order, generation params

The intensity model is the A-PRIORI rule set in intensity.py (never tuned against GDELT).
"""
import os, sys, json, math, datetime as dt, warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from kerykeion import AstrologicalSubject
import swisseph as swe
import intensity as I

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)

# ---- span ----------------------------------------------------------------------------------------
UNION_START   = dt.date(int(os.environ.get("AQ_START_Y", "1979")), 1, 1)
UNION_END     = dt.date(2253, 2, 9)
PRODUCT_START = dt.date(2020, 2, 25)
PRODUCT_END   = dt.date(2253, 2, 9)
# NY (for the 10:00 America/New_York instant; planet longitudes are geocentric so lat/lon don't
# affect them — location only sets the clock's DST offset).
NY_LAT, NY_LNG, TZ = 40.7128, -74.0060, "America/New_York"

BODIES = ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto","Rahu","Ketu"]
KER_ATTR = {"Sun":"sun","Moon":"moon","Mercury":"mercury","Venus":"venus","Mars":"mars",
            "Jupiter":"jupiter","Saturn":"saturn","Uranus":"uranus","Neptune":"neptune","Pluto":"pluto"}
SLOW    = ["Jupiter","Saturn","Uranus","Neptune","Pluto","Rahu"]   # ingress-watch
OUTER   = ["Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"]   # station-watch
SIGNS   = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
NAK_ARC = 360.0/27.0
PADA_ARC = 360.0/108.0

swe.set_sid_mode(swe.SIDM_LAHIRI, 0, 0)

def nak(lon):
    n = int(lon // NAK_ARC)
    pada = int((lon % NAK_ARC) // PADA_ARC) + 1
    return n, pada

# ---- eclipses over the span ----------------------------------------------------------------------
def eclipse_kind(retflag):
    if retflag & swe.ECL_TOTAL:   return "total"
    if retflag & swe.ECL_ANNULAR: return "annular"
    if retflag & swe.ECL_ANNULAR_TOTAL: return "annular"
    return "partial"

def build_eclipses(jd0, jd1):
    """Return dict {date_iso: {'kind','lon','lunar'}} spread over +-3 days of each eclipse max."""
    marks = {}
    def add(tmax, lunar, kind):
        body = swe.MOON if lunar else swe.SUN
        lon = swe.calc_ut(tmax, body, swe.FLG_SWIEPH|swe.FLG_SIDEREAL)[0][0]
        y,m,d,_ = swe.revjul(tmax)
        base = dt.date(int(y),int(m),int(d))
        for off in range(-3,4):
            iso = (base+dt.timedelta(days=off)).isoformat()
            prev = marks.get(iso)
            w = I.ECLIPSE_W.get(kind,5.0)
            if prev is None or w > I.ECLIPSE_W.get(prev["kind"],5.0):
                marks[iso] = {"kind":kind, "lon":lon, "lunar":lunar}
    # solar
    jd = jd0
    while jd < jd1:
        ret, tret = swe.sol_eclipse_when_glob(jd, swe.FLG_SWIEPH, 0, False)
        tmax = tret[0]
        if tmax >= jd1: break
        add(tmax, False, eclipse_kind(ret)); jd = tmax + 5
    # lunar
    jd = jd0
    while jd < jd1:
        ret, tret = swe.lun_eclipse_when(jd, swe.FLG_SWIEPH, 0, False)
        tmax = tret[0]
        if tmax >= jd1: break
        add(tmax, True, eclipse_kind(ret)); jd = tmax + 5
    return marks

# ---- main pass -----------------------------------------------------------------------------------
def reading(triggers):
    if not triggers: return "A quiet sky — no exact slow-planet aspect, ingress, station or eclipse."
    return "; ".join(triggers[:4])

def main():
    jd0 = swe.julday(UNION_START.year, UNION_START.month, UNION_START.day, 12.0)
    jd1 = swe.julday(UNION_END.year, UNION_END.month, UNION_END.day, 12.0)
    sys.stderr.write("finding eclipses...\n"); sys.stderr.flush()
    ecl = build_eclipses(jd0, jd1)
    sys.stderr.write(f"  {len(ecl)} eclipse-window days\n"); sys.stderr.flush()

    daily_rows = []                    # for csv
    eph_rows = []                      # full-span ephemeris (date + 12 sidereal longitudes) for the competition
    decades = defaultdict(list)        # product chunks
    notable = []                       # top-intensity product days
    prev_sign = {}; prev_retro = {}
    n = 0
    day = UNION_START
    while day <= UNION_END:
        s = AstrologicalSubject("d", day.year, day.month, day.day, 10, 0,
                                lng=NY_LNG, lat=NY_LAT, tz_str=TZ,
                                zodiac_type="Sidereal", sidereal_mode="LAHIRI",
                                online=False, disable_chiron_and_lilith=True)
        jd = s.julian_day
        bodies = {}
        for nm in BODIES:
            if nm in KER_ATTR:
                p = getattr(s, KER_ATTR[nm])
                lon = float(p.abs_pos) % 360.0
                retro = bool(p.retrograde)
            elif nm == "Rahu":
                xx = swe.calc_ut(jd, swe.MEAN_NODE, swe.FLG_SWIEPH|swe.FLG_SIDEREAL|swe.FLG_SPEED)[0]
                lon = xx[0] % 360.0; retro = xx[3] < 0
            else:  # Ketu = Rahu + 180
                lon = (bodies["Rahu"]["lon"] + 180.0) % 360.0; retro = bodies["Rahu"]["retro"]
            bodies[nm] = {"lon": lon, "retro": retro, "sign": int(lon // 30)}

        # ingress / station vs previous day
        ingress = [nm for nm in SLOW  if nm in prev_sign  and bodies[nm]["sign"]  != prev_sign[nm]]
        station = [nm for nm in OUTER if nm in prev_retro and bodies[nm]["retro"] != prev_retro[nm]]
        for nm in BODIES:
            prev_sign[nm] = bodies[nm]["sign"]; prev_retro[nm] = bodies[nm]["retro"]

        eday = ecl.get(day.isoformat())
        sc, raw, trig = I.score({"bodies":bodies, "ingress":ingress, "station":station, "eclipse":eday})

        iso = day.isoformat()
        n_hard = sum(1 for (a,b,name,d) in I.aspects_between(bodies)
                     if name in I.HARD and frozenset((a,b)) in I.PAIR_WEIGHT)
        daily_rows.append((iso, round(sc,3), round(raw,4), n_hard,
                           1 if eday else 0, len(ingress), len(station),
                           trig[0] if trig else ""))
        eph_rows.append([iso] + [round(bodies[nm]["lon"], 4) for nm in BODIES])

        # product outputs (2020-2253 only)
        if PRODUCT_START <= day <= PRODUCT_END:
            packed = [int(round(bodies[nm]["lon"]*10)) for nm in BODIES]
            rmask = 0
            for i,nm in enumerate(BODIES):
                if bodies[nm]["retro"]: rmask |= (1<<i)
            decades[(day.year//10)*10].append({"d":iso, "i":int(round(sc)), "b":packed, "r":rmask})
            if sc >= 55:
                notable.append({"d":iso, "i":round(sc,1), "raw":round(raw,2),
                                "triggers":trig, "reading":reading(trig)})
        n += 1
        if n % 5000 == 0:
            sys.stderr.write(f"  {iso}  ({n} days)\n"); sys.stderr.flush()
        day += dt.timedelta(days=1)

    # write daily intensity csv
    import csv
    with open(os.path.join(OUT,"daily_intensity.csv"),"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["date","intensity","raw","n_hard_aspects","eclipse","n_ingress","n_station","top_trigger"])
        w.writerows(daily_rows)
    with open(os.path.join(OUT,"ephemeris_daily.csv"),"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + [b.lower() for b in BODIES])
        w.writerows(eph_rows)

    # product chunks
    chunk_files = []
    for dec in sorted(decades):
        fn = f"astro-{dec}s.json"
        json.dump({"decade":dec, "bodies":BODIES, "days":decades[dec]},
                  open(os.path.join(OUT,fn),"w"), separators=(",",":"))
        chunk_files.append(fn)

    notable.sort(key=lambda x:-x["i"])
    json.dump({"count":len(notable), "days":notable[:2000]},
              open(os.path.join(OUT,"astro-notable.json"),"w"), indent=1)

    json.dump({"engine":"kerykeion+swisseph", "ayanamsa":"LAHIRI",
               "instant":"10:00 America/New_York", "bodies":BODIES,
               "union_span":[UNION_START.isoformat(), UNION_END.isoformat()],
               "product_span":[PRODUCT_START.isoformat(), PRODUCT_END.isoformat()],
               "total_days":len(daily_rows), "chunks":chunk_files,
               "notable_threshold":55, "notable_count":len(notable)},
              open(os.path.join(OUT,"manifest.json"),"w"), indent=1)
    sys.stderr.write(f"DONE: {len(daily_rows)} days, {len(chunk_files)} chunks, {len(notable)} notable\n")

if __name__ == "__main__":
    main()
