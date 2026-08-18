import numpy as np, datetime as dt, os
from jhora.panchanga import drik
from jhora import utils
drik.set_ayanamsa_mode("LAHIRI"); place = drik.Place("Greenwich", 0.0, 0.0, 0.0)
d0, d1 = dt.date(1991, 1, 1), dt.date(2026, 12, 31)
DAYS = [d0 + dt.timedelta(k) for k in range((d1-d0).days+1)]
LON = np.zeros((len(DAYS), 8)); TITHI = np.zeros(len(DAYS)); NAK = np.zeros(len(DAYS))
for i, d in enumerate(DAYS):
    jd = utils.julian_day_number(drik.Date(d.year, d.month, d.day), (12, 0, 0))
    LON[i] = [float(drik.sidereal_longitude(jd, p)) for p in range(8)]
    try: TITHI[i] = drik.tithi(jd, place)[0]
    except Exception: TITHI[i] = -1
    NAK[i] = np.floor(LON[i,1]/(360/27))
    if i % 2000 == 0: print(f"  {d} · Moon {LON[i,1]:.1f} Sun {LON[i,0]:.1f}", flush=True)
np.savez_compressed("ephemeris_daily_1991_2026.npz", lon=LON, tithi=TITHI, nak=NAK, d0=str(d0))
print("done", len(DAYS), "days")
