"""Daily sidereal (Lahiri) ephemeris from kerykeion / Swiss Ephemeris — ONE engine, all bodies named."""
import numpy as np, datetime as dt
from kerykeion import AstrologicalSubject
BOD = ["sun","moon","mercury","venus","mars","jupiter","saturn","uranus","neptune","pluto","true_node"]
d0, d1 = dt.date(1991,1,1), dt.date(2026,12,31)
DAYS = [d0 + dt.timedelta(k) for k in range((d1-d0).days+1)]
LON = np.zeros((len(DAYS), len(BOD)))
for i, d in enumerate(DAYS):
    s = AstrologicalSubject("t", d.year, d.month, d.day, 12, 0, city="Greenwich", nation="GB", lng=0.0, lat=51.48,
                            tz_str="UTC", online=False, zodiac_type="Sidereal", sidereal_mode="LAHIRI", houses_system_identifier="P")
    LON[i] = [getattr(s, b).abs_pos for b in BOD]
    if i % 2000 == 0: print(f"  {d} · Sun {LON[i,0]:.1f} Moon {LON[i,1]:.1f} Mars {LON[i,4]:.1f}", flush=True)
np.savez_compressed("ephemeris_ker_1991_2026.npz", lon=LON, bodies=np.array(BOD), d0=str(d0))
print("done", len(DAYS))
