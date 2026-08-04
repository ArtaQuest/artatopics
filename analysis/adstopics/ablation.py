#!/usr/bin/env python3
"""ASTROLOGY SIGNAL ABLATION (operator 2026-07-19): systematically measure the honest marginal
contribution of EVERY kind of astrological signal — extra bodies (Moon, Lilith, node), retrogrades,
stations, out-of-bounds declination, declination parallels, eclipses, ingresses, speed, and
planet-planet aspects — over the Sun annual-cycle baseline, with a cross-year-transfer gate so only
signals that GENERALISE across years are credited. Reports add-one-in and leave-one-out deltas on
BOTH validation (the honest selector) and the untouched test window.

  python3 analysis/adstopics/ablation.py
"""
import importlib.util as u, itertools, os
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
rc = _load("analysis/adstopics/ratechange.py", "rc")

E = pd.read_csv("analysis/_ephemeris_extended.csv")
grid = pd.DatetimeIndex(pd.to_datetime(E["Time"]))
i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
BOD = sorted(set(c.rsplit("_", 1)[0] for c in E.columns if c != "Time"))
LON = {b: np.deg2rad(E[f"{b}_lon"].to_numpy(float)) for b in BOD}
SPD = {b: E[f"{b}_spd"].to_numpy(float) for b in BOD if f"{b}_spd" in E.columns}
DEC = {b: E[f"{b}_dec"].to_numpy(float) for b in BOD if f"{b}_dec" in E.columns}
PLANETS = [b for b in ["sun", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "chiron"] if b in LON]


def sl(a, n):
    return a[i0:i0 + n]


def families(n):
    """Return {family: feature matrix (n, F)} — every deterministic sky signal, sliced to the topic grid."""
    F = {}
    K = 6
    # 1. SUN — the annual cycle (baseline)
    F["sun"] = np.column_stack([f(k * sl(LON["sun"], n)) for k in range(1, K + 1) for f in (np.cos, np.sin)])
    # 2. other planets' longitude harmonics
    F["planets_lon"] = np.column_stack([f(k * sl(LON[b], n)) for b in PLANETS if b != "sun"
                                        for k in range(1, K + 1) for f in (np.cos, np.sin)])
    # 3. MOON longitude + synodic phase
    F["moon_lon"] = np.column_stack([f(k * sl(LON["moon"], n)) for k in range(1, 5) for f in (np.cos, np.sin)])
    ph = sl(LON["moon"], n) - sl(LON["sun"], n)
    F["moon_phase"] = np.column_stack([f(k * ph) for k in range(1, 5) for f in (np.cos, np.sin)])
    # 4. LILITH (Black Moon)
    F["lilith"] = np.column_stack([f(k * sl(LON["lilith"], n)) for k in range(1, 4) for f in (np.cos, np.sin)])
    # 5. NODE
    F["node"] = np.column_stack([f(k * sl(LON["node"], n)) for k in range(1, 4) for f in (np.cos, np.sin)])
    # 6. RETROGRADES — per-body flag (speed < 0)
    F["retro"] = np.column_stack([(sl(SPD[b], n) < 0).astype(float) for b in SPD if b not in ("sun", "moon")])
    # 7. STATIONS — |speed| in the lowest decile per body (planet stationary)
    st = []
    for b in SPD:
        if b in ("sun", "moon"): continue
        s = np.abs(SPD[b]); st.append((sl(s, n) < np.quantile(np.abs(SPD[b]), 0.15)).astype(float))
    F["station"] = np.column_stack(st)
    # 8. SPEED — signed normalized speed
    F["speed"] = np.column_stack([np.clip(sl(SPD[b], n) / (np.abs(SPD[b]).std() + 1e-9), -3, 3)
                                  for b in SPD if b not in ("sun", "moon")])
    # 9. OUT-OF-BOUNDS declination (|dec| > 23.44)
    F["oob"] = np.column_stack([(np.abs(sl(DEC[b], n)) > 23.44).astype(float) for b in DEC if b != "sun"])
    # 10. DECLINATION harmonics
    F["dec"] = np.column_stack([sl(DEC[b], n) / 30.0 for b in DEC])
    # 11. DECLINATION PARALLELS (|dec_p - dec_q| small)
    par = []
    for p, q in itertools.combinations([b for b in DEC if b != "sun"], 2):
        par.append((np.abs(sl(DEC[p], n) - sl(DEC[q], n)) < 1.0).astype(float))
    F["parallel"] = np.column_stack(par)
    # 12. ECLIPSES — Moon near node AND near new/full (|Moon-Sun| near 0 or pi)
    mn = np.abs(np.angle(np.exp(1j * (sl(LON["moon"], n) - sl(LON["node"], n)))))
    ms = np.abs(np.angle(np.exp(1j * (sl(LON["moon"], n) - sl(LON["sun"], n)))))
    ecl = ((mn < np.deg2rad(15)) & ((ms < np.deg2rad(20)) | (np.abs(ms - np.pi) < np.deg2rad(20)))).astype(float)
    F["eclipse"] = ecl[:, None]
    # 13. INGRESSES — a planet within 5° of a sign boundary (longitude mod 30)
    ing = []
    for b in PLANETS[1:]:
        d = np.rad2deg(sl(LON[b], n)) % 30; ing.append(((d < 5) | (d > 25)).astype(float))
    F["ingress"] = np.column_stack(ing)
    # 14. ASPECTS — genuine planet-planet (Sun excluded), harmonics 1..3
    asp = []
    for p, q in itertools.combinations([b for b in PLANETS if b != "sun"], 2):
        sep = sl(LON[p], n) - sl(LON[q], n)
        for k in range(1, 4):
            asp.append(np.cos(k * sep)); asp.append(np.sin(k * sep))
    F["aspects"] = np.column_stack(asp)
    return F


def furnace(b, Phi, wall_fit, lo, hi, ridge=1.0):
    """Cross-year-gated per-topic ridge furnace: fit coeffs on months < wall_fit, predict [lo,hi)."""
    Tn = b.Y.shape[0]; Fn = Phi.shape[1]
    lw = np.where(b.tie, 0.0, b.L * 2.0 - 1.0)
    tr = np.arange(1, wall_fit); Lc = lw[:, tr]; Ph = Phi[tr]
    mu, sd = Ph.mean(0), Ph.std(0) + 1e-9; Phs = (Ph - mu) / sd
    G = Phs.T @ Phs + ridge * Fn * np.eye(Fn); Ginv = np.linalg.inv(G)
    C = (Lc @ Phs) @ Ginv
    yr = (tr - 1) // 12; fold = yr % 2; gate = np.zeros((Tn, Fn))
    for fi, sc in ((0, 1), (1, 0)):
        mf = fold == fi; ms = fold == sc
        Cf = (Lc[:, mf] @ Phs[mf]) @ np.linalg.inv(Phs[mf].T @ Phs[mf] + ridge * Fn * np.eye(Fn))
        pred = Cf @ Phs[ms].T
        gate += (Lc[:, ms] * pred).sum(1, keepdims=True) / (np.abs(Lc[:, ms]).sum(1, keepdims=True) + 1e-9) * (Cf * C > 0)
    gate = np.clip(gate / 2.0, 0, None); gate = gate / (gate.max(1, keepdims=True) + 1e-9)
    cols = np.arange(lo, hi); Pht = (Phi[cols] - mu) / sd
    Z = (C * gate) @ Pht.T
    acc = (((Z > 0) == (b.L[:, cols] == 1)) * (~b.tie[:, cols])).sum() / (~b.tie[:, cols]).sum()
    return acc


def main():
    names, Y = rc.bal.load_all(); b = rc.BenchR(Y); rc.ceilings(b)
    F = families(b.n)
    print(f"  families: {[(k, v.shape[1]) for k, v in F.items()]}", flush=True)
    def score(sets, tag):
        Phi = np.column_stack([F[s] for s in sets])
        va = furnace(b, Phi, b.a, b.a, b.b); te = furnace(b, Phi, b.b, b.b, b.n)
        print(f"  {tag:34s} val {va:.4f}  test {te:.4f}", flush=True); return va, te
    print("\n  == SUN BASELINE ==", flush=True)
    vb, tb = score(["sun"], "sun (annual cycle)")
    print("\n  == ADD-ONE-IN (marginal over Sun) — ranked by validation delta ==", flush=True)
    deltas = []
    for k in F:
        if k == "sun": continue
        va, te = score(["sun", k], f"sun + {k}")
        deltas.append((k, va - vb, te - tb))
    print("\n  == MARGINAL DELTA TABLE (val Δ / test Δ over Sun) ==", flush=True)
    for k, dv, dt in sorted(deltas, key=lambda x: -x[1]):
        flag = "  <-- helps val" if dv > 0.001 else ("  (val-selector rejects)" if dv <= 0 else "")
        print(f"    {k:12s}  valΔ {dv:+.4f}   testΔ {dt:+.4f}{flag}", flush=True)
    print("\n  == KITCHEN SINK (all families) ==", flush=True)
    score(list(F.keys()), "ALL signals")
    # leave-one-out from the val-positive set
    pos = ["sun"] + [k for k, dv, _ in deltas if dv > 0.001]
    if len(pos) > 1:
        print(f"\n  == BEST VAL-POSITIVE SET {pos} ==", flush=True)
        score(pos, "sun + val-positive families")


if __name__ == "__main__":
    main()
