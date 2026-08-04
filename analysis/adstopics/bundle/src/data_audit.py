#!/usr/bin/env python3
"""adstopics DATA AUDIT — triple-check the dataset before the final runs are trusted.

Checks: file census vs vocabulary; per-series shape (rows, month grid, monotonicity, duplicates);
value sanity (range, NaN, interpolation load, flatness, max<10 census); cross-series duplicate
detection (identical series = Trends alias collapse); step-artifact scan (Trends renorm events);
phase-matrix sanity (range, NaN, moon cache alignment); protocol wall arithmetic; test-window
tie load (excluded months). Exits nonzero on any hard failure.
"""
import importlib.util as u, json, os, sys, hashlib
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
tf = _load("analysis/trends_fit.py", "tf")

hard_fails, warns = [], []
grid = pd.DatetimeIndex(tf.GRID)
i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
iend = len(grid) - tf.DROP_LAST
gwin = grid[i0:iend]
print(f"grid window: {gwin[0]:%Y-%m} .. {gwin[-1]:%Y-%m} ({len(gwin)} months) · DROP_LAST={tf.DROP_LAST}")
assert len(gwin) == 210, "protocol window is not 210 months"

vocab = json.load(open("analysis/adstopics/vocabulary.json"))
slugs = {t: tf.slug(t) for t in vocab}
# slug collisions (two topics -> one file)
rev = {}
for t, sl in slugs.items():
    rev.setdefault(sl, []).append(t)
coll = {sl: ts for sl, ts in rev.items() if len(ts) > 1}
if coll:
    warns.append(f"slug collisions: {len(coll)} (topics sharing a file): {list(coll.items())[:5]}")

missing = [t for t, sl in slugs.items() if not os.path.exists(f"analysis/adstopics/series/{sl}.csv")]
print(f"vocabulary {len(vocab)} · files present {len(vocab)-len(missing)} · missing {len(missing)}")
if missing:
    hard_fails.append(f"missing series files: {len(missing)} e.g. {missing[:5]}")

stats = dict(sparse=0, interp_heavy=0, flat=0, low_max=0, bad_range=0, dup_time=0, short=0)
sigs = {}
tie_load = []
step_suspects = 0
n_full = 0
for sl in sorted(set(slugs.values())):
    p = f"analysis/adstopics/series/{sl}.csv"
    if not os.path.exists(p): continue
    df = pd.read_csv(p)
    if df["Time"].duplicated().any(): stats["dup_time"] += 1
    df["Time"] = pd.to_datetime(df["Time"])
    v = pd.to_numeric(df.drop_duplicates("Time").set_index("Time")["v"].reindex(gwin), errors="coerce")
    nn = v.notna().sum()
    if nn < 210 * 0.5: stats["sparse"] += 1; continue
    if nn < 210 * 0.9: stats["interp_heavy"] += 1
    y = v.interpolate(limit_direction="both").to_numpy(float)
    if not np.isfinite(y).all(): stats["short"] += 1; continue
    if y.min() < 0 or y.max() > 100: stats["bad_range"] += 1
    if y.max() <= 0: stats["flat"] += 1; continue
    if y.max() < 10: stats["low_max"] += 1
    n_full += 1
    sigs.setdefault(hashlib.sha1(y.tobytes()).hexdigest(), []).append(sl)
    dy = np.diff(y)
    tie_load.append(float((dy[-24:] == 0).mean()))
    # step artifact: a single month-over-month jump > 8x the series' median abs change and > 30 units
    mac = np.median(np.abs(dy)) or 1.0
    if (np.abs(dy) > max(8 * mac, 30)).sum() >= 1 and y.std() < 25:
        step_suspects += 1

dups = {h: s for h, s in sigs.items() if len(s) > 1}
print(f"usable-full {n_full} · stats {stats}")
print(f"identical-series groups: {len(dups)} covering {sum(len(s) for s in dups.values())} files"
      f" e.g. {list(dups.values())[:3]}")
print(f"test-window tie load: mean {np.mean(tie_load):.3f} · median {np.median(tie_load):.3f}"
      f" · >50% ties: {(np.array(tie_load) > 0.5).sum()} topics")
print(f"step-artifact suspects: {step_suspects}")
if stats["bad_range"]: hard_fails.append(f"{stats['bad_range']} series out of [0,100]")
if stats["dup_time"]: warns.append(f"{stats['dup_time']} files had duplicate Time rows (deduped on load)")

# phases
lon = tf.ephemeris()
moon = pd.read_csv("analysis/adstopics/_moon_monthly.csv")["moon_synodic"].to_numpy(float)
X = np.column_stack([moon[i0:iend]] + [np.asarray(lon[b], float)[i0:iend] for b in tf.BODIES])
assert X.shape == (210, 12), f"phase matrix shape {X.shape}"
if not np.isfinite(X).all(): hard_fails.append("NaN in phase matrix")
if X.min() < 0 or X.max() >= 360.001: hard_fails.append(f"phase range [{X.min()},{X.max()}]")
sun = X[:, 1]
adv = np.diff(sun) % 360
if not ((adv > 25) & (adv < 35)).all():
    warns.append(f"sun monthly advance outside 25-35 deg somewhere: min {adv.min():.1f} max {adv.max():.1f}")
mo = np.diff(X[:, 0]) % 360
print(f"phases OK: sun advance {adv.mean():.2f}±{adv.std():.2f} deg/mo · moon {mo.mean():.1f} deg/mo")

print("\nHARD FAILS:", hard_fails if hard_fails else "none")
print("WARNINGS:", warns if warns else "none")
sys.exit(1 if hard_fails else 0)
