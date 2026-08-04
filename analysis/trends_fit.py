#!/usr/bin/env python3
"""
ArtaQuest — sidereal trend fit  (monthly resolution, flat pages)
===============================================================
DATA: for each topic, Google-Trends worldwide interest 2004→now is fetched once at MONTHLY
resolution (~270 points) and reindexed to a fixed monthly GRID. The most recent 12 months are
EXCLUDED from the fit (DROP_LAST) to avoid real-time / recency bias — they are dropped, not held
back as a test set, so the reported R² is IN-SAMPLE. Fetches are cached on disk.

MODEL: exactly 11 parameters — one non-negative body weight each, NO intercept; a shared phase angle
phi is swept around the zodiac on a 5° grid (a sign-selector, not a fitted coefficient). At each phi
build one feature per body,
   feature(body,t) = sinc( wrapped_angle(longitude_body(t) − phi) [rad] )
the fixed annual SINC kernel (first zero ≈57°) — a localized peak when the body sits at phi, whose
side-lobes give a little harmonic diversity a pure cosine lacks (pure cosine dropped R² 0.75→0.55). Fit
   search(t) ~ sum_body weight · feature(body,t)
by NON-NEGATIVE least squares (each body's weight >= 0, so a body only ADDS to interest; no regularisation). The phi of MINIMUM
mean-squared-error is the topic's ANGLE → its sign; the in-sample R² there is the headline score.
Bodies (11): Sun, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, Chiron,
lunar Node — one cosine each at its full orbital period (Moon removed; no harmonics).

SHARES: a body's raw weight is NOT directly its contribution (features differ in scale/variance; with
NON-NEGATIVE weights every body only adds, so weights no longer blow up and cancel). The pages instead
report a simple variance decomposition: flatten each
body's feature to its MEAN (keeping the fitted weights) and recompute the empirical R² — the drop
is that body's contribution, v_b = R²_full − R²(body b → its mean) = Var(weight_b·feature_b)/Var(y).
Each v_b is a variance, hence NON-NEGATIVE; the twelve are normalised to sum EXACTLY to R². All 12
bodies are listed for every topic — no thresholds.

PAGES: one flat page per topic (angle→sign, tuning curve, fit, per-body R²-contribution) + one
page per cycle (every topic ranked by its R²-contribution to that cycle) + the index + the 12
sign-houses. ALL bodies and ALL topics are listed — no thresholds. "Correlation is not causation".

  python3 analysis/trends_fit.py --n 5      # process the next 5 un-done topics (in order)
  python3 analysis/trends_fit.py            # process every remaining topic
  python3 analysis/trends_fit.py --rebuild  # rebuild ALL pages from the registry (no network)
"""
import os, sys, json, time, base64, urllib.parse, re, logging, warnings, io
import numpy as np, pandas as pd, requests
from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares, nnls as _nnls
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/cert.pem")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/ssl/cert.pem")
os.environ.setdefault("KERYKEION_GEONAMES_USERNAME", "dummy")
warnings.filterwarnings("ignore"); logging.getLogger().setLevel(logging.ERROR)

# ───────────────────────────── CONFIG ─────────────────────────────
SIGNS = ['Aries','Taurus','Gemini','Cancer','Leo','Virgo','Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces']
BODIES = ['sun','mercury','venus','mars','jupiter','saturn','uranus','neptune','pluto','chiron','node']  # 11 (Moon removed; Lilith removed 2026-07-01)
PERIOD_YEARS = {'sun':1.0,'moon':0.080849,'mercury':0.241,'venus':0.615,'mars':1.881,'jupiter':11.86,
                'saturn':29.46,'uranus':84.0,'neptune':164.8,'pluto':248.0,'chiron':50.0,'node':18.6}
# The Moon is a SYNODIC clock — its behaviourally-relevant cycle is its PHASE (illumination: full↔new moon),
# the moon–sun angle, period 29.53 d (0.080849 yr), NOT its sidereal position (27.32 d). Verified directly:
# on daily "full moon" interest the synodic feature explains 20% vs the sidereal feature's 1% (the rest of
# the bodies stay sidereal). See eff_lon(), which swaps the Moon's longitude for its elongation from the Sun.
SYNODIC_MOON = True
ATTR = {'node':'true_node'}                                     # lunar north node — kerykeion attribute name
SIDEREAL_MODE = "USHASHASHI"                                   # ayanamsa (zero-point of the sidereal zodiac). The
# ayanamsa is a constant offset that does NOT change R² (the phase absorbs it), but it DOES rotate the 12-sign labelling.
# USHASHASHI chosen 2026-06-30 by MAXIMUM AVERAGE REP: of all 47 kerykeion sidereal modes + Tropical (Western) it gives
# the highest mean rep — decisiveness, top sign-area ÷ runner-up (R²-gated, capped 10×) — over the 436 occupations
# (mean rep 1.0840; range across modes 1.0737–1.0840). i.e. the zero-point that places occupations most decisively in
# their signs. (The aggregate-distribution entropy is FLAT across all modes, ~3.536 bits, so it doesn't discriminate;
# rep does.) Full per-system table + the exhaustive settings coverage in analysis/_ayanamsa_table.md.
TERMS = ["Sport","Finance","Science","History","Art","Health","Law","Engineering","Education","Business","Technology","Culture"]
REFS = np.arange(0, 360, 5)                                      # phase swept at 5° (0,5,…,355) → 72 angles, 6 per 30° sign band
# (2026-07-01: 5° grid; each of the 12 sign bands gets a whole set of 5° points 0,5,10,15,20,25)
# PER-KEYWORD PERIOD: the kernel inverse-width w = 1/PERIOD is tuned per field (like the phase), reported in YEARS.
# For each candidate period we do the full phase sweep and score it by F1(rep, R²); the period with the highest F1
# is the field's. Grid spans ~0.5–4.5 yr (w 0.22–2.0), the range where the sinc is resolvable at monthly cadence.
PERIOD_GRID = [0.5, 0.6, 0.75, 0.9, 1.0, 1.15, 1.4, 1.7, 2.0, 2.4, 2.9, 3.5, 4.5]
# This is only the SEED grid — a good starting centre. fit_topic then holds a SYMMETRIC geometric window
# around the best period and re-centres it until the winner sits in the MIDDLE (never stranded at an edge).
PBOUND = (0.2, 20.0)   # period window (yr): ~monthly-Nyquist floor · ~the ~20-yr data span as the ceiling
PSTEP  = 1.22          # geometric step between adjacent window periods (matches the seed grid's ~1.2× spacing)
PK     = 6             # SEARCH half-window: PK periods each side of the centre while hunting the winner
PSHOW  = 2             # PLOT half-window: only PSHOW each side of the winner → 2·PSHOW+1 = 5 bars, winner centred
START, END = pd.Timestamp("2004-01-01"), pd.Timestamp("2026-06-24")
ALL_TIME = f"{START.date()} {END.date()}"

# MONTHLY resolution: the raw all-time Google-Trends export (one fetch per topic, ~270 points) — exactly the
# data the verified reference model uses. No chunk-stitching/rescaling (that damaged the signal); no detrend.
GRID = pd.date_range(START, END, freq="MS")
DROP_LAST = 12            # drop the most recent 12 months from the fit — avoids real-time / recency bias
DATA_DIR = os.environ.get("AQ_DATA_MONTHLY", "analysis/data_monthly")   # env override: alternate monthly cache (astro4 per-property series)
CHUNK_DIR, PAGES_DIR = "analysis/data_chunks", "analysis/pages"
EPHEM = "analysis/_ephemeris_monthly.csv"
REGISTRY, DISC_FILE = "analysis/topics.json", "analysis/disciplines.json"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
def nice(b): return b.title()
def slug(s): return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')
def pct(x): return ("−" if x < 0 else "") + f"{abs(x) * 100:.1f}%"     # signed %, 1 dp, unicode minus
for d in (DATA_DIR, CHUNK_DIR, PAGES_DIR): os.makedirs(d, exist_ok=True)

# ── Cycles: one per body, at its full orbital period. (No harmonics — they shatter the single clear peak.)
HARM = {b: [1] for b in BODIES}
TERMS_H = [(b, 1) for b in BODIES]
HKEY = [f"{b}:1" for b in BODIES]
assert len(BODIES) == 11, f"11 parameters = one non-negative weight per body, NO bias, phase swept for the sign, pure-cosine kernel. (Moon removed, Lilith removed 2026-07-01) got {len(BODIES)}"
def h_years(b, k): return PERIOD_YEARS[b] / k
def h_slug(b, k):  return b if k == 1 else f"{b}-{k}x"
def _parse(hk):    b, k = hk.split(":"); return b, int(k)
def period_words(yv):
    d = yv * 365.25
    if d < 45:   return "about a month"
    if d < 330:  return f"about {round(d / 30.44)} months"
    if yv < 1.5: return "about a year"
    return f"about {round(yv)} years"
def h_label(b, k): return f"{nice(b)} · {period_words(h_years(b, k))}"          # period disambiguates octaves
def art(b): return {"sun": "the Sun", "moon": "the Moon", "node": "the lunar node"}.get(b, nice(b))

# ─────────────────────── EXTRACTION (daily stitch) ────────────────
def _strip(t): return t[t.find("{"):] if "{" in t else t

def make_session():
    import browser_cookie3
    s = requests.Session(); s.cookies = browser_cookie3.chrome(domain_name="google.com")
    s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9", "Referer": "https://trends.google.com/"})
    return s

# UNIVERSAL baseline ladder (the brand "ArtaQuest", split into two real-volume words): every query is fetched as a
# 3-term comparison [query, 'arta', 'quest']. 'quest' (~38× 'arta') is the stable WORKING REFERENCE — it stays
# measurable across the whole keyword range — and 'arta' is the finer FALLBACK rung for a keyword that dwarfs quest.
# The query is returned on ONE universal scale: % of 'quest' (quest mean ≡ 100). Two keywords fetched this way are
# directly comparable — they share the one quest↔arta reference, recovering the cross-keyword magnitude Trends hides.
ANCHOR_A, ANCHOR_B = "arta", "quest"
_CALIB_FILE = "analysis/_anchor_calib.json"
QUEST_PER_ARTA = (json.load(open(_CALIB_FILE)).get("quest_per_arta", 37.7) if os.path.exists(_CALIB_FILE) else 37.7)
LAST_ANCHOR = {}                                    # audit: fetch_range records each query's measured {arta, quest, scale}

def calibrate(s):
    """(Re)measure the baseline ratio quest/arta over the full range and cache it (so the fallback rung stays current
       as the anchors' volumes drift). Returns quest_per_arta. Run occasionally; cheap (one comparison)."""
    global QUEST_PER_ARTA
    LAST_ANCHOR.pop("__calib__", None)
    fetch_range(ANCHOR_B, s, ALL_TIME)                             # measures arta & quest within the same [.,arta,quest] frame
    info = LAST_ANCHOR.get(ANCHOR_B, {})
    am, bm = info.get("arta", 0.0), info.get("quest", 0.0)
    if am and am >= 1.0:
        QUEST_PER_ARTA = round(float(bm / am), 3)
        json.dump({"quest_per_arta": QUEST_PER_ARTA, "arta": am, "quest": bm}, open(_CALIB_FILE, "w"))
    return QUEST_PER_ARTA

_EXPLORE = "https://trends.google.com/trends/api/explore?hl=en-US&tz=0&req="
_MULTILINE = "https://trends.google.com/trends/api/widgetdata/multiline?hl=en-US&tz=0&req="
def _fetch_timeline(req, s):
    """explore → multiline, riding out 429s (≈26 min) rather than skipping. Returns the timelineData list or None."""
    for wait in (20, 60, 180, 420, 900, None):
        try:
            time.sleep(2)                          # polite spacing so we don't trigger the limit in the first place
            r1 = s.get(_EXPLORE + urllib.parse.quote(json.dumps(req)), timeout=25)
            if r1.status_code == 429: raise RuntimeError("explore 429")
            w = next(x for x in json.loads(_strip(r1.text))["widgets"] if x["id"] == "TIMESERIES")
            r2 = s.get(_MULTILINE + urllib.parse.quote(json.dumps(w["request"], separators=(",", ":"))) + "&token=" + w["token"], timeout=25)
            if r2.status_code == 429: raise RuntimeError("multiline 429")
            return json.loads(_strip(r2.text))["default"]["timelineData"]
        except StopIteration: return None
        except Exception as e:
            if wait is None: print(f"      FAILED {e}", flush=True); return None
            time.sleep(wait)

def fetch_single(query, s, tr):
    """One Trends range, SINGLE field (no anchor) → the query normalised to its OWN max=100 = MAXIMUM precision for
       the trend SHAPE. (The anchored [query,arta,quest] frame compresses a low-popularity field to a few integers,
       losing precision; every field deserves equal, full-resolution shape.) Popularity vs quest is recovered
       separately by fetch_range (monthly). Cached by query+range."""
    cache = f"{CHUNK_DIR}/{slug(query)}__one__{tr.replace(' ', '_')}.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache); df["Time"] = pd.to_datetime(df["Time"]); return df
    if s is None: return None
    td = _fetch_timeline({"comparisonItem": [{"keyword": query, "geo": "", "time": tr}], "category": 0, "property": ""}, s)
    if td is None: return None
    v = np.array([float(d["value"][0]) for d in td])               # single field, normalised to its own max=100
    df = pd.DataFrame({"Time": pd.to_datetime([int(d["time"]) for d in td], unit="s"), "v": v})
    df.to_csv(cache, index=False); return df

def fetch_range(query, s, tr):
    """One Trends range, universal-baseline-anchored [query, arta, quest] → the query as % of 'quest' (see above).
       Used ONLY for the cross-field POPULARITY now (a single monthly comparison); the trend SHAPE comes from the
       higher-precision single-field fetch_single. Cached by query+range."""
    cache = f"{CHUNK_DIR}/{slug(query)}__vs-aq__{tr.replace(' ', '_')}.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache); df["Time"] = pd.to_datetime(df["Time"]); return df
    if s is None: return None
    td = _fetch_timeline({"comparisonItem": [{"keyword": query, "geo": "", "time": tr},
                                             {"keyword": ANCHOR_A, "geo": "", "time": tr},
                                             {"keyword": ANCHOR_B, "geo": "", "time": tr}], "category": 0, "property": ""}, s)
    if td is None: return None
    qv = np.array([float(d["value"][0]) for d in td])              # query
    am = float(np.mean([float(d["value"][1]) for d in td]))        # 'arta'  mean (fallback rung)
    bm = float(np.mean([float(d["value"][2]) for d in td]))        # 'quest' mean (working reference)
    saturated = bm < 1.0 and am < 0.5                             # both anchors round to ~0 → magnitude not recoverable
    scale = bm if bm >= 1.0 else (am * QUEST_PER_ARTA if am >= 0.5 else 1.0)
    v = qv / scale * 100.0                                         # UNIVERSAL: query as % of 'quest'
    LAST_ANCHOR[query] = {"arta": round(am, 3), "quest": round(bm, 3), "scale": round(scale, 3), "saturated": bool(saturated)}
    df = pd.DataFrame({"Time": pd.to_datetime([int(d["time"]) for d in td], unit="s"), "v": v})
    df.to_csv(cache, index=False); return df

def fetch_daily(query, s):
    """The raw all-time MONTHLY export, reindexed to GRID — exactly the verified reference's data. One request
       per topic, no chunk-stitching or rescaling (which damaged the signal). Cached."""
    cache = f"{DATA_DIR}/{slug(query)}.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache); df["Time"] = pd.to_datetime(df["Time"]); return df
    bb = fetch_range(query, s, ALL_TIME)
    if bb is None or bb.empty: return None
    bb = bb.copy(); bb["v"] = pd.to_numeric(bb["v"], errors="coerce")          # Trends '<1' etc. → NaN, never object
    ser = bb.drop_duplicates("Time").set_index("Time").sort_index()["v"].reindex(GRID).interpolate(limit_direction="both")
    out = pd.DataFrame({"Time": GRID, "v": ser.to_numpy()})
    out.to_csv(cache, index=False); print(f"    {len(out)} monthly pts", flush=True); return out

# ─────────────────────────── EPHEMERIS ────────────────────────────
def ephemeris():
    if os.path.exists(EPHEM):
        e = pd.read_csv(EPHEM)
        if len(e) == len(GRID): return {b: e[b].values for b in BODIES if b in e.columns}
    from kerykeion import AstrologicalSubject
    print(f"  computing daily ephemeris for {len(GRID)} dates… (one-time)", flush=True)
    lon = {b: [] for b in BODIES}
    for dt in GRID:
        try:
            sub = AstrologicalSubject("Q", dt.year, dt.month, dt.day, 12, 0, lat=0, lng=0, tz_str="UTC", zodiac_type="Sidereal", sidereal_mode=SIDEREAL_MODE)
            for b in BODIES:
                p = getattr(sub, ATTR.get(b, b), None); lon[b].append(float(p.abs_pos) % 360 if p is not None and hasattr(p, "abs_pos") else np.nan)
        except Exception:
            for b in BODIES: lon[b].append(np.nan)
    lon = {b: np.array(v) for b, v in lon.items() if np.isfinite(np.array(v)).all()}
    out = pd.DataFrame(lon); out.insert(0, "Time", GRID); out.to_csv(EPHEM, index=False)
    return lon

MEAN_PERIODS = "analysis/_mean_periods.json"
def mean_periods():
    """Each body's EXACT orbital period, measured day-by-day from the sidereal ephemeris. Runs the ephemeris DAILY from
       2004-01-01 to the end of the data window; for each body it UNWRAPS the longitude and fits the mean angular
       velocity ω (deg/year) by a straight line of unwrapped-longitude vs time-in-years, then P = 360/|ω| years. The
       node moves retrograde (ω < 0) — |ω| handles it. Cached to _mean_periods.json (the daily run is slow — done once)."""
    if os.path.exists(MEAN_PERIODS):
        try: return {k: float(v) for k, v in json.load(open(MEAN_PERIODS)).items()}
        except Exception: pass
    from kerykeion import AstrologicalSubject
    days = pd.date_range(START, GRID[-1], freq="D")
    print(f"  computing DAILY ephemeris for {len(days)} days to measure exact periods… (one-time)", flush=True)
    lon = {b: [] for b in BODIES}
    for dt in days:
        try:
            sub = AstrologicalSubject("Q", dt.year, dt.month, dt.day, 12, 0, lat=0, lng=0, tz_str="UTC", zodiac_type="Sidereal", sidereal_mode=SIDEREAL_MODE)
            for b in BODIES:
                p = getattr(sub, ATTR.get(b, b), None); lon[b].append(float(p.abs_pos) % 360 if p is not None and hasattr(p, "abs_pos") else np.nan)
        except Exception:
            for b in BODIES: lon[b].append(np.nan)
    t_yr = (np.asarray(days, dtype="datetime64[ns]").astype("int64") / 1e9) / (365.25 * 86400.0)   # time in years
    periods = {}
    for b in BODIES:
        arr = np.asarray(lon[b], float); ok = np.isfinite(arr)
        if ok.sum() < 2: continue
        unwrapped = np.degrees(np.unwrap(np.radians(arr[ok])))          # remove 360° wraps → smooth longitude
        omega = np.polyfit(t_yr[ok], unwrapped, 1)[0]                   # deg/year (signed; node is retrograde)
        if abs(omega) > 1e-9: periods[b] = 360.0 / abs(omega)
    json.dump(periods, open(MEAN_PERIODS, "w"), indent=1)
    print("  measured periods (yr): " + ", ".join(f"{b}={periods[b]:.3f}" for b in BODIES if b in periods), flush=True)
    return periods

# ──────────────────────────── MODEL ───────────────────────────────
def _r2(y, p):
    ss = float(np.sum((y - y.mean()) ** 2))
    if ss <= 1e-9: return 0.0                       # constant series — no variance to explain (not a perfect fit)
    return 1.0 - float(np.sum((y - p) ** 2)) / ss

# A PURE COSINE kernel per body: cos(Δ) of the body's wrapped angular distance Δ to the shared trial phase. This is a
# pure sinusoid of each body's angular position, so it oscillates at exactly that body's OWN orbital period (2026-07-01:
# replaced the sinc kernel + all frequency plumbing). Fast bodies wave many times over 22 years; slow bodies (Neptune,
# Pluto) barely complete an arc, contributing a near-constant secular term. No width parameter, no FFT, no tuning.
ANNUAL_FREQ = 1.0
DEFAULT_FREQ = ANNUAL_FREQ                # unused; kept defined so any stray caller/import doesn't break
PERIOD_SCALE = ANNUAL_FREQ               # unused back-compat alias
def kern_at(dist_deg, w=1.0):
    """SINC of a body's wrapped angular distance to the trial phase, at inverse-width w = 1/PERIOD: sinc(w·deg2rad(Δ)).
       w is the per-field frequency (larger w = narrower kernel = shorter period P=1/w); tuned per keyword like the
       phase. First zero at |Δ| = 57.3/w°. Its side-lobes give a little harmonic diversity a pure cosine lacks."""
    return np.sinc(np.deg2rad(np.asarray(dist_deg, float)) * w)
def kern(dist_deg):
    """Back-compat: the sinc kernel."""
    return kern_at(dist_deg)

def eff_lon(lon):
    """Effective longitude per body for the features. The Moon is its SYNODIC PHASE — elongation from the Sun
       (moon_lon − sun_lon, period 29.53 d), the full/new-moon cycle that actually drives human behaviour —
       rather than its sidereal position. Every other body keeps its sidereal longitude."""
    if not (SYNODIC_MOON and "sun" in lon and "moon" in lon): return lon
    out = dict(lon); out["moon"] = (np.asarray(lon["moon"], float) - np.asarray(lon["sun"], float)) % 360
    return out

def designs(lon):
    """At each candidate angle r, one column per body: the PURE COSINE of the wrapped angular distance to r — a broad
       sinusoid oscillating at that body's own orbital period. The only fitted quantities are the shared phase r and
       the 11 non-negative weights. The Moon is taken SYNODIC (its phase; see eff_lon)."""
    return designs_at(lon)

def designs_at(lon, w=1.0):
    """Per-angle design matrices at inverse-width w=1/period. One column per body: sinc of the wrapped angular distance
       to the candidate phase angle, at width w."""
    lon = eff_lon(lon)
    return [np.column_stack([kern_at((lon[b] - r + 180) % 360 - 180, w) for b in BODIES]) for r in REFS]

def _huber_fit(X, y, fscale):
    """Non-negative HUBER fit of  y ≈ X·w  (w ≥ 0, NO bias/intercept — removed 2026-07-01: exactly one parameter per
       body). Huber loss downweights the spiky outliers that dominate raw Google-Trends interest under squared error.
       Returns (weights, bias=0.0, prediction, cost) — bias kept in the signature as a constant zero for callers."""
    n, p = X.shape
    x0 = np.zeros(p)
    r = least_squares(lambda b: X @ b - y, x0, bounds=(np.zeros(p), np.full(p, np.inf)), loss="huber", f_scale=fscale, max_nfev=p * 30)
    return r.x, 0.0, X @ r.x, float(r.cost)

def _f1(r2, rep):
    """F1 (harmonic mean) of the fit R² and the decisiveness rep, both mapped to [0,1] so neither dominates:
       R² clipped to [0,1]; rep (runner-up gain %) as rep/100 capped at 1 (≥100% gain → 1). 0 if either is 0."""
    a = min(max(float(r2), 0.0), 1.0); b = min(max(float(rep), 0.0) / 100.0, 1.0)
    return (2.0 * a * b / (a + b)) if (a + b) > 1e-12 else 0.0

def _fit_at_width(y, lon, w, fscale):
    """One non-negative HUBER phase sweep at inverse-width w. Returns the headline fit + its rep at this period."""
    n = len(y); des = [X[:n] for X in designs_at(lon, w)]
    costs, fits = [], []
    for X in des:
        coef, res = _nnls(X, y); pred = X @ coef                   # fast direct non-negative least squares (L2)
        costs.append(float(res)); fits.append((coef, 0.0, pred))   # res = ||Xw − y|| residual norm
    mx = max(costs); tuning = [mx - c for c in costs]              # fit quality vs angle (≥ 0)
    areas = _sign_areas(tuning); tot = float(areas.sum()) or 1.0
    ss = [round(float(areas[k] / tot), 4) for k in range(12)]
    win = int(np.argmax(areas)); _ss = sorted(ss, reverse=True)
    hr = (_ss[0] / _ss[1]) if len(_ss) > 1 and _ss[1] > 1e-9 else 99.0
    band = [j for j in range(len(REFS)) if int(REFS[j] // 30) % 12 == win]
    bi = band[int(np.argmax([tuning[j] for j in band]))]
    ref = float(REFS[bi]); coef, bias, pred = fits[bi]; r2 = float(_r2(y, pred))
    rep = rep_score({"r2": r2, "sign_scores": ss})
    return dict(w=w, tuning=tuning, ss=ss, win=win, hr=hr, ref=ref, coef=coef, pred=pred, r2=r2, rep=rep, bi=bi)

def fit_topic(y, lon):
    """EXACTLY 11 PARAMETERS: one non-negative weight per body (11 bodies), NO intercept/bias. Two things are tuned per
       keyword (each a selector, not a fitted coefficient): the shared PHASE r (swept 5°, → the sign) and the kernel
       PERIOD P (swept over PERIOD_GRID, reported in years; inverse-width w = 1/P). For every candidate P we run the
       full phase sweep and score it by F1(rep, R²); the P with the HIGHEST F1 is the field's period (min-max: NNLS
       minimises the loss inside each sweep, we maximise the balance of fit and decisiveness across periods). rec
       carries period (yr), frequency (=1/P), f1, plus the usual phase/sign/weights/R²/rep of the winning period."""
    y = np.asarray(y, float); n = len(y)
    fscale = max(1.0, 1.345 * 1.4826 * float(np.median(np.abs(y - np.median(y)))))   # Huber knee ≈ robust residual scale
    # PER-KEYWORD PERIOD tuned by MAX F1(rep, R²). The winning P must never sit at an EDGE of the plotted curve —
    # that means the true optimum lies BEYOND it, silently truncated — and the user wants it dead-CENTRE so the
    # peak is visibly surrounded. So we hold a SYMMETRIC geometric window of PK points each side of a centre
    # period (centre·PSTEP^k, k=−PK…+PK, clamped to PBOUND yr) and iteratively RE-CENTRE it on the current best:
    # if the best is not the centre we recentre there and re-search, dynamically adding options on whichever
    # side the peak drifts toward, until the best IS the middle. Each candidate P is fit once (cached).
    cache = {}
    def _ev(P):
        P = round(float(min(PBOUND[1], max(PBOUND[0], P))), 3)
        if P not in cache:
            rr = _fit_at_width(y, lon, 1.0 / P, fscale)
            cache[P] = (rr, float(_f1(rr["r2"], rr["rep"])))
        return P
    def _window(centre):                                          # PK geometric steps each side of centre (unique, clamped)
        return sorted({ _ev(centre * (PSTEP ** k)) for k in range(-PK, PK + 1) })
    for P in PERIOD_GRID: _ev(P)                                  # seed grid → a good starting centre
    grid0 = sorted(cache); centre = grid0[int(np.argmax([cache[P][1] for P in grid0]))]
    win = _window(centre)
    for _ in range(40):                                           # recentre until the winner sits in the middle
        best = max(win, key=lambda P: cache[P][1])
        if abs(best - centre) < 1e-6: break
        centre = best; win = _window(centre)
    # The full window found the (interior) winner; the PLOTTED curve shows only PSHOW steps each side of it —
    # exactly 2·PSHOW+1 = 5 bars with the resonance period dead-CENTRE (fewer only if a PBOUND clamp collapses them).
    winner = max(win, key=lambda P: cache[P][1])
    show = sorted({ _ev(winner * (PSTEP ** k)) for k in range(-PSHOW, PSHOW + 1) })
    p_curve = [{"period": P, "r2": round(cache[P][0]["r2"], 4),
                "rep": round(float(cache[P][0]["rep"]), 2), "f1": round(cache[P][1], 4)} for P in show]
    f1s = [cache[P][1] for P in show]; i = int(np.argmax(f1s)); win = show
    P = win[i]; r = cache[P][0]; f1 = f1s[i]
    tuning = r["tuning"]; mean_t = float(np.mean(tuning)); pred = r["pred"]
    sig = (tuning[r["bi"]] / mean_t * 100.0) if mean_t > 1e-9 else 0.0
    rec = dict(tuning=[float(v) for v in tuning], sign_scores=r["ss"], ref=r["ref"], sign=SIGNS[r["win"]],
               house_ratio=round(float(r["hr"]), 3), sig=float(sig), period=round(float(P), 3),
               frequency=round(1.0 / P, 5), f1=round(float(f1), 4), period_curve=p_curve,
               r2=float(r["r2"]), mse=float(np.mean((y - pred) ** 2)), rmse=float(np.mean((y - pred) ** 2) ** 0.5),
               bias=0.0, weights={HKEY[i]: float(c) for i, c in enumerate(r["coef"])}, fitted=pred.tolist())
    rec["rep"] = r["rep"]
    return rec

def _sign_areas(tuning):
    """Area (∫ fit-quality) under each of the 12 sign bands of the tuning curve (REFS = 5° grid → sum over 30° each)."""
    t = np.asarray(tuning, float); refs = np.asarray(REFS, float)
    return np.array([float(t[(refs >= k * 30) & (refs < k * 30 + 30)].sum()) for k in range(12)])


def seasonal_sign(y, lon):
    """SUN-ANCHORED seasonal sign (2026-06-30): the sidereal sign the field's interest genuinely PEAKS in. The
       multi-body fit's phase is dragged off-season by the slow planets (they fit the multi-year trend), so the SIGN
       is defined separately + cleanly: de-trend the series (remove a centred 13-month rolling median → the seasonal
       anomaly), bin it by the Sun's sidereal sign each month, and take the sign of the largest mean anomaly.
       Returns (win, sign_scores[12], house_ratio). The 11-body fit still supplies R²/body-shares."""
    yv = np.asarray(y, float); n = len(yv)
    sl = np.asarray(lon["sun"], float)[:n]
    tr = pd.Series(yv).rolling(13, center=True, min_periods=1).median().to_numpy()
    sea = yv - tr                                                  # seasonal anomaly (trend removed)
    sidx = (sl // 30).astype(int) % 12
    means = np.array([sea[sidx == k].mean() if (sidx == k).any() else 0.0 for k in range(12)])
    means = means - means.min()                                   # shift so the trough sign = 0 → areas ≥ 0
    tot = float(means.sum()) or 1.0
    scores = [round(float(means[k] / tot), 4) for k in range(12)]
    win = int(np.argmax(means))
    ss = sorted(means, reverse=True)
    ratio = (ss[0] / ss[1]) if len(ss) > 1 and ss[1] > 1e-9 else 1.0
    return win, scores, float(ratio)

def rep_score(rec):
    """Representativeness = the runner-up GAIN as a PERCENT: (max(area)/second-max(area) − 1) × 100 over the 12
       sign-area scores — how much bigger, in %, the primary house's sign-area is than the runner-up's (5 → 5% bigger,
       100 → twice as big). GATED: a fit no better than the mean (R² ≤ 0) or with a vanishing runner-up has no real
       seasonality → 0 (no decisiveness). The underlying ratio is capped at 10× (so rep ≤ 900) to stop degenerate
       blow-ups dominating the ranking. (2026-07-01: rescaled from the raw ratio 1..10 to (ratio−1)×100.)"""
    if float(rec.get("r2", 0.0)) <= 0.0:                           # garbage fit (≤ mean) → not decisive about anything
        return 0.0
    scores = rec.get("sign_scores")
    if not scores:                                                 # legacy rec → derive the 12-vector from the tuning curve
        a = _sign_areas(rec["tuning"]); t = float(a.sum()) or 1.0; scores = (a / t).tolist()
    s = sorted([float(v) for v in scores], reverse=True) if scores else [0.0]
    if len(s) < 2 or s[1] <= 1e-9:
        return 0.0
    return (min(s[0] / s[1], 10.0) - 1.0) * 100.0                  # runner-up GAIN % (cap 10× → ≤ 900)

def body_shares(rec, lon, y):
    """Simple variance decomposition: flatten each body's feature to its MEAN (removing its variation, keeping the
       fitted weights) and recompute the empirical R²; the DROP from the full R² is that body's contribution,
          v_b = R²_full − R²(body b set to its mean) = Var(weight_b · feature_b) / Var(search)  ≥ 0.
       The twelve v_b are then normalised to sum EXACTLY to R². Non-negative (it's a variance), sums to R², and far
       simpler than a Shapley refit — just 12 recomputations, no refitting. Needs the actual series y."""
    ref = float(rec["ref"]); n = len(y); y = np.asarray(y, float)
    iw = float(rec.get("frequency", 1.0))                            # the field's tuned inverse-width (1/period)
    sst = float(np.sum((y - y.mean()) ** 2)) or 1.0
    w = np.array([rec["weights"][hk] for hk in HKEY])
    el = eff_lon(lon)                                                # Moon → synodic phase, matching designs()
    X = np.column_stack([kern_at((el[b][:n] - ref + 180) % 360 - 180, iw) for b in BODIES])
    pred = float(rec["bias"]) + X @ w
    r2_full = 1.0 - float(np.sum((y - pred) ** 2)) / sst
    v = np.empty(len(BODIES))
    for i in range(len(BODIES)):
        pred_i = pred - w[i] * (X[:, i] - X[:, i].mean())          # body i flattened to its mean value
        v[i] = r2_full - (1.0 - float(np.sum((y - pred_i) ** 2)) / sst)   # R² lost ⇒ Var(w·f)/Var(y) ≥ 0
    v = np.clip(v, 0.0, None); tot = float(v.sum())
    return {HKEY[i]: float(rec["r2"] * v[i] / tot if tot > 1e-12 else 0.0) for i in range(len(BODIES))}   # Σ = R²

def recompute_shares(reg, lon):
    """(Re)compute every topic's variance R²-shares from its cached series — needs no network. Returns a y-cache."""
    ys = {}
    for r in reg.values():
        _, y = load_y(r["label"]); ys[r["key"]] = y
        r["shares"] = body_shares(r, lon, y) if y is not None else {hk: 0.0 for hk in r["weights"]}
        r["rep"] = rep_score(r)                                          # representativeness % (amplitude × centredness)
    save_reg(reg); print(f"  computed variance R²-shares + representativeness for {len(reg)} topics", flush=True)
    return ys

# ───────────────────────────── STYLE + CHARTS ─────────────────────
GOLD, BLUE, INK, MUTE = "#E8B923", "#2352E8", "#0C1E32", "#6B7A90"
plt.rcParams.update({"axes.edgecolor": "#d8deea", "axes.linewidth": .8, "grid.color": "#eef1f7",
                     "xtick.color": MUTE, "ytick.color": MUTE, "text.color": INK, "axes.labelcolor": MUTE,
                     "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 11})
STYLE = """<meta name=viewport content="width=device-width,initial-scale=1"><style>
:root{--gold:#E8B923;--blue:#2352E8;--ink:#0C1E32;--mute:#6B7A90;--bg:#F6F8FC;--card:#fff;--line:#E7EBF3}
*{box-sizing:border-box}body{margin:0;font:16px/1.65 -apple-system,"Segoe UI",system-ui,sans-serif;color:var(--ink);background:var(--bg)}
.wrap{max-width:860px;margin:0 auto;padding:2.2rem 1.2rem 4rem}
h1{font-size:2rem;line-height:1.15;margin:.3em 0}h2{font-size:1.2rem;margin:1.6em 0 .4em}
.lede{font-size:1.15rem;color:#2a3a4f}.cap{font-size:.85rem;color:var(--mute);margin:.6em 0 0}
.big{font-size:2.5rem;font-weight:800;color:var(--gold);line-height:1}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:1.1rem;margin:1.1rem 0;box-shadow:0 1px 3px rgba(12,30,50,.05)}
.card img{width:100%;display:block;border-radius:8px}.back{color:var(--blue);text-decoration:none;font-weight:600;font-size:.95rem}
.hero{text-align:center;padding:1.4rem 0 .4rem}.hero h1{font-size:2.3rem}.hero p{font-size:1.15rem;color:#2a3a4f;max-width:640px;margin:.6em auto 0}
.row{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.55rem 0;border-bottom:1px solid var(--line)}
.row a{font-weight:600;text-decoration:none;color:var(--ink)}.row .v{font-weight:800;color:var(--gold);white-space:nowrap}.row .m{color:var(--mute);font-weight:500;font-size:.88rem}
.foot{color:var(--mute);font-size:.82rem;margin-top:2.4rem;border-top:1px solid var(--line);padding-top:1rem}</style>"""

def _svg(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="svg", bbox_inches="tight", facecolor="white"); plt.close(fig)
    return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode()

def tuning_chart(tuning, ref):
    tv = np.array(tuning) * 100
    f, a = plt.subplots(figsize=(9.8, 3.5)); bs = int(ref // 30) * 30
    a.axvspan(bs, bs + 30, color=GOLD, alpha=.16, zorder=0)
    a.plot(REFS, tv, color=BLUE, lw=2.5, zorder=3); a.axvline(ref, color=GOLD, lw=2.4, zorder=2)
    a.scatter([ref], [tv[int(round(ref / 5)) % len(tv)]], s=80, color=GOLD, edgecolor="white", lw=1.6, zorder=4)
    hi, lo = tv.max(), tv.min(); span = max(hi - lo, .1)
    for i, sg in enumerate(SIGNS):
        a.axvline(i * 30, color="#e3e8f2", lw=.8, zorder=1)
        a.text(i * 30 + 15, hi + span * .05, sg[:3], ha="center", va="bottom", fontsize=8, color="#5b6b82", zorder=3)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    a.set_xlim(0, 360); a.set_ylim(lo - span * .12, hi + span * .2); a.set_xticks(range(0, 361, 60))
    a.set_xlabel("phase — position around the zodiac (°)"); a.set_ylabel("fit quality (higher = lower error)")
    a.grid(axis="y", alpha=.5); return _svg(f)

def fit_chart(t, y, fitted):
    f, a = plt.subplots(figsize=(9.8, 3.3))
    a.plot(t, y, color=BLUE, lw=.8, alpha=.45, label="what the world searched")
    a.plot(t, fitted, color=GOLD, lw=2.2, label="what the sky predicts")
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    a.grid(axis="y", alpha=.5); a.legend(frameon=False, fontsize=10.5, loc="upper left"); a.set_ylabel("search interest (cube-root)")
    return _svg(f)

def cycles_chart(items):
    """items = ALL 12 (body:k, R²-contribution) pairs, already sorted by contribution (signed)."""
    labs = [h_label(*_parse(hk)) for hk, _ in items]
    vals = [sh * 100 for _, sh in items]
    f, a = plt.subplots(figsize=(9.8, max(1.2, .42 * len(items))))
    a.barh(labs, vals, color=[GOLD] + ["#9fb6e0"] * (len(items) - 1)); a.invert_yaxis()
    a.axvline(0, color="#d8deea", lw=.8)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    a.set_xlabel("share of the 22-year search history explained — R² contribution (%)")
    a.grid(axis="x", alpha=.5); return _svg(f)

# ───────────────────────────── PAGES ──────────────────────────────
def _share(rec, hk):
    """A body's signed R²-contribution within this topic (Cov(w·feature, fitted)/Var(search)); sums to R²."""
    return rec.get("shares", {}).get(hk, 0.0)

def render_topic(rec, t, y):
    allb = sorted(rec.get("shares", {}).items(), key=lambda x: -x[1])   # ALL 12 bodies, by R²-contribution, no threshold
    r2 = rec["r2"] * 100
    topb, topk = _parse(allb[0][0]) if allb else (BODIES[0], 1)
    rows = "".join(f'<div class=row><a href="cycle-{h_slug(*_parse(hk))}.html">{h_label(*_parse(hk))}</a>'
                   f'<span class=v>{pct(sh)}</span></div>' for hk, sh in allb)
    total = f'<div class=row style="border-top:2px solid var(--line)"><span style="font-weight:600">Together (= R²)</span><span class=v>{r2:.1f}%</span></div>'
    cyc = (f'<h2>What each body contributes</h2>'
           f'<div class=card><img src="{cycles_chart(allb)}"></div>'
           f'<div class=card>{rows}{total}</div>'
           f'<p class=cap>Each figure is that body\'s share of the 22-year search history the fit explains — found by flattening the body to its mean value and recomputing R²; the drop is its contribution, Var(weight·cycle)/Var(search). Non-negative, and the twelve are normalised to sum to R². This is what really drives the rhythm — not the raw weights, which are huge and cancelling for the near-constant slow bodies.</p>') if allb else ''
    fitsec = (f'<h2>The rhythm it fits</h2>'
              f'<div class=card><img src="{fit_chart(t, y, np.array(rec["fitted"]))}"><p class=cap><b style="color:#2352E8">Blue</b> — what the world actually searched. <b style="color:#b8901a">Gold</b> — the rhythm the sky predicts.</p></div>') if y is not None else ''
    html = f"""<!doctype html><html lang=en><meta charset=utf-8><title>{rec['label']} — its sidereal cycle</title>{STYLE}
<div class=wrap><a class=back href="index.html">← The atlas</a>
<h1>{rec['label']}</h1>
<p class=lede>Across 22 years, the world's search for {rec['label'].lower()} keeps best time with one angle of the zodiac:</p>
<p><span class=big>{rec['sign']}</span></p>
<p class=cap>its angle: {round(rec['ref'])}° around the wheel · {round(rec['ref'])%30}° into the sign</p>
<p class=lede style="margin-top:1.3rem">A least-squares fit of the twelve bodies explains <b>{r2:.1f}%</b> of its 22-year monthly search history (in-sample). Its strongest single cycle is <b>{h_label(topb, topk)}</b>.</p>
<p class=lede><b>Representativeness {rec.get('rep',0)*100:.1f}%</b> — R² ({r2:.1f}%) × the share of the fit-quality area under the tuning curve that falls inside {rec['sign']}'s 30° (rather than spread around the zodiac).</p>
<h2>How sharp is the match?</h2>
<div class=card><img src="{tuning_chart(rec['tuning'], rec['ref'])}"><p class=cap>How well the bodies fit at every angle around the zodiac. The gold band marks the winner — <b>{rec['sign']}</b>; a clear peak means a strong preference, a flat line means none.</p></div>
{fitsec}
{cyc}
<p class=foot><b>Correlation is not causation.</b> This is a statistical curiosity — it only measures whether two rhythms happen to line up over 22 years. Nothing here claims the sky causes anything, or that astrology is real.<br>Monthly Google-Trends, 2004 to a year ago (recent year excluded to avoid real-time bias; R² is in-sample) · {len(BODIES)} sidereal bodies · sinc features scaled by orbital period · cube-root-transformed interest · non-negative least squares (each body only adds) · sign = the angle of lowest mean-squared-error · body figures = variance share (flatten-to-mean R² drop, normalised to sum to R²)</p>
</div>"""
    open(f"{PAGES_DIR}/{rec['key']}.html", "w").write(html)

def render_cycle(b, k, reg):
    """One page per cycle — EVERY topic, ranked by this body's signed R²-contribution to it (no threshold)."""
    hk = f"{b}:{k}"; yv = h_years(b, k)
    users = sorted(((r, _share(r, hk)) for r in reg.values()), key=lambda x: -x[1])
    rows = "".join(f'<div class=row><a href="{r["key"]}.html">{r["label"]}</a>'
                   f'<span class=v>{pct(sh)} <span class=m>· {r["sign"]}</span></span></div>'
                   for r, sh in users) or '<p class=cap>No topics yet.</p>'
    kind = f"the rhythm of {art(b)}" if k == 1 else f"a faster harmonic of {art(b)}'s {period_words(PERIOD_YEARS[b])} cycle"
    html = f"""<!doctype html><html lang=en><meta charset=utf-8><title>{nice(b)} · {period_words(yv)} — a sidereal cycle</title>{STYLE}
<div class=wrap><a class=back href="cycles.html">← All cycles</a>
<h1>{nice(b)} · {period_words(yv)}</h1>
<p class=lede>A rhythm that repeats <b>{period_words(yv)}</b> — {kind}.</p>
<h2>Every topic, ranked by this cycle's R² contribution</h2>
<div class=card>{rows}</div>
<p class=cap>Each figure is the share of that topic's 22-year search history this one cycle explains — found by flattening the cycle to its mean and recomputing R² (the drop = Var(weight·cycle)/Var(search)). Non-negative; per topic all twelve are normalised to sum to R². All topics are listed.</p>
<p class=foot><b>Correlation is not causation.</b> A statistical curiosity — not a claim of cause, nor that astrology is real.</p>
</div>"""
    open(f"{PAGES_DIR}/cycle-{h_slug(b, k)}.html", "w").write(html)

def render_houses(reg):
    """Each SIGN is a house. Its TITLE (★) = the most REPRESENTATIVE keyword on it (highest rep = √(R²·centredness)).
       Every keyword sharing the sign is listed, ranked by representativeness; an off-centre angle is flagged."""
    central = lambda r: abs((r["ref"] % 30) - 15) <= 10                       # within 10° of the sign centre
    BY = {s: [] for s in SIGNS}
    for r in reg.values(): BY[r["sign"]].append(r)
    blocks = []
    for s in SIGNS:
        rs = sorted(BY[s], key=lambda r: -r.get("rep", 0.0))                 # rank by representativeness
        title = rs[0] if rs else None                                        # most representative keyword on the sign
        rows = ("".join(f'<div class=row><a href="{r["key"]}.html">{"★ " if r is title else ""}{r["label"]}'
                        f'{"" if central(r) else " <span class=m>· off-centre</span>"}</a>'
                        f'<span class=v>{r.get("rep",0)*100:.1f}% <span class=m>· R² {r["r2"]*100:.1f}%</span></span></div>' for r in rs)
                if rs else '<p class=cap>No keyword landed on this sign.</p>')
        tt = f'<span style="color:var(--gold)">{title["label"]}</span>' if title else '<span class=m>— none</span>'
        blocks.append(f'<h2>{s} → {tt}</h2><div class=card>{rows}</div>')
    html = f"""<!doctype html><html lang=en><meta charset=utf-8><title>ArtaQuest — the twelve sign-houses</title>{STYLE}
<div class=wrap><div class=hero><h1>The twelve houses</h1>
<p>Each sign is a house. Its title (★, gold) is the most <b>representative</b> keyword on it — highest rep% = R² × the share of its tuning-curve area that falls inside this sign. Every keyword is listed, ranked by rep%.</p>
<p><a class=back href="index.html">← The atlas</a></p></div>
{"".join(blocks)}
<p class=foot><b>Correlation is not causation.</b> A statistical curiosity, not a claim of cause — and not a claim that astrology is real.</p></div>"""
    open(f"{PAGES_DIR}/houses.html", "w").write(html)

def rebuild(reg, shares=True, topics=True):
    """shares=False skips the (resolution-specific) Shapley/variance recompute — use it when rebuilding the index over
       a MIXED-resolution registry so monthly topics' stored shares aren't recomputed against a weekly ephemeris.
       topics=False skips re-rendering individual topic pages (just refresh index/houses/cycles)."""
    ys = {}
    if shares:
        lon = ephemeris(); ys = recompute_shares(reg, lon)             # offline: variance R²-shares from cached series
    n_t = 0
    if topics:
        for r in reg.values():                                        # re-render EVERY topic page from cache (no refetch)
            t, _ = load_y(r["label"]); render_topic(r, t, ys.get(r["key"])); n_t += 1   # y=None ⇒ breakdown only
    render_houses(reg)
    recs = sorted(reg.values(), key=lambda r: -r.get("rep", 0.0))
    rows = "".join(f'<div class=row><a href="{r["key"]}.html">{r["label"]}</a>'
                   f'<span class=v>{r.get("rep",0)*100:.1f}% <span class=m>· {r["sign"]} · R² {r["r2"]*100:.1f}%</span></span></div>' for r in recs)
    html = f"""<!doctype html><html lang=en><meta charset=utf-8><title>ArtaQuest — sidereal trend atlas</title>{STYLE}
<div class=wrap><div class=hero><h1>The sidereal trend atlas</h1>
<p>For every topic we find the angle of the zodiac whose celestial rhythms best fit 22 years of the world's monthly searches for it. Ranked by <b>representativeness</b> — R² × the share of the fit-quality area that falls inside the assigned sign: strong fits whose quality concentrates in one sign rank highest.</p>
<p><a class=back href="cycles.html">Browse by cycle →</a> &nbsp;·&nbsp; <a class=back href="houses.html">The 12 sign-houses →</a></p></div>
<div class=card style="text-align:center"><span class=big>{len(recs)}</span> topics mapped · {len(TERMS_H)} cycles</div>
<div class=card>{rows or '<p class=cap>None yet.</p>'}</div>
<p class=foot><b>Correlation is not causation.</b> A statistical curiosity, not a claim of cause — and not a claim that astrology is real. Monthly Google-Trends, 2004 to a year ago (recent year excluded; R² is in-sample) · {len(BODIES)} sidereal bodies · sinc features scaled by orbital period · cube-root-transformed interest · non-negative least squares (each body only adds) · sign = the angle of lowest mean-squared-error. analysis/trends_fit.py</p></div>"""
    open(f"{PAGES_DIR}/index.html", "w").write(html)
    for (b, k) in TERMS_H: render_cycle(b, k, reg)                     # one page per cycle
    lead = lambda hk: sum(1 for r in reg.values() if r.get("shares") and max(r["shares"], key=r["shares"].get) == hk)
    crows = "".join(f'<div class=row><a href="cycle-{h_slug(b,k)}.html">{h_label(b,k)}</a>'
                    f'<span class=v>{lead(f"{b}:{k}")} <span class=m>topics</span></span></div>'
                    for (b, k) in sorted(TERMS_H, key=lambda bk: h_years(*bk)))
    chtml = f"""<!doctype html><html lang=en><meta charset=utf-8><title>ArtaQuest — the cycles</title>{STYLE}
<div class=wrap><div class=hero><h1>The cycles</h1>
<p>Every rhythm the model can lean on, from the fastest (a month) to the slowest (centuries). Each page ranks every topic by this cycle's R² contribution; the count is how many topics it is the <b>top</b> contributor for.</p>
<p><a class=back href="index.html">← Back to topics</a></p></div>
<div class=card>{crows}</div>
<p class=foot><b>Correlation is not causation.</b> A statistical curiosity — not a claim of cause, nor that astrology is real.</p></div>"""
    open(f"{PAGES_DIR}/cycles.html", "w").write(chtml)
    print(f"  → rebuilt {n_t} topic pages, index, houses, {len(TERMS_H)} cycle pages")

# ────────────────────────────── MAIN ──────────────────────────────
def load_reg(): return json.load(open(REGISTRY)) if os.path.exists(REGISTRY) else {}
def save_reg(r): json.dump(r, open(REGISTRY, "w"), indent=1)

def is_single_word(label): return len(str(label).split()) == 1     # one token, no spaces (a clean Trends query)

def topic_order():
    """SINGLE-WORD keywords only — the 12 field terms, then the single-word disciplines. Multi-word topics are
       purged: a multi-word phrase is a noisy, ambiguous Google-Trends query."""
    items = [{"key": slug(t), "label": t} for t in TERMS]
    if os.path.exists(DISC_FILE):
        for d in json.load(open(DISC_FILE)).get("disciplines", []):
            items.append({"key": d["key"], "label": d["label"]})
    seen, out = set(), []
    for it in items:
        if not is_single_word(it["label"]): continue                # purge multi-word
        if it["key"] not in seen: seen.add(it["key"]); out.append(it)
    return out

def checkpoint(reg):
    """Push to prod every 10 iterations: rebuild the index + commit the analysis (explicit paths)."""
    import subprocess
    rebuild(reg)
    subprocess.run(["git", "add", "analysis/trends_fit.py", "analysis/topics.json", "analysis/data_monthly", "analysis/pages"], check=False)
    subprocess.run(["git", "commit", "-q", "-m", f"analysis: weekly sidereal atlas checkpoint — {len(reg)} topics"], check=False)
    print(f"    [checkpoint: committed {len(reg)} topics]", flush=True)

def y_transform(a):
    """IDENTITY — the model now fits RAW search interest directly (the cube-root normalisation was purged,
       operator 2026-06-27). Every fit, R², residual and tuning curve is on the raw 0–100 scale the chart shows,
       so what you see is exactly what the model fits. Kept as a function so call-sites need no change."""
    return np.asarray(a, float)

def clean_y(df):
    """Trends series → clean RAW float array on GRID (cube-root purged 2026-06-27; y_transform is now identity), or
       None if too sparse to trust (a failed-chunk gap). Interpolates small gaps; the model fits this directly."""
    ys = pd.to_numeric(df["v"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if ys.notna().sum() < len(GRID) * 0.5: return None
    y = ys.interpolate(limit_direction="both").ffill().bfill().values.astype(float)
    return y_transform(y) if np.isfinite(y).all() else None

def load_y(label):
    """Reload a topic's cached monthly series (the fit window, last year dropped) for re-rendering — no refetch.
       Keyed by the label slug (how fetch_daily cached it). Returns (times, y) or (None, None) if absent/partial."""
    p = f"{DATA_DIR}/{slug(label)}.csv"
    if not os.path.exists(p): return None, None
    df = pd.read_csv(p); df["Time"] = pd.to_datetime(df["Time"])
    M = df.drop_duplicates("Time").set_index("Time")["v"].reindex(GRID)   # align to the monthly GRID (robust to off-by-one len)
    y = clean_y(pd.DataFrame({"v": M.values}))
    if y is None: return None, None
    return np.asarray(GRID)[:-DROP_LAST], y[:-DROP_LAST]

def refit_all(reg):
    """Re-run the (MSE) fit for EVERY stored topic from its cached series — no network. Selection by MSE can move an
       angle vs. the old MAE pick, so this is a true rerun: ref/sign/weights/fitted/r²/sig are all recomputed."""
    lon = ephemeris(); n = 0                                  # fit_topic sweeps the shared phase itself → it takes lon
    for k, r in list(reg.items()):
        _, y = load_y(r["label"])
        if y is None: print(f"  skip {r['label']} (no cached series)"); continue
        rec = fit_topic(y, lon); rec.update(key=r["key"], label=r["label"]); reg[k] = rec; n += 1
    save_reg(reg); print(f"  refit {n} topics from cache (MSE selection)", flush=True)
    rebuild(reg)                                                    # rebuild computes variance shares + renders all pages

def main():
    reg = load_reg()
    if "--refit" in sys.argv: refit_all(reg); return
    if "--rebuild" in sys.argv: rebuild(reg); return
    todo = [it for it in topic_order() if it["key"] not in reg]
    if "--n" in sys.argv: todo = todo[:int(sys.argv[sys.argv.index("--n") + 1])]
    if not todo: print("  all topics done"); rebuild(reg); return
    print(f"[{len(todo)} topics to do — monthly data · pure-cosine kernel · NNLS · Huber tuning · R²-shares]")
    s = None
    try: s = make_session(); print(f"[auth] {len(s.cookies)} cookies")
    except Exception as e: print(f"[auth] no cookies ({e})")
    lon = ephemeris(); miss = [b for b in BODIES if b not in lon]
    if miss: print(f"  ⚠ missing ephemeris for {miss} — aborting"); return
    print(f"  {len(BODIES)} bodies: " + ", ".join(nice(b) for b in BODIES) + f"  ·  fit window {START.year}–{(END - pd.DateOffset(months=DROP_LAST)).year}")
    n = 0
    for it in todo:
        print(f"  {it['label']}", flush=True)
        df = fetch_daily(it["label"], s)
        if df is None or len(df) != len(GRID): print(f"    skip (no/partial data)"); continue
        y = clean_y(df)
        if y is None: print(f"    skip (too sparse / unfillable)"); continue
        y = y[:-DROP_LAST]; tt = df["Time"].values[:-DROP_LAST]   # drop the most recent year (recency bias)
        rec = fit_topic(y, lon); rec.update(key=it["key"], label=it["label"])
        rec["shares"] = body_shares(rec, lon, y)         # per-body variance R²-share (non-negative, sums to R²)
        reg[it["key"]] = rec; render_topic(rec, tt, y); save_reg(reg)
        print(f"    → {round(rec['ref']):3d}° {rec['sign']:11s} R² {round(rec['r2']*100):3d}%  RMSE {rec['rmse']:.2f}", flush=True)
        n += 1
        if n % 10 == 0: checkpoint(reg)                  # ship every 10 iterations
    checkpoint(reg)

if __name__ == "__main__":
    main()
