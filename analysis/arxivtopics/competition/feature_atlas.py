#!/usr/bin/env python3
"""THE FEATURE ATLAS — every astrological and numerological feature, one at a time.

For each feature x: a TWO-PARAMETER logistic regression, logit P(gains slice) = a + b*x, fitted on
the 30 years before the 1997 wall, then scored on 1997-2024 by within-year AUC. Two parameters is
the whole model, so nothing can hide in an interaction, and because a 2-param logistic is monotone
in x the AUC is the feature's own separating power with its DIRECTION (the sign of b) fixed on
train. A feature whose direction flips after the wall therefore scores BELOW 0.5 — which is the
point: this is an out-of-sample test, not a description of the training years.

Every feature carries a plain-language explanation of what it claims. Features that are a function
of the YEAR ALONE are included on purpose and marked CONTROL: the metric is computed inside each
year, so they cannot deviate from 0.5 except by accident, and they calibrate the noise floor.

  python3 analysis/arxivtopics/competition/feature_atlas.py
"""
import os, sys, json, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import arxiv_fit as af

BUN = os.path.expanduser("~/.artaquest-dev/artacomp/piecomp")
tr = pd.read_csv(f"{BUN}/train.csv"); te = pd.read_csv(f"{BUN}/test.csv"); sol = pd.read_csv(f"{BUN}/solution.csv")
yte = sol.set_index("id").loc[te["id"]]["target"].to_numpy()
WALL_Y, WINDOW = 1997, 30
ALL = pd.concat([tr[["field","year","target"]], te[["field","year"]].assign(target=yte)], ignore_index=True)
Y = ALL["target"].to_numpy(); YR = ALL["year"].to_numpy(); FLD = ALL["field"].to_numpy()
names, Yv, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
years = [int(y) for y in labels]; Y0 = years[0]
BODY = ["Mars","Jupiter","Saturn","Uranus","Neptune","Pluto","Lunar Node"]
PERIOD = np.array([1.88, 11.86, 29.46, 84.0, 164.8, 248.0, 18.6])
birth = tr.groupby("field")["year"].min().to_dict()
for f in te["field"].unique(): birth.setdefault(f, WALL_Y)
B = np.array([birth[f] for f in FLD]); AGE = (YR - B).astype(float)
NAT = TH[B - Y0]; CUR = TH[YR - Y0]                       # (rows, 7)
AYAN = np.deg2rad(23.85)
TAU = 2*np.pi
FIT = (YR < WALL_Y) & (YR >= WALL_Y - WINDOW); TEST = YR >= WALL_Y

def wauc(y, s, yrs):
    o = []
    for v in np.unique(yrs):
        m = yrs == v
        if len(set(y[m])) < 2: continue
        o.append(roc_auc_score(y[m], s[m]))
    return float(np.mean(o))

CAT = []                                                   # (name, tradition, explanation, vector)
def add(name, trad, expl, vec):
    v = np.asarray(vec, float)
    if np.nanstd(v) < 1e-12: return
    CAT.append((name, trad, expl, np.nan_to_num(v)))

SIGNS = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
ASPECT = [("conjunction",0.0,10,"the two bodies occupy the same degree — the classical union, said to fuse their natures"),
          ("opposition",np.pi,10,"exactly across the zodiac — the classical tension of two forces pulling apart"),
          ("trine",TAU/3,8,"120 degrees apart — traditionally the easiest and most fortunate angle"),
          ("square",np.pi/2,8,"90 degrees apart — traditionally the angle of friction and forced action"),
          ("sextile",np.pi/6*2,6,"60 degrees apart — a mild supportive angle, opportunity rather than event")]

# ── 1. transits of each body back to its own natal degree ──
for i, b in enumerate(BODY):
    d = (CUR[:,i] - NAT[:,i]) % TAU
    add(f"transit_{b}_to_natal_cos", "Western transits",
        f"How near transiting {b} is to the degree it held when the field was founded. Peaks at the "
        f"{b} return (every {PERIOD[i]:.1f} years), the classical marker of a new chapter in that "
        f"planet's affairs.", np.cos(d))
    add(f"transit_{b}_to_natal_sin", "Western transits",
        f"The quarter-cycle phase of transiting {b} against its natal degree — positive while {b} "
        f"is separating from the return, negative while it is applying back to it.", np.sin(d))
    for h, hname in ((2,"opposition axis"),(3,"trine axis"),(4,"square axis"),(6,"sextile axis")):
        add(f"harmonic{h}_{b}", "Harmonic astrology",
            f"The {h}th harmonic of transiting {b} against its natal degree — the {hname}. Harmonic "
            f"astrology holds that dividing the circle by {h} exposes a distinct layer of meaning.",
            np.cos(h*d))
    for an, ang, orb, why in ASPECT:
        sep = np.abs(((d - ang + np.pi) % TAU) - np.pi)
        add(f"aspect_{an}_{b}_natal", "Western aspects",
            f"Transiting {b} within {orb} degrees of {an} to its own natal place: {why}. Scored 1 at "
            f"exact and fading to 0 at the edge of orb.", np.clip(1 - sep/np.deg2rad(orb), 0, 1))
    add(f"return_count_{b}", "Western transits",
        f"How many complete {b} returns the field has lived through ({PERIOD[i]:.1f} years each) — "
        f"its age measured on {b}'s clock rather than the calendar's.", np.floor(AGE/PERIOD[i]))
    add(f"return_phase_{b}", "Western transits",
        f"Fraction of the current {b} cycle elapsed since the field's founding, 0 at a return and "
        f"approaching 1 just before the next.", (AGE/PERIOD[i]) % 1.0)
    add(f"natal_{b}_sign_cos", "Natal chart",
        f"Where {b} stood at the field's founding, as a smooth cyclic coordinate — the natal placement "
        f"itself, the fixed promise rather than any passing transit.", np.cos(NAT[:,i]))
    add(f"natal_{b}_sign_sin", "Natal chart",
        f"The companion coordinate to the above; together they locate natal {b} in the zodiac without "
        f"the artificial break at 0 degrees Aries.", np.sin(NAT[:,i]))
    add(f"CONTROL_transiting_{b}_position", "CONTROL",
        f"Where {b} stands in year t, ignoring the field entirely. Identical for every field in a "
        f"given year, so within-year AUC cannot leave 0.5 except by rounding — a noise-floor gauge.",
        np.cos(CUR[:,i]))

# ── 2. transiting body to natal OTHER body (the 49-cell grid) ──
for i, bi in enumerate(BODY):
    for k, bk in enumerate(BODY):
        if i == k: continue
        d = (CUR[:,i] - NAT[:,k]) % TAU
        add(f"transit_{bi}_to_natal_{bk}", "Western transits",
            f"Transiting {bi} against the degree natal {bk} occupies. The workhorse claim of predictive "
            f"astrology: a moving planet contacting a birth placement times events in that placement's "
            f"affairs.", np.cos(d))

# ── 3. Vedic / sidereal ──
NATS = (NAT - AYAN) % TAU; CURS = (CUR - AYAN) % TAU
for i, b in enumerate(BODY):
    off = (np.floor(CURS[:,i]/TAU*27) - np.floor(NATS[:,i]/TAU*27)) % 27
    add(f"nakshatra_offset_{b}", "Vedic",
        f"How many of the 27 nakshatras (lunar mansions) transiting {b} sits from its natal mansion. "
        f"Vedic astrology reads the mansion, not the 12-fold sign, as the finer unit of fate.",
        np.cos(TAU*off/27))
    add(f"sidereal_transit_{b}", "Vedic",
        f"The same transit-to-natal angle as the Western measure but in the sidereal zodiac, which is "
        f"tied to the fixed stars rather than the equinox — the two drift ~24 degrees apart today.",
        np.cos((CURS[:,i]-NATS[:,i]) % TAU))
DASHA = [("Ketu",7),("Venus",20),("Sun",6),("Moon",10),("Mars",7),("Rahu",18),("Jupiter",16),("Saturn",19),("Mercury",17)]
def dasha_idx(nat_node, age):
    pos = (nat_node % TAU)/TAU*27.0; k = int(pos) % 27; frac = pos - int(pos)
    idx = k % 9; rem = (1-frac)*DASHA[idx][1]; t = age
    if t < rem: return idx, 1 - (rem - t)/DASHA[idx][1]
    t -= rem; idx = (idx+1) % 9
    while t >= DASHA[idx][1]: t -= DASHA[idx][1]; idx = (idx+1) % 9
    return idx, t/DASHA[idx][1]
DI = np.array([dasha_idx(NATS[r,6], AGE[r]) for r in range(len(AGE))], dtype=object)
dlord = np.array([d[0] for d in DI]); dthru = np.array([float(d[1]) for d in DI])
for li,(ln,ly) in enumerate(DASHA):
    add(f"dasha_{ln}", "Vedic (Vimshottari)",
        f"Whether the field is currently running its {ln} mahadasha, the {ly}-year chapter in the "
        f"120-year Vimshottari sequence. The sequence is seeded by the natal lunar node's mansion "
        f"(classically the Moon's — we carry the node), so each field runs its own schedule.",
        (dlord == li).astype(float))
add("dasha_progress", "Vedic (Vimshottari)",
    "How far the field is through its current mahadasha, 0 at the start and 1 at the handover. "
    "Classical practice reads the opening and closing of a period as its most eventful stretches.", dthru)

# ── 4. Chinese ──
bb, cb = (B-4) % 12, (YR-4) % 12; bs, cs = (B-4) % 10, (YR-4) % 10
db, ds = (cb-bb) % 12, (cs-bs) % 10
add("chinese_branch_offset", "Chinese", "Distance between the field's founding earthly branch (its "
    "zodiac animal) and the current year's, on the 12-year cycle.", np.cos(TAU*db/12))
add("chinese_stem_offset", "Chinese", "Distance between the founding heavenly stem and the current "
    "year's on the 10-year cycle — the elemental half of the sexagenary calendar.", np.cos(TAU*ds/10))
add("chinese_trine_harmony", "Chinese", "Whether the current year's animal belongs to the same "
    "four-animal harmony group as the field's founding animal (branches four apart), traditionally "
    "the most auspicious relation between years.", (db % 4 == 0).astype(float))
add("chinese_clash", "Chinese", "Whether the current animal directly opposes the founding animal "
    "(six branches apart) — the classical clash year, held to bring upheaval.", (db == 6).astype(float))
add("sexagenary_phase", "Chinese", "Position in the full 60-year stem-and-branch cycle since "
    "founding; a complete return of the sexagenary calendar marks a life's full round.", np.cos(TAU*(AGE % 60)/60))

# ── 5. Mayan ──
for mod, nm, why in ((260,"tzolkin","the 260-day sacred round, the core divinatory count"),
                     (365,"haab","the 365-day solar year of the civil calendar"),
                     (13,"trecena","the 13-day numbered cycle"),
                     (20,"veintena","the 20 named day-signs")):
    add(f"mayan_{nm}", "Mayan", f"The field's position in {why}, counted in years since its founding.",
        np.cos(TAU*(AGE % mod)/mod))

# ── 6. Numerology ──
PYTH = {c:(i%9)+1 for i,c in enumerate("abcdefghijklmnopqrstuvwxyz")}
CHAL = {**{c:1 for c in "aijqy"},**{c:2 for c in "bkr"},**{c:3 for c in "cgls"},**{c:4 for c in "dmt"},
        **{c:5 for c in "ehnx"},**{c:6 for c in "uvw"},**{c:7 for c in "oz"},**{c:8 for c in "fp"}}
def droot(x):
    x=int(abs(x))
    while x>9: x=sum(int(c) for c in str(x))
    return x or 9
UF = sorted(set(FLD))
gp = {f: sum(PYTH.get(c,0) for c in f.lower()) for f in UF}
gc = {f: sum(CHAL.get(c,0) for c in f.lower()) for f in UF}
GP_ = np.array([gp[f] for f in FLD]); GC_ = np.array([gc[f] for f in FLD])
GPR = np.array([droot(v) for v in GP_]); YRR = np.array([droot(v) for v in YR])
add("gematria_pythagorean", "Numerology", "The field's name summed with the Pythagorean cipher "
    "(a=1 ... i=9, then repeating) — the 'expression number' of the name.", GP_)
add("gematria_chaldean", "Numerology", "The same name under the older Chaldean cipher, which assigns "
    "letters by sound rather than alphabetical order and never uses 9.", GC_)
add("name_root", "Numerology", "The field's name number reduced to a single digit, its root — the "
    "quantity numerology treats as the name's essential character.", GPR)
add("personal_year", "Numerology", "The classical personal year: name root plus the year's own root, "
    "reduced again. Numerology reads this as a 9-year cycle of themes running under a name.",
    np.array([droot(a+b_) for a,b_ in zip(GPR, YRR)]))
add("personal_year_cos", "Numerology", "The same 9-year personal-year cycle as a smooth cyclic "
    "coordinate, so year 9 and year 1 sit next to each other as they should.",
    np.cos(TAU*np.array([droot(a+b_) for a,b_ in zip(GPR,YRR)])/9))
add("age_mod_9", "Numerology", "Where the field sits in its own 9-year cycle counted from founding, "
    "the epicycle numerology places beneath the personal year.", np.cos(TAU*(AGE % 9)/9))
add("master_number_name", "Numerology", "Whether the name totals a master number (11, 22, 33) before "
    "reduction — held to mark unusual potency.", ((GP_ % 11 == 0) & (GP_ > 0)).astype(float))
add("name_root_equals_year_root", "Numerology", "Whether the name's root equals the year's root, a "
    "resonance numerology treats as an amplifying year for that name.", (GPR == YRR).astype(float))
add("founding_year_root", "Numerology", "The digit root of the field's founding year — its birth "
    "number, the numerological analogue of a natal placement.",
    np.array([droot(v) for v in B]))
add("name_length", "Numerology", "How many letters the field's name carries; some schools read "
    "length itself as significant before any cipher is applied.",
    np.array([len(f.replace(' ','')) for f in FLD], float))
add("vowel_count", "Numerology", "Vowels in the name — the 'soul urge' letters, which numerology "
    "separates from consonants as the inner rather than outer character.",
    np.array([sum(c in "aeiou" for c in f.lower()) for f in FLD], float))
add("CONTROL_year_root", "CONTROL", "The digit root of the calendar year alone, identical for every "
    "field that year — a control, and it should sit at 0.5.", YRR.astype(float))

# ── 7. profections, progressions, midpoints ──
add("annual_profection", "Hellenistic", "The annual profection: age modulo 12, the house the year "
    "'profects' to. Hellenistic practice advances one house per year of life and reads that house's "
    "ruler as the year's lord.", np.cos(TAU*(AGE % 12)/12))
for i,b in enumerate(BODY):
    prog = (NAT[:,i] + AGE*(TAU/PERIOD[i])/365.25) % TAU
    add(f"progressed_{b}", "Secondary progressions",
        f"Transiting {b} against its SECONDARY PROGRESSED place — the classical day-for-a-year "
        f"technique, where each day after birth stands for one year of life.",
        np.cos((CUR[:,i]-prog) % TAU))
PAIRS = [(i,k) for i in range(7) for k in range(i+1,7)]
for i,k in PAIRS:
    mt = (CUR[:,i]+CUR[:,k])/2; mn = (NAT[:,i]+NAT[:,k])/2
    add(f"midpoint_{BODY[i]}_{BODY[k]}", "Midpoints (Ebertin)",
        f"The transiting midpoint of {BODY[i]} and {BODY[k]} against the same midpoint in the natal "
        f"chart. Midpoint astrology treats the halfway degree between two planets as a sensitive "
        f"point that behaves like a body in its own right.", np.cos((mt-mn) % TAU))
add("age", "Chart mechanics", "The field's age in years — not a doctrine of any tradition, but the "
    "quantity every age-based cycle above is built from, listed so its own effect is visible.", AGE)

print(f"catalogue: {len(CAT)} features", flush=True)
rows = []
for nm, trad, expl, v in CAT:
    xf = v[FIT]
    if np.std(xf) < 1e-12: continue
    mu, sd = xf.mean(), xf.std()
    z = (v - mu)/sd
    # 2-parameter logistic: only the SIGN of b matters for ranking; fit it on train
    yb = Y[FIT]
    corr = np.corrcoef(z[FIT], yb)[0,1]
    sign = 1.0 if corr >= 0 else -1.0
    a_tr = wauc(Y[FIT], sign*z[FIT], YR[FIT])
    a_te = wauc(Y[TEST], sign*z[TEST], YR[TEST])
    rows.append(dict(feature=nm, tradition=trad, explanation=expl,
                     direction=("higher favours gaining slice" if sign > 0 else "lower favours gaining slice"),
                     train_auc=round(a_tr,4), test_auc=round(a_te,4)))
df = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
# noise floor: permute labels within year, refit direction, same pipeline
rng = np.random.RandomState(0); null = []
for _ in range(300):
    yp = Y.copy()
    for v in np.unique(YR):
        m = YR == v; yp[m] = rng.permutation(yp[m])
    v = CAT[rng.randint(len(CAT))][3]
    z = (v - v[FIT].mean())/(v[FIT].std()+1e-12)
    s = 1.0 if np.corrcoef(z[FIT], yp[FIT])[0,1] >= 0 else -1.0
    null.append(wauc(yp[TEST], s*z[TEST], YR[TEST]))
lo, hi = float(np.percentile(null,2.5)), float(np.percentile(null,97.5))
print(f"noise floor from 300 within-year label permutations: 95% band {lo:.4f}-{hi:.4f}", flush=True)
print(f"features above the band: {(df.test_auc > hi).sum()} of {len(df)} "
      f"(expected by chance ~{0.025*len(df):.0f})", flush=True)
df.to_csv(f"{BUN}/feature_atlas.csv", index=False)
json.dump({"n_features": len(df), "null_band": [round(lo,4), round(hi,4)],
           "above_band": int((df.test_auc > hi).sum()), "expected_by_chance": round(0.025*len(df),1),
           "top": df.head(20)[["feature","tradition","test_auc","train_auc"]].to_dict("records")},
          open(f"{BUN}/atlas_summary.json","w"), indent=1)
print("\n— top 20 by out-of-sample AUC:", flush=True)
for _, r in df.head(20).iterrows():
    print(f"  {r.test_auc:.4f}  (train {r.train_auc:.4f})  {r.feature:<38} {r.tradition}", flush=True)
print("\n— the controls (should sit at 0.5):", flush=True)
for _, r in df[df.tradition == "CONTROL"].iterrows():
    print(f"  {r.test_auc:.4f}  {r.feature}", flush=True)

# ── 8. which of these are AGE in disguise? ──────────────────────────────────────────────────────
# Within a single year, ranking fields by natal Pluto is ranking them by founding date, because
# Pluto takes 248 years to circle and the fields span ~170. Any such feature is the age effect
# wearing a planet's name. Measured, not assumed: within-year rank correlation against age.
from scipy.stats import spearmanr
def age_proxy(v):
    rs = []
    for y in np.unique(YR[TEST]):
        m = (YR == y) & TEST
        if m.sum() < 10 or np.std(v[m]) < 1e-12: continue
        rs.append(abs(spearmanr(v[m], AGE[m]).statistic))
    return float(np.nanmean(rs)) if rs else 0.0
prox = {nm: age_proxy(v) for nm, _, _, v in CAT}
df["age_rank_corr"] = df["feature"].map(prox).round(3)
df["is_age_proxy"] = df["age_rank_corr"] > 0.95
df.to_csv(f"{BUN}/feature_atlas.csv", index=False)
clean = df[(~df.is_age_proxy) & (df.tradition != "CONTROL")]
print(f"\n— age proxies: {int(df.is_age_proxy.sum())} of {len(df)} features rank fields within a year "
      f"almost exactly as age does (|rho| > 0.95)", flush=True)
print(f"  age itself scores {float(df[df.feature=='age'].test_auc.iloc[0]):.4f}", flush=True)
print(f"\n— top 15 features that are NOT age in disguise:", flush=True)
for _, r in clean.head(15).iterrows():
    print(f"  {r.test_auc:.4f} (train {r.train_auc:.4f}, age-corr {r.age_rank_corr:.2f})  "
          f"{r.feature:<36} {r.tradition}", flush=True)
print(f"\n  above the {hi:.4f} noise ceiling, excluding age proxies and controls: "
      f"{int((clean.test_auc > hi).sum())} of {len(clean)} (chance would give ~{0.025*len(clean):.0f})", flush=True)
json.dump({"n_features": len(df), "null_band": [round(lo,4), round(hi,4)],
           "age_proxies": int(df.is_age_proxy.sum()),
           "age_auc": float(df[df.feature=='age'].test_auc.iloc[0]),
           "non_proxy_above_band": int((clean.test_auc > hi).sum()),
           "non_proxy_expected": round(0.025*len(clean),1),
           "top_non_proxy": clean.head(20)[["feature","tradition","test_auc","train_auc","age_rank_corr"]].to_dict("records")},
          open(f"{BUN}/atlas_summary.json","w"), indent=1)
