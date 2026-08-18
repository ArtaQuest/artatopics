#!/usr/bin/env python3
"""THE GRAND ATLAS — thousands of single-number astrological and numerological features, each with
its own two-parameter logistic regression, ranked by out-of-sample AUC.

For every feature x: logit P(field gains slice next year) = a + b*x, fitted on the 30 years before
the 1997 wall. Two parameters, so nothing hides in an interaction; and since the fit is monotone in
x, the score is the feature's own separating power with its DIRECTION (the sign of b) fixed on
train. A feature whose direction flips after the wall scores below 0.5 — the test is genuinely out
of sample.

Scored per topic, as specified: one AUC per field over that field's own benchmark years, averaged
across fields. Pooled within-year AUC is reported beside it, because the two metrics answer
different questions and disagree.

Generators, all combinatorial over the classical vocabulary:
  transit grid     7 transiting x 7 natal bodies x harmonics 1-12 x {cos,sin}   tropical + sidereal
  aspects          every classical and minor angle, orb-scored, over all 49 body pairs
  antiscia         solstice-mirror points, the "shadow degrees" of traditional practice
  midpoints        transiting pair -> natal body, natal pair -> transiting body, and pair -> pair
  harmonic charts  the nth-harmonic chart position of each body, n = 1..12
  returns          completed returns, cycle phase, and applying/separating for each body
  progressions     secondary progressions of every body against every natal body
  dashas           Vimshottari maha x antar lord pairs, the 120-year Vedic sequence
  chinese          stem, branch, element, trine group, clash, and full sexagenary phase
  mayan            tzolkin daysign and trecena, haab, and the long-count cycles
  numerology       4 ciphers x 5 name-parts x reductions, and their classical year interactions

  python3 analysis/arxivtopics/competition/grand_atlas.py
"""
import os, sys, json, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import arxiv_fit as af

BUN = os.path.expanduser("~/.artaquest-dev/artacomp/piecomp")
tr = pd.read_csv(f"{BUN}/train.csv"); te = pd.read_csv(f"{BUN}/test.csv"); sol = pd.read_csv(f"{BUN}/solution.csv")
yte = sol.set_index("id").loc[te["id"]]["target"].to_numpy()
WALL_Y, WINDOW = 1997, 30
ALL = pd.concat([tr[["field","year","target"]], te[["field","year"]].assign(target=yte)], ignore_index=True)
Y = ALL["target"].to_numpy().astype(float); YR = ALL["year"].to_numpy(); FLD = ALL["field"].to_numpy()
names, Yv, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
years = [int(y) for y in labels]; Y0 = years[0]
BODY = ["Mars","Jupiter","Saturn","Uranus","Neptune","Pluto","Node"]
PER = np.array([1.88, 11.86, 29.46, 84.0, 164.8, 248.0, 18.6])
birth = tr.groupby("field")["year"].min().to_dict()
for f in te["field"].unique(): birth.setdefault(f, WALL_Y)
B = np.array([birth[f] for f in FLD]); AGE = (YR - B).astype(float)
NAT = TH[B - Y0]; CUR = TH[YR - Y0]
AYAN = np.deg2rad(23.85); TAU = 2*np.pi
NATS = (NAT - AYAN) % TAU; CURS = (CUR - AYAN) % TAU
FIT = (YR < WALL_Y) & (YR >= WALL_Y - WINDOW); TEST = YR >= WALL_Y
PAIRS = [(i,k) for i in range(7) for k in range(i+1,7)]

# ── per-topic AUC over any row subset: pad each field's rows into a matrix, rank within row ──
def make_scorer(mask):
    f_ = FLD[mask]; y_ = Y[mask]
    ro = {f: np.where(f_ == f)[0] for f in np.unique(f_)}
    kp = [f for f in ro if len(ro[f]) >= 6 and 0 < y_[ro[f]].sum() < len(ro[f])]
    if not kp: return None
    m_ = max(len(ro[f]) for f in kp)
    I = np.full((len(kp), m_), -1, int); K = np.zeros((len(kp), m_), bool)
    for r, f in enumerate(kp):
        ix = ro[f]; I[r, :len(ix)] = ix; K[r, :len(ix)] = True
    L = np.where(K, y_[np.clip(I,0,None)], 0.0)
    npos = L.sum(1); nneg = K.sum(1) - npos
    def score(v_sub, lab=None):
        LL = L if lab is None else np.where(K, lab[np.clip(I,0,None)], 0.0)
        np_ = LL.sum(1); nn_ = K.sum(1) - np_
        good = (np_ > 0) & (nn_ > 0)
        V = np.where(K, v_sub[np.clip(I,0,None)], -np.inf)
        R = np.where(K, rankdata(np.where(K, V, -np.inf), axis=1), 0.0)
        a = ((R*LL).sum(1) - np_*(np_+1)/2) / np.maximum(np_*nn_, 1e-9)
        return float(np.mean(a[good])), int(good.sum())
    return score
# original test-window machinery (kept: the noise-floor block below uses IDX/MSK directly)
tf = FLD[TEST]; ty_ = Y[TEST]
uf = np.unique(tf); rowsof = {f: np.where(tf == f)[0] for f in uf}
keep = [f for f in uf if len(rowsof[f]) >= 6 and 0 < ty_[rowsof[f]].sum() < len(rowsof[f])]
M = max(len(rowsof[f]) for f in keep)
IDX = np.full((len(keep), M), -1, int); MSK = np.zeros((len(keep), M), bool)
for r, f in enumerate(keep):
    ix = rowsof[f]; IDX[r, :len(ix)] = ix; MSK[r, :len(ix)] = True
LAB = np.where(MSK, ty_[np.clip(IDX,0,None)], 0.0)
NPOS = LAB.sum(1); NNEG = MSK.sum(1) - NPOS
def per_topic_auc(v_test):
    V = np.where(MSK, v_test[np.clip(IDX,0,None)], -np.inf)
    R = rankdata(np.where(MSK, V, -np.inf), axis=1)
    R = np.where(MSK, R, 0.0)
    S = (R*LAB).sum(1)
    return float(np.mean((S - NPOS*(NPOS+1)/2) / (NPOS*NNEG)))
# the same machinery on the TRAIN window, so features can be ranked by train and judged on test
trf = FLD[FIT]; tr_y = Y[FIT]
rowsof_tr = {f: np.where(trf == f)[0] for f in np.unique(trf)}
keep_tr = [f for f in rowsof_tr if len(rowsof_tr[f]) >= 6 and 0 < tr_y[rowsof_tr[f]].sum() < len(rowsof_tr[f])]
Mt = max(len(rowsof_tr[f]) for f in keep_tr)
IDXt = np.full((len(keep_tr), Mt), -1, int); MSKt = np.zeros((len(keep_tr), Mt), bool)
for r, f in enumerate(keep_tr):
    ix = rowsof_tr[f]; IDXt[r, :len(ix)] = ix; MSKt[r, :len(ix)] = True
LABt = np.where(MSKt, tr_y[np.clip(IDXt,0,None)], 0.0)
NPOSt = LABt.sum(1); NNEGt = MSKt.sum(1) - NPOSt
def per_topic_auc_train(v_fit):
    V = np.where(MSKt, v_fit[np.clip(IDXt,0,None)], -np.inf)
    R = np.where(MSKt, rankdata(np.where(MSKt, V, -np.inf), axis=1), 0.0)
    S = (R*LABt).sum(1)
    return float(np.mean((S - NPOSt*(NPOSt+1)/2) / (NPOSt*NNEGt)))

SC_TEST = make_scorer(TEST); SC_TRAIN = make_scorer(FIT)
USAGE = sol.set_index("id").loc[te["id"]]["Usage"].to_numpy()
UALL = np.concatenate([np.array(["Train"]*len(tr)), USAGE])
PUB = TEST & (UALL == "Public"); PRI = TEST & (UALL == "Private")
SC_PUB = make_scorer(PUB); SC_PRI = make_scorer(PRI)
def decalendar(v):
    """Strip the calendar: subtract each year's cross-field mean, leaving only the part of the
    feature that distinguishes FIELDS within that year. A purely global transit becomes zero."""
    out = v.astype(float).copy()
    for y in np.unique(YR):
        m = YR == y
        out[m] -= out[m].mean()
    return out

yrs_t = YR[TEST]; uy = np.unique(yrs_t)
ymask = {y: (yrs_t == y) for y in uy if 0 < Y[TEST][yrs_t==y].sum() < (yrs_t==y).sum()}
def within_year_auc(v_test):
    return float(np.mean([roc_auc_score(Y[TEST][m], v_test[m]) for m in ymask.values()]))

# ── generators ──────────────────────────────────────────────────────────────────────────────────
ORD = {c: i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
PYTH = {c: (i % 9)+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
CHAL = {**{c:1 for c in "aijqy"},**{c:2 for c in "bkr"},**{c:3 for c in "cgls"},**{c:4 for c in "dmt"},
        **{c:5 for c in "ehnx"},**{c:6 for c in "uvw"},**{c:7 for c in "oz"},**{c:8 for c in "fp"}}
CIPHER = {"Pythagorean": PYTH, "Chaldean": CHAL, "English ordinal": ORD,
          "reverse ordinal": {c: 27-ORD[c] for c in ORD}}
CIPHER_WHY = {"Pythagorean":"a=1..i=9 repeating, the modern standard cipher",
    "Chaldean":"the older cipher that assigns letters by SOUND and never uses 9",
    "English ordinal":"plain alphabetical position, the cipher of English gematria",
    "reverse ordinal":"alphabetical position counted backwards, a common gematria variant"}
def parts(f):
    lo = f.lower(); w = lo.split()
    return {"whole name": lo, "first word": w[0], "last word": w[-1],
            "vowels only": "".join(c for c in lo if c in "aeiou"),
            "consonants only": "".join(c for c in lo if c.isalpha() and c not in "aeiou")}
PART_WHY = {"whole name":"the full name, the expression number",
    "first word":"the first word alone", "last word":"the final word alone",
    "vowels only":"the vowels, which numerology calls the soul urge — the inner character",
    "consonants only":"the consonants, the outer or personality number"}
def droot(x):
    x = int(abs(x))
    while x > 9: x = sum(int(c) for c in str(x))
    return x or 9
UF = sorted(set(FLD))
PARTVAL = {(cn, pn): {f: sum(cm.get(c,0) for c in parts(f)[pn]) for f in UF}
           for cn, cm in CIPHER.items() for pn in PART_WHY}

def gen():
    # 1. transit grid, tropical and sidereal, harmonics 1-12
    for zn, ZC, ZN, why in (("tropical", CUR, NAT, "the tropical zodiac, tied to the equinox"),
                            ("sidereal", CURS, NATS, "the sidereal zodiac of Vedic practice, tied to the fixed stars")):
        for i, bi in enumerate(BODY):
            for k, bk in enumerate(BODY):
                d = (ZC[:,i] - ZN[:,k]) % TAU
                for h in range(1, 13):
                    for fn, fname in ((np.cos, "cos"), (np.sin, "sin")):
                        if h > 6 and fname == "sin": continue
                        yield (f"{zn}_H{h}_{fn.__name__}_transit_{bi}_natal_{bk}",
                               f"Transit grid ({zn})",
                               f"Transiting {bi} against the degree natal {bk} held at the field's founding, "
                               f"read at harmonic {h} ({fname} phase), in {why}. Harmonic {h} divides the "
                               f"circle into {h} parts; h=1 is the conjunction axis, h=2 the opposition, "
                               f"h=3 the trine, h=4 the square, h=6 the sextile.",
                               fn(h*d))
    # 2. aspects, orb-scored, every classical and minor angle
    ANG = [("conjunction",0,10),("semisextile",30,3),("decile",36,2),("semisquare",45,3),
           ("septile",51.43,2),("sextile",60,6),("quintile",72,3),("square",90,8),
           ("trine",120,8),("sesquiquadrate",135,3),("biquintile",144,3),("quincunx",150,3),
           ("opposition",180,10)]
    for i, bi in enumerate(BODY):
        for k, bk in enumerate(BODY):
            d = (CUR[:,i] - NAT[:,k]) % TAU
            for an, deg, orb in ANG:
                sep = np.abs(((d - np.deg2rad(deg) + np.pi) % TAU) - np.pi)
                yield (f"aspect_{an}_{bi}_to_natal_{bk}", "Aspects",
                       f"Transiting {bi} within {orb}° of a {an} ({deg}°) to natal {bk}. Scored 1 at exact "
                       f"and fading to 0 at the edge of orb — the orb convention of traditional practice, "
                       f"wide for the major angles and tight for the minor ones.",
                       np.clip(1 - sep/np.deg2rad(orb), 0, 1))
    # 3. antiscia — the solstice mirror
    for i, bi in enumerate(BODY):
        for k, bk in enumerate(BODY):
            anti = (np.pi - CUR[:,i]) % TAU
            yield (f"antiscia_{bi}_to_natal_{bk}", "Antiscia",
                   f"The antiscion of transiting {bi} — its mirror across the solstice axis, where two "
                   f"degrees share the same daylight length — against natal {bk}. Traditional astrology "
                   f"treats antiscia as hidden points of contact invisible to ordinary aspects.",
                   np.cos((anti - NAT[:,k]) % TAU))
    # 4. midpoints
    for (i,k) in PAIRS:
        mt = (CUR[:,i]+CUR[:,k])/2; mn = (NAT[:,i]+NAT[:,k])/2
        for q, bq in enumerate(BODY):
            yield (f"midpoint_T{BODY[i]}_{BODY[k]}_to_natal_{bq}", "Midpoints (Ebertin)",
                   f"The transiting midpoint of {BODY[i]} and {BODY[k]} — the halfway degree between them — "
                   f"against natal {bq}. Cosmobiology treats a midpoint as a sensitive point behaving like "
                   f"a planet in its own right, and a body arriving there as the trigger.",
                   np.cos((mt - NAT[:,q]) % TAU))
            yield (f"midpoint_N{BODY[i]}_{BODY[k]}_to_transit_{bq}", "Midpoints (Ebertin)",
                   f"The NATAL midpoint of {BODY[i]} and {BODY[k]}, the fixed sensitive point in the birth "
                   f"chart, contacted by transiting {bq}.",
                   np.cos((CUR[:,q] - mn) % TAU))
        for (a2,b2) in PAIRS:
            mn2 = (NAT[:,a2]+NAT[:,b2])/2
            yield (f"midpoint_T{BODY[i]}_{BODY[k]}_to_N{BODY[a2]}_{BODY[b2]}", "Midpoints (Ebertin)",
                   f"Transiting midpoint of {BODY[i]}/{BODY[k]} against the natal midpoint of "
                   f"{BODY[a2]}/{BODY[b2]} — midpoint contacting midpoint, the deepest layer of the "
                   f"cosmobiological method.", np.cos((mt - mn2) % TAU))
    # 5. harmonic charts, returns, progressions
    for i, b in enumerate(BODY):
        for h in range(1, 13):
            yield (f"harmonic_chart_{h}_natal_{b}", "Harmonic charts",
                   f"Natal {b} multiplied into the {h}th harmonic chart — the whole chart rescaled by {h}, "
                   f"a technique that exposes structures invisible in the radix.", np.cos(h*NAT[:,i]))
        yield (f"return_count_{b}", "Returns",
               f"Completed {b} returns since founding, one every {PER[i]:.1f} years — the field's age "
               f"measured on {b}'s clock.", np.floor(AGE/PER[i]))
        yield (f"return_phase_{b}", "Returns",
               f"Fraction of the current {b} cycle elapsed, 0 at the return and approaching 1 before the "
               f"next.", (AGE/PER[i]) % 1.0)
        yield (f"applying_{b}", "Returns",
               f"Whether transiting {b} is APPLYING to its natal degree (closing) rather than separating — "
               f"traditional practice reads an applying contact as the potent half.",
               (np.sin((CUR[:,i]-NAT[:,i]) % TAU) < 0).astype(float))
        prog = (NAT[:,i] + AGE*(TAU/PER[i])/365.25) % TAU
        for k, bk in enumerate(BODY):
            yield (f"progressed_{b}_to_natal_{bk}", "Secondary progressions",
                   f"Secondary-progressed {b} — advanced by the day-for-a-year rule — against natal {bk}.",
                   np.cos((prog - NAT[:,k]) % TAU))
    # 6. Vimshottari dasha lord pairs
    DASHA = [("Ketu",7),("Venus",20),("Sun",6),("Moon",10),("Mars",7),("Rahu",18),("Jupiter",16),("Saturn",19),("Mercury",17)]
    def dsh(nn, age):
        pos = (nn % TAU)/TAU*27.0; k = int(pos) % 27; frac = pos-int(pos)
        idx = k % 9; rem = (1-frac)*DASHA[idx][1]; t = age
        if t < rem: return idx, int(((1-(rem-t)/DASHA[idx][1])*9)) % 9
        t -= rem; idx = (idx+1) % 9
        while t >= DASHA[idx][1]: t -= DASHA[idx][1]; idx = (idx+1) % 9
        return idx, int((t/DASHA[idx][1])*9) % 9
    DD = np.array([dsh(NATS[r,6], AGE[r]) for r in range(len(AGE))])
    for a_ in range(9):
        yield (f"dasha_maha_{DASHA[a_][0]}", "Vimshottari dasha",
               f"Running the {DASHA[a_][0]} mahadasha, a {DASHA[a_][1]}-year chapter of the 120-year "
               f"Vimshottari cycle, seeded from the natal node's nakshatra.", (DD[:,0]==a_).astype(float))
        for b_ in range(9):
            yield (f"dasha_{DASHA[a_][0]}_in_{DASHA[b_][0]}", "Vimshottari dasha",
                   f"The {DASHA[b_][0]} antardasha inside the {DASHA[a_][0]} mahadasha — the sub-period "
                   f"that Vedic practice reads for the specific flavour of a chapter.",
                   ((DD[:,0]==a_)&(DD[:,1]==b_)).astype(float))
    # 7. Chinese
    ANIMALS = ["Rat","Ox","Tiger","Rabbit","Dragon","Snake","Horse","Goat","Monkey","Rooster","Dog","Pig"]
    ELEM = ["Yang Wood","Yin Wood","Yang Fire","Yin Fire","Yang Earth","Yin Earth","Yang Metal","Yin Metal","Yang Water","Yin Water"]
    bb, cb = (B-4) % 12, (YR-4) % 12; bs, cs = (B-4) % 10, (YR-4) % 10
    for a_ in range(12):
        yield (f"chinese_natal_animal_{ANIMALS[a_]}", "Chinese",
               f"The field was founded in a year of the {ANIMALS[a_]} — its animal sign.", (bb==a_).astype(float))
        yield (f"chinese_current_animal_{ANIMALS[a_]}", "Chinese",
               f"The current year is a year of the {ANIMALS[a_]}.", (cb==a_).astype(float))
    for a_ in range(10):
        yield (f"chinese_natal_stem_{ELEM[a_].replace(' ','_')}", "Chinese",
               f"The field's founding heavenly stem is {ELEM[a_]} — the elemental half of the sexagenary year.",
               (bs==a_).astype(float))
    yield ("chinese_trine_harmony", "Chinese", "The current animal shares the field's four-animal harmony "
           "group (branches four apart), the most auspicious relation in the cycle.", ((cb-bb)%4==0).astype(float))
    yield ("chinese_clash", "Chinese", "The current animal directly opposes the founding animal, six branches "
           "away — the classical clash year of upheaval.", (((cb-bb)%12)==6).astype(float))
    for h in (1,2,3):
        yield (f"sexagenary_phase_H{h}", "Chinese", f"Position in the 60-year stem-and-branch cycle since "
               f"founding, harmonic {h}. A full sexagenary return is the Chinese measure of a completed life.",
               np.cos(h*TAU*(AGE % 60)/60))
    # 8. Mayan
    for m_, nm_, why in ((260,"tzolkin","the 260-day sacred round, the divinatory core of the calendar"),
                         (365,"haab","the 365-day civil year"),(13,"trecena","the 13-day numbered cycle"),
                         (20,"veintena","the 20 named day-signs"),(52,"calendar round","the 52-year round where tzolkin and haab realign")):
        for h in (1,2,3):
            yield (f"mayan_{nm_}_H{h}", "Mayan", f"The field's position in {why}, counted in years from "
                   f"founding, harmonic {h}.", np.cos(h*TAU*(AGE % m_)/m_))
    # 9. numerology
    YRR = np.array([droot(v) for v in YR]); BRR = np.array([droot(v) for v in B])
    for (cn, pn), vals in PARTVAL.items():
        g = np.array([vals[f] for f in FLD], float)
        gr = np.array([droot(v) for v in g])
        yield (f"num_{cn}_{pn}_sum".replace(" ","_"), "Numerology",
               f"The {pn} of the field's name summed under the {cn} cipher ({CIPHER_WHY[cn]}); "
               f"{PART_WHY[pn]}.", g)
        yield (f"num_{cn}_{pn}_root".replace(" ","_"), "Numerology",
               f"The same total reduced to a single digit — the root numerology treats as its essence.", gr)
        yield (f"num_{cn}_{pn}_personal_year".replace(" ","_"), "Numerology",
               f"The classical PERSONAL YEAR: the {pn} root under {cn} added to the year's own root and "
               f"reduced, giving a 9-year cycle of themes running beneath the name.",
               np.array([droot(a+b_) for a,b_ in zip(gr, YRR)], float))
        yield (f"num_{cn}_{pn}_essence".replace(" ","_"), "Numerology",
               f"The 'essence' variant: the {pn} root under {cn} combined with the field's FOUNDING year "
               f"root rather than the current one — a fixed rather than moving number.",
               np.array([droot(a+b_) for a,b_ in zip(gr, BRR)], float))
        for m_ in (7, 9, 11, 12, 22):
            yield (f"num_{cn}_{pn}_mod{m_}".replace(" ","_"), "Numerology",
                   f"The {pn} total under {cn} modulo {m_}" +
                   (" — 11 and 22 are the master numbers, never reduced in classical practice."
                    if m_ in (11,22) else f" — the {m_}-fold cycle."), g % m_)
        yield (f"num_{cn}_{pn}_x_age".replace(" ","_"), "Numerology",
               f"The {pn} root under {cn} advanced by the field's age, reduced — the name's number walking "
               f"forward one step per year of life.",
               np.array([droot(a+int(c)) for a,c in zip(gr, AGE)], float))
    yield ("num_age_mod_9", "Numerology", "Where the field sits in its own 9-year cycle counted from "
           "founding — the epicycle beneath the personal year.", AGE % 9)
    yield ("num_founding_root", "Numerology", "Digit root of the founding year, the field's birth number.", BRR.astype(float))
    yield ("CONTROL_year_root", "CONTROL", "Digit root of the calendar year alone — identical for every "
           "field that year. A control on the calendar's own contribution.", YRR.astype(float))
    for i,b in enumerate(BODY):
        yield (f"CONTROL_transiting_{b}", "CONTROL", f"Where {b} stands in year t, ignoring the field "
               f"entirely — identical for every field that year. This is what a transit feature can earn "
               f"knowing nothing about the chart.", np.cos(CUR[:,i]))
    yield ("CONTROL_age", "CONTROL", "The field's age in years — no tradition claims it, but every "
           "age-based cycle is built from it, so its own effect must be visible.", AGE)

print("generating and scoring …", flush=True)
rows = []; n = 0
yf = Y[FIT]
for nm, trad, expl, v in gen():
    n += 1
    v = np.nan_to_num(np.asarray(v, float))
    sd = v[FIT].std()
    if sd < 1e-12: continue
    z = (v - v[FIT].mean())/sd
    c = np.corrcoef(z[FIT], yf)[0,1]
    if not np.isfinite(c): c = 0.0
    sgn = 1.0 if c >= 0 else -1.0
    zt = sgn*z
    ct = np.corrcoef(zt[TEST], Y[TEST])[0,1]
    zm = decalendar(zt)
    rows.append(dict(family=trad, name=nm,
                     train=round(SC_TRAIN(zt[FIT])[0],4),
                     held=round(SC_TEST(zt[TEST])[0],4),
                     matched=round(SC_TEST(zm[TEST])[0],4),
                     public=round(SC_PUB(zt[PUB])[0],4) if SC_PUB else np.nan,
                     private=round(SC_PRI(zt[PRI])[0],4) if SC_PRI else np.nan,
                     within_year=round(within_year_auc(zt[TEST]),4),
                     flipped=bool(np.isfinite(ct) and ct < 0),
                     direction=("higher favours gaining slice" if sgn>0 else "lower favours gaining slice"),
                     explanation=expl))
    if len(rows) % 500 == 0: print(f"  {len(rows)} scored …", flush=True)
df = pd.DataFrame(rows).sort_values("train", ascending=False)
print(f"generated {n} features, scored {len(df)}", flush=True)
# ── NULLMAX: the multiplicity-corrected ceiling. With thousands of features the right question is
# not "does this feature beat a single feature's noise band" but "does the BEST of them beat the
# best that pure noise produces over the same number of tries". Labels are permuted inside each
# field, every feature is rescored, and the MAXIMUM is recorded — repeated to get its spread.
print("\ncomputing nullmax over all features (this is the honest ceiling) …", flush=True)
VEC = []
for nm, trad, expl, v in gen():
    v = np.nan_to_num(np.asarray(v, float))
    if v[FIT].std() < 1e-12: continue
    VEC.append(((v - v[FIT].mean())/v[FIT].std()).astype(np.float32))
VEC = np.asarray(VEC)
rng = np.random.RandomState(0)
nmax_tr, nmax_he, singles = [], [], []
for rep in range(8):
    yp = Y.copy()
    for f in np.unique(FLD):
        m = FLD == f; yp[m] = rng.permutation(yp[m])
    bt, bh = -1.0, -1.0
    for r in range(VEC.shape[0]):
        z = VEC[r].astype(float)
        c = np.corrcoef(z[FIT], yp[FIT])[0,1]
        sg = 1.0 if (np.isfinite(c) and c >= 0) else -1.0
        at = SC_TRAIN((sg*z)[FIT], lab=yp[FIT])[0]
        ah = SC_TEST((sg*z)[TEST], lab=yp[TEST])[0]
        bt = max(bt, at); bh = max(bh, ah)
        if r % 97 == 0: singles.append(ah)
    nmax_tr.append(bt); nmax_he.append(bh)
    print(f"  permutation {rep+1}/8: best-of-{VEC.shape[0]} train {bt:.4f} · held {bh:.4f}", flush=True)
NMT, NMH = float(np.mean(nmax_tr)), float(np.mean(nmax_he))
se_single = float(np.std(singles))
cal = float(df[df.family=="CONTROL"].held.max())
df["beats_nullmax"] = df.held > NMH
df.to_csv(f"{BUN}/grand_atlas.csv", index=False)
print(f"\n  NULLMAX (best of {VEC.shape[0]} noise features): train {NMT:.4f} · held {NMH:.4f}", flush=True)
print(f"  single-feature noise SE: {se_single:.4f}", flush=True)
print(f"  best CONTROL (knows nothing of the chart): {cal:.4f}", flush=True)
print(f"  features beating NULLMAX on held: {int(df.beats_nullmax.sum())} of {len(df)}", flush=True)
print(f"  direction flipped across the wall: {int(df.flipped.sum())} of {len(df)}", flush=True)
print("\n— TOP 20 BY TRAIN, and what each then did:", flush=True)
print(f"  {'train':>7}{'held':>8}{'matched':>9}{'public':>8}{'private':>9}  feature", flush=True)
for _, r in df.head(20).iterrows():
    print(f"  {r.train:>7.4f}{r.held:>8.4f}{r.matched:>9.4f}{r.public:>8.4f}{r.private:>9.4f}  {r['name'][:46]}", flush=True)
for N in (1,10,50,100,500,1000):
    h = df.head(N)
    print(f"  top {N:>4} by train: train {h.train.mean():.4f} -> held {h.held.mean():.4f} "
          f"· matched {h.matched.mean():.4f} · {int((h.held>NMH).sum())} beat nullmax", flush=True)
print(f"\n  corr(train, held) over {len(df)} features: {np.corrcoef(df.train, df.held)[0,1]:+.4f}", flush=True)
print(f"  corr(train, matched): {np.corrcoef(df.train, df.matched)[0,1]:+.4f}", flush=True)
g = df.groupby("family").agg(n=("held","size"), best_train=("train","max"), best_held=("held","max"),
                             best_matched=("matched","max"), median_held=("held","median"))
print("\n— by tradition:", flush=True); print(g.sort_values("best_train", ascending=False).to_string(), flush=True)
json.dump({"n_features": int(len(df)), "nullmax_train": round(NMT,4), "nullmax_held": round(NMH,4),
           "se_single": round(se_single,4), "control_bar": round(cal,4),
           "beats_nullmax": int(df.beats_nullmax.sum()), "n_flipped": int(df.flipped.sum()),
           "corr_train_held": round(float(np.corrcoef(df.train, df.held)[0,1]),4),
           "n_train_rows": int(FIT.sum()), "n_held_rows": int(TEST.sum()),
           "top_by_train": df.head(30)[["name","family","train","held","matched"]].to_dict("records")},
          open(f"{BUN}/grand_atlas_meta.json","w"), indent=1)
