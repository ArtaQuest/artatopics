#!/usr/bin/env python3
"""Export the research atlas as one JSON the SPA bundles (src/data/research.json).
MONTHLY model (11 bodies, fixed SINC kernel; per-keyword resonance PERIOD tuned by max F1(rep,R2); EXACTLY 11 params
= one non-negative weight per body, NO bias, phase swept on a 5° grid); series downsampled for compact line charts.
Run after the monthly fit + the recency crop sweep (recency_experiment.py). NOTE: the registry still tags each
record res="weekly" for historical reasons — that is just the registry's selection tag, the data is monthly.
  python3 analysis/export_research.py
"""
import importlib.util as _u, os, json
import numpy as np, pandas as pd
_spec = _u.spec_from_file_location("tf", os.path.join(os.path.dirname(__file__) or ".", "trends_fit.py"))
tf = _u.module_from_spec(_spec); _spec.loader.exec_module(tf)
# (the weekly patch was removed 2026-06-30 — the model is MONTHLY now; tf's own defaults
#  GRID=MS / DROP_LAST=12 / DATA_DIR=data_monthly are correct, and load_y needs them to find the series)

OUT = "artaquest-web/src/data/research.json"
try:
    _MODEL_PRED = json.load(open("analysis/adstopics/astro_ts_pred.json"))
except Exception:
    _MODEL_PRED = {}
def ds(a, n=160):
    a = np.asarray(a, float)
    if len(a) <= n: return a
    return a[np.linspace(0, len(a) - 1, n).round().astype(int)]

def main():
    try:
        import importlib.util as _u
        _s = _u.spec_from_file_location("sw", "analysis/_stopwords.py"); _m = _u.module_from_spec(_s); _s.loader.exec_module(_m); is_junk = _m.is_junk
    except Exception:
        def is_junk(w): return False
    # Topic-centric registry: the fields the 567 topics selected by COMMONNESS (collect_daily.py). Each daily fit
    # carries its pos (noun→WHAT / adj→HOW), commonness, and source topics.
    _reg = os.environ.get("AQ_REG", "analysis/_fields_weekly.json")
    reg = json.load(open(_reg)) if os.path.exists(_reg) else {}
    # PUBLISH EVERY analysed field. The house_ratio (max/2nd-max area) > 1.5 rule only gates whether a field may be
    # the house REPRESENTATIVE (title) — carried per-field as houseRatio; it does NOT hide a field from the atlas.
    weekly = [r for r in reg.values() if r.get("res") == "weekly"]   # NO selection — publish every analysed ISCO occupation
    for r in weekly:                                              # recompute rep with the CURRENT formula (so a rep-formula
        if r.get("class_model") in ("best-fit", "topic500-winner", "topic500-phase", "sinc-gd", "gauss-gd", "astroattention", "astroattention-anchor"):  # reclassified records keep their stored rep — the sidereal
            continue                                              # area-share is meaningless for the season sign
        try: r["rep"] = tf.rep_score(r)                           # change applies without re-collecting every field)
        except Exception: pass
    pub = sorted(weekly, key=lambda r: -r.get("rep", 0.0))
    # POPULARITY FILTER (operator 2026-07-21): publish only topics with >=10% mean worldwide interest
    # (matches the study loader — below-10% topics are excluded from the science and the site).
    pub = [r for r in pub if float(r.get("popularity", 0)) >= 10.0]
    topics = []
    daily = {}   # full-resolution MONTHLY raw series per field — lazy-loaded + zoomable on the field page
    for r in pub:
        t, y = tf.load_y(r["label"])
        sh = sorted(r.get("shares", {}).items(), key=lambda x: -x[1])
        rec = {
            "key": r["key"], "label": r["label"], "sign": r["sign"], "ref": round(r["ref"]),
            "axis": r.get("axis", "house"),   # 'house'=Professions (careers) · 'topic'=Skills — drives the two separate pages
            "system": r.get("system", ""),    # source typology-system slug (for Skills→systems listing)
            "r2": round(r["r2"], 4), "rep": round(r["rep"], 4), "stitch": round(r.get("stitch_corr", 0), 3),
            "pos": r.get("pos", "noun"), "freq": int(r.get("freq", 0)), "topics": list(r.get("topics", [])),  # WHAT/HOW + commonness + source topics
            "popularity": round(float(r.get("popularity", 0)), 1),  # GLOBAL score: mean interest as % of the 'quest' baseline (cross-field comparable)
            "thread": int(r["thread"]) if r.get("thread") else None,
            "tuning": [round(v, 4) for v in r["tuning"]],   # fit quality vs phase angle (sinc kernel, 5° phase grid)
            # SOFT house assignment: a 12-vector of soft-scores (∫ fit-quality under each sign band, normalised, sums to 1)
            # + the DECISIVENESS of the primary house over the runner-up (a rep of the house only when > 1.5).
            "signScores": list(r.get("sign_scores", [])),
            "houseRatio": round(float(r.get("house_ratio", 0.0)), 3),
            # DOMINANT rhythm (years) — 1 (annual season) or null (a secular trend, no clean period).
            "period": (round(float(r["period"]), 3) if r.get("period") is not None else None), "f1": round(float(r.get("f1", 0.0)), 4),
            "periodProm": (round(float(r["period_prom"]), 4) if r.get("period_prom") is not None else None),
            "periodCurve": [{"period": c.get("period"), "r2": round(float(c.get("r2", 0)), 4),
                             "rep": (round(float(c["rep"]), 2) if c.get("rep") is not None else None),
                             "f1": round(float(c.get("f1", 0)), 4)}
                            for c in r.get("period_curve", [])],
            "shares": [{"slug": tf.h_slug(*tf._parse(hk)), "label": tf.h_label(*tf._parse(hk)),
                        "pct": round(v * 100, 1)} for hk, v in sh],
            "series": None,
        }
        # ── topic-500 sinc-GD read-out: the topic's SEASON (the fitted global phase p → its zodiac sign +
        # the peak MONTH), the per-body FREQUENCIES f_i, the model's 12-month FORECAST beyond the clean
        # series, and the share of explained variance carried by the annual season. sidSign/classModel ride along.
        if r.get("season"):
            s = r["season"]
            rec["season"] = {"sign": s.get("sign"), "peak": round(float(s.get("peak", 0))),
                             "month": (int(s["month"]) if s.get("month") is not None else None),   # 0-11, consistent with the sign
                             "r2": round(float(s.get("r2", 0)), 4)}
        # per-body FREQUENCIES f_i (inverse widths of each body's sinc term) — a small bars/table on the page.
        fr = r.get("freqs") or {}
        if fr:
            rec["frequencies"] = [{"body": tf.nice(tf._parse(hk)[0]), "freq": round(float(v), 4)} for hk, v in fr.items()]
        rec["rejectH0"] = bool(r.get("reject_h0", False))       # season-led (a real annual season) vs trend-led (top-level)
        if r.get("forecast") is not None:
            rec["forecast"] = [round(float(v), 1) for v in r["forecast"]]
        if r.get("annual_frac") is not None:
            rec["annualFrac"] = round(float(r["annual_frac"]), 4)
        rec["trend"] = bool(r.get("trend", False))          # trend-led (vs season-led) — drives the page's two-mode hero
        rec["trendDir"] = r.get("trend_dir", "steady")      # rising / falling / steady
        if r.get("sid_sign"):
            rec["sidSign"] = r["sid_sign"]
        rec["classModel"] = r.get("class_model", "")
        # PHASE-ROTARY enc/dec read-out (operator 2026-07-19): every topic's estimated seasonal
        # phase, its peak-growth month, the dominant sidereal body + its Sun weight-share.
        if r.get("attn_phase") is not None:
            rec["phaseEst"] = round(float(r["attn_phase"]), 1)
            rec["peakMonth"] = r.get("peak_month", "")
            rec["dominantPlanet"] = r.get("dominant_planet", "")
            if r.get("sign"): rec["phaseSign"] = r["sign"]      # the universal phase's astrological sign (phi // 30)
            if r.get("sun_share") is not None: rec["sunShare"] = round(float(r["sun_share"]), 3)
            if r.get("phase_stable") is not None: rec["phaseStable"] = bool(r["phase_stable"])
        if r.get("dir_acc_test") is not None and np.isfinite(r["dir_acc_test"]):
            rec["dirAccTest"] = round(float(r["dir_acc_test"]), 4)
        # PHASOR finalization (operator 2026-07-20): interference-model fit quality — full fit,
        # walls-fit in-sample, and the honest last-24-months out-of-sample R².
        if r.get("dom_share") is not None: rec["domShare"] = round(float(r["dom_share"]), 3)
        if r.get("top_aspect"): rec["topAspect"] = r["top_aspect"]
        for src, dst in (("r2_full", "r2Full"), ("r2_ins", "r2Ins"), ("r2_oos", "r2Oos")):
            if r.get(src) is not None and np.isfinite(r[src]):
                rec[dst] = round(float(r[src]), 4)
        if r.get("ads_category"):
            rec["adsCategory"] = r["ads_category"]
            rec["adsPath"] = r.get("ads_path", "")
            rec["adsNPaths"] = int(r.get("ads_n_paths", 1))
        # WORLD-EVENT measure (a GDELT global-daily conflict/cooperation share unified into the atlas from
        # the retired /astro page) — flagged so /skills badges it and reads it as a measure, not a search topic.
        if r.get("world_event"):
            rec["worldEvent"] = True
        if y is not None:
            ya, fit = np.asarray(y, float), np.asarray(r["fitted"], float)
            m = min(len(ya), len(fit))                                   # align actual to the fitted training range
            ya, fa = ya[:m], fit[:m]                                     # RAW search interest (cbrt purged — no inverse needed)
            t0 = pd.Timestamp(t[0])
            start = int(t0.timestamp())                                  # unix seconds of day 0 → real dates on the chart
            yr0, yr1 = str(t0.year), str(pd.Timestamp(t[m - 1]).year)
            rec["series"] = {"actual": [round(float(v), 1) for v in ds(ya)],
                             "fit": [round(float(v), 1) for v in ds(fa)], "from": yr0, "to": yr1}
            # FULL daily resolution for the zoomable field chart — raw interest + a real start date + per-day residual.
            _bmeta = _MODEL_PRED.get("_meta") if isinstance(_MODEL_PRED, dict) else None
            _phasor = bool(_bmeta and _bmeta.get("task") == "phasor-r2")
            _mp = None
            try:
                _mp = _MODEL_PRED.get(r["key"])
            except Exception:
                _mp = None
            daily[r["key"]] = {"actual": [round(float(v)) for v in ya],
                               "fit": [round(float(v), 1) for v in fa],
                               "err": [round(float(a - b), 1) for a, b in zip(ya, fa)],   # per-point error (actual − model)
                               "forecast": [round(float(v), 1) for v in r.get("forecast", [])],  # 12-month winner FORECAST beyond the clean series
                               "start": start, "step": 2629746, "from": yr0, "to": yr1}   # MONTHLY cadence (~30.44 days)
            if _phasor:
                # PHASOR (the model of record, operator 2026-07-20): |b+Σa·e^(i(θ-p))| predicts the
                # RAW data directly — the balancing/square-wave era is PURGED from the chunks. The
                # curve is fitted on months < wall, a pure ephemeris forecast after it; predStart
                # anchors the 2008- model grid inside the longer raw series.
                if _mp and _mp.get("pred"):
                    pv = [round(float(v), 1) for v in _mp["pred"]]
                    # the prediction may extend EXT months beyond the data (to today) — anchor by
                    # the DATA months only: predStart = len(actual) - (len(pred) - ext).
                    _ext = int(_bmeta.get("ext", 0))
                    off = len(ya) - (len(pv) - _ext)
                    if off >= 0:
                        daily[r["key"]]["pred"] = pv
                        daily[r["key"]]["predStart"] = off
                        daily[r["key"]]["predWall"] = int(_bmeta.get("dep_wall", _bmeta.get("wall", 186)))
                        daily[r["key"]]["benchWall"] = int(_bmeta.get("wall", 186))
                        if _mp.get("bench"): daily[r["key"]]["bench"] = _mp["bench"]
                        daily[r["key"]]["r2Ins"] = _mp.get("r2Ins"); daily[r["key"]]["r2Oos"] = _mp.get("r2Oos")
                        if _mp.get("sunCurve"): daily[r["key"]]["sunCurve"] = _mp["sunCurve"]
                        if _mp.get("bodyShares"): daily[r["key"]]["bodyShares"] = _mp["bodyShares"]
                        if _mp.get("bodyPhases"): daily[r["key"]]["bodyPhases"] = _mp["bodyPhases"]
                        if _mp.get("topAspects"): daily[r["key"]]["topAspects"] = _mp["topAspects"]
                        if _mp.get("transits"): daily[r["key"]]["transits"] = _mp["transits"]
            else:
                # legacy direction-wave tasks (pre-phasor archives only)
                if _bmeta and _bmeta.get("task") == "ratechange-higher":
                    _yy = np.asarray(ya, float)
                    _r = np.zeros(len(_yy)); _r[1:] = (_yy[1:] - _yy[:-1]) / np.maximum(_yy[:-1], 1.0)
                    sqv = np.sign(_r[1:])
                elif _bmeta and _bmeta.get("task") == "balanced-highlow":
                    _tau = float(_bmeta["tau"])
                    sqv = np.where(np.asarray(ya)[1:] > _tau, 1, -1).astype(float)
                else:
                    dyv = np.diff(ya)
                    sqv = np.sign(dyv)
                    for _t in range(len(sqv)):
                        if _t == 0:
                            sqv[0] = 1 if sqv[0] == 0 else sqv[0]
                        elif sqv[_t] == 0:
                            sqv[_t] = sqv[_t - 1]
                H24 = 24
                if _mp and "sqPred" in _mp:
                    cut = len(sqv) - H24
                    sq_pred = [int(v) for v in _mp["sqPred"]]
                    sq_act = [int(v) for v in sqv[cut:cut + H24]]
                    sq_hit = [int(v) for v in _mp["sqHit"]]
                    sq_tie = [int(v) for v in _mp.get("sqTie", [1 if h == -1 else 0 for h in sq_hit])]
                elif len(sqv) > H24 + 24:
                    cut = len(sqv) - H24
                    moy_ = (np.arange(1, len(ya))) % 12
                    ages = (cut - 1 - np.arange(cut)) / 12.0
                    wts = 0.8 ** ages
                    prof = np.zeros(12)
                    for _m in range(12):
                        selm = moy_[:cut] == _m
                        prof[_m] = (sqv[:cut][selm] * wts[selm]).sum() / max(wts[selm].sum(), 1e-9)
                    sq_pred = [1 if prof[moy_[cut + k]] >= 0 else -1 for k in range(H24)]
                    sq_act = [int(v) for v in sqv[cut:cut + H24]]
                    sq_hit = [int(p == a2) for p, a2 in zip(sq_pred, sq_act)]
                    sq_tie = [1 if a2 == 0 else 0 for a2 in sq_act]
                else:
                    sq_pred, sq_act, sq_hit, sq_tie = [], [], [], []
                daily[r["key"]].update(sq=[int(v) for v in sqv], sqPred=sq_pred, sqHit=sq_hit, sqTie=sq_tie)
        topics.append(rec)
    # EXACT orbital periods measured day-by-day from the ephemeris (cached _mean_periods.json); fall back to the
    # tabulated PERIOD_YEARS where a body is missing. The cycles are sorted + displayed by this measured period.
    try:
        mper = tf.mean_periods()
    except Exception:
        mper = {}
    per = lambda b: float(mper.get(b, tf.PERIOD_YEARS[b]))
    cycles = [{"slug": tf.h_slug(b, 1), "label": f"{tf.nice(b)} · {tf.period_words(per(b))}", "body": tf.nice(b),
               "period": tf.period_words(per(b))}
              for b in sorted(tf.BODIES, key=per)]
    recency = json.load(open("analysis/_recency.json")) if os.path.exists("analysis/_recency.json") else None
    n = len(topics)
    GIST = "https://colab.research.google.com/gist/artaquest/6d2a073d195d3c075ac6d93d3c6f899d/"
    articles = [
        {"id": "seasonality", "title": "The Topics Seasonality Model", "authors": "ArtaQuest Foundation",
         "date": "2026", "thread": 1107, "notebook": "seasonality.ipynb", "colab": GIST + "seasonality.ipynb",
         "abstract": f"Does the timing of the world's curiosity keep time with the sky? We fit 22 years of MONTHLY "
                     f"Google-Trends interest for {n} search fields (skill terms derived from ISCO-08 occupations, plus "
                     f"learning topics) to eleven sidereal bodies — each a parameter-free SINC of its angular distance "
                     f"Δ to a shared phase, at a resonance PERIOD tuned per keyword (reported in years) to maximise F1 of the fit R² and the sign decisiveness. Bodies' own periods are measured day-by-day from "
                     f"the ephemeris — with exactly one non-negative weight per body and NO intercept (11 parameters), the "
                     f"shared phase swept on a 5° grid to place each field's zodiac sign. The median in-sample R² is "
                     f"carried mostly by the slow outer bodies acting as a trend basis plus the Sun's annual term: drop "
                     f"the slow bodies and it collapses. A transparent least-squares re-description of trend and "
                     f"seasonality, not a claim of cause."},
        {"id": "recency", "title": "Google Trends Recency Bias", "authors": "ArtaQuest Foundation",
         "date": "2026", "thread": 1108, "notebook": "recency.ipynb", "colab": GIST + "recency.ipynb",
         "abstract": "How much recent Google-Trends data should a model ignore? We crop the recent tail of each MONTHLY "
                     "series and refit, sweeping 0 to 6 years, and score each crop by pure in-sample R² — no hold-out. "
                     "Fitting power climbs as the noisy recent year is dropped, knees within roughly the first year, "
                     "then falls as too much history is discarded — so the model crops one year."},
    ]
    # Authorship (co-authorship supported — array; uid links to an ArtaQuest account for the profile listing).
    AUTHORS = [{"name": "Arash Ashrafnejad", "affiliation": "ArtaQuest Foundation", "uid": 138324856, "orcid": ""}]
    for a in articles:
        a["authors"] = AUTHORS
    # Citable locators (Nature-style): Volume 1 (2026), one article number per paper.
    ANUM = {"seasonality": 1, "recency": 2}
    for a in articles:
        a["volume"] = 1; a["issue"] = 1; a["articleNumber"] = ANUM[a["id"]]
    PAPER = {"seasonality": "topics-seasonality-model", "recency": "google-trends-recency-bias"}
    DOIS = json.load(open("analysis/_dois.json")) if os.path.exists("analysis/_dois.json") else {}
    PTITLE = {"seasonality": "The Topics Seasonality Model: Fitting 22 Years of Worldwide Search Interest to Sidereal Cycles",
              "recency": "Recency Bias in Google Trends: A Fitting-Power Criterion for Cropping the Recent Tail"}
    def fam_giv(nm):
        p = nm.split(); return p[-1], " ".join(p[:-1])
    for a in articles:
        sl = PAPER[a["id"]]; t = PTITLE[a["id"]]; url = f"https://artaquest.org/papers/{sl}.html"
        a["pdf"] = f"https://artaquest.org/papers/{sl}.pdf"; a["paper"] = url
        a["license"] = "CC BY 4.0"
        doi = DOIS.get(sl); a["doi"] = doi
        doi_apa = f" https://doi.org/{doi}" if doi else ""
        doi_nat = f" https://doi.org/{doi}" if doi else ""
        au = a["authors"]; fg = [fam_giv(x["name"]) for x in au]
        vol = a["volume"]; an = a["articleNumber"]
        apa_a = ", ".join(f"{f}, {g[0]}." for f, g in fg)
        nat_a = ", ".join(f"{g[0]}. {f}" for f, g in fg)
        bib_a = " and ".join(f"{f}, {g}" for f, g in fg)
        doi_bib = f"  doi     = {{{doi}}},\n" if doi else ""
        a["bibtex"] = ("@article{ashrafnejad2026" + sl.replace("-", "") + ",\n"
                       f"  title   = {{{t}}},\n  author  = {{{bib_a}}},\n  journal = {{Journal of Seasonality}},\n"
                       f"  volume  = {{{vol}}},\n  number  = {{{an}}},\n  year    = {{2026}},\n"
                       f"{doi_bib}  publisher = {{ArtaQuest Foundation}},\n  url     = {{{url}}}\n}}")
        a["cite"] = {
            "bibtex": a["bibtex"],
            "apa": f"{apa_a} (2026). {t}. Journal of Seasonality, {vol}, Article {an}.{doi_apa or ' ' + url}",
            "nature": f"{nat_a}. {t}. Journal of Seasonality {vol}, {an} (2026).{doi_nat}",
            "ris": "\n".join(["TY  - JOUR", f"TI  - {t}"] + [f"AU  - {f}, {g}" for f, g in fg] +
                             ["PY  - 2026", "JO  - Journal of Seasonality", f"VL  - {vol}", f"IS  - {an}"] +
                             ([f"DO  - {doi}"] if doi else []) +
                             ["PB  - ArtaQuest Foundation", f"UR  - {url}", "ER  - "]),
        }
    # Nature-style end-matter statements (mirrors analysis/build_papers.py:endmatter).
    DB = "https://artaquest.org/wp-content/uploads/research/"
    def initials(nm): return ".".join(x[0] for x in nm.split()) + "."
    for a in articles:
        inits = ", ".join(initials(x["name"]) for x in a["authors"])
        a["statements"] = [
            {"h": "Data availability", "txt": f"All data underlying this study are openly available at {DB} — the "
             f"full-resolution monthly series, the sidereal ephemeris, and the result tables."
             + (f" The dataset is permanently archived on Zenodo (DOI {DOIS['dataset']}, "
                f"https://doi.org/{DOIS['dataset']})." if DOIS.get("dataset") else "")},
            {"h": "Code availability", "txt": f"The complete analysis code is open in the ArtaQuest repository "
             f"(analysis/). A one-click Google Colab notebook reproduces every figure and number: {a['colab']}"},
            {"h": "Author contributions", "txt": f"{inits} designed the study, performed the analysis, and wrote the manuscript."},
            {"h": "Competing interests", "txt": "The author declares no competing financial or non-financial interests."},
            {"h": "Funding", "txt": "This work received no external funding and was conducted under the ArtaQuest Foundation."},
        ]
    # The shared monthly TIME AXIS (YYYY-MM) for the fit+forecast plot — the clean series [0, clean_end)
    # plus the 12 forecast months, identical for every topic (each daily chunk's actual/forecast align to
    # it), so the CompResultsChart-style plot needs no per-topic month strings. clean_end = GRID − 12.
    _grid = pd.DatetimeIndex(tf.GRID)
    months_axis = [d.strftime("%Y-%m") for d in _grid]                       # 2004-01 … 2026-06 (full grid)
    clean_end = len(_grid) - tf.DROP_LAST
    _agg = (_MODEL_PRED.get("_meta") or {}).get("aggregate") if isinstance(_MODEL_PRED, dict) else None
    out = {"n": n, "signs": tf.SIGNS, "cycles": cycles, "topics": topics, "recency": recency, "aggregate": _agg,
           "monthsAxis": months_axis, "cleanEnd": clean_end,
           "articles": articles, "dataBase": "https://artaquest.org/wp-content/uploads/research/"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), separators=(",", ":"), allow_nan=False)
    # Full daily raw series — one file PER FIELD, dynamic-imported by key only when that field page opens, so each
    # page loads ~25 KB (not the whole atlas) and it scales to thousands of fields. Stale files are pruned.
    import glob
    DAILY_DIR = "artaquest-web/src/data/field-daily"
    os.makedirs(DAILY_DIR, exist_ok=True)
    for f in glob.glob(f"{DAILY_DIR}/*.json"):
        if os.path.basename(f)[:-5] not in daily:
            os.remove(f)
    for key, obj in daily.items():
        json.dump(obj, open(f"{DAILY_DIR}/{key}.json", "w"), separators=(",", ":"))
    # Legacy single-file bundle removed in favour of per-field files.
    if os.path.exists("artaquest-web/src/data/research-daily.json"):
        os.remove("artaquest-web/src/data/research-daily.json")
    print(f"  wrote {len(daily)} per-field daily files → {DAILY_DIR}/ (avg {sum(os.path.getsize(f'{DAILY_DIR}/{k}.json') for k in daily)//max(1,len(daily))//1024} KB)")
    # Lightweight article index for profile "Publications" (so Profile.tsx need not load the full atlas).
    idx = [{"id": a["id"], "title": a["title"], "date": a["date"], "authors": a["authors"],
            "url": f"/research/?article={a['id']}", "pdf": a["pdf"], "journal": "Journal of Seasonality",
            "doi": a.get("doi"), "volume": a.get("volume"), "articleNumber": a.get("articleNumber")} for a in articles]
    json.dump(idx, open("artaquest-web/src/data/articles.json", "w"), separators=(",", ":"))
    print(f"  wrote {OUT}: {len(topics)} topics, {len(cycles)} cycles, {os.path.getsize(OUT)//1024} KB · + articles.json ({len(idx)})")

if __name__ == "__main__":
    main()
