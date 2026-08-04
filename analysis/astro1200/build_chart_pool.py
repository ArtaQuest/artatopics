#!/usr/bin/env python3
"""Astro-1200 — build topic1200 from the LUNAR-MONTH Google News top-50 charts (2008-2025).

Source of record: one Trends Explore "Top queries" CSV per lunation (new moon -> day before the next new moon; Worldwide; Google News property) — news_lunar_<fullmoon-date>.csv. Each lunation's
top-50 = its TOP 25 + RISING 25 rows.

- A lunation's SEASON = the sidereal sign of the Sun at the lunation midpoint (USHASHASHI).
- SCORE: per lunation a topic scores its ACTUAL TOP-chart interest value (1-100) when it appears in
  that lunation's top-50 and 0 when it does not (RISING-only rows carry no 0-100 value -> 0).
- RANKING/SELECTION (universal, same for every property): avg_score = the topic's mean per-lunation
  score over ALL charted lunations (2008-2025; absent lunations count 0); all topics are ranked by it
  and the top 20% of the ranked unique terms form the dataset (20th-percentile rule).
- CLASSIFICATION: argmax season by SUMMED SCORE (ties: appearance count, then season order).
- PRE-REGISTERED classification = the ARGMAX season over a term's charted lunations (halloween
  charting most in its autumn season is classified to that season); ties broken by the season with
  the higher summed TOP interest, then season order — fully deterministic, purely data-driven.

Filters at parse: lowercase ASCII, <=2 words after stripping the token "news" ("india news" merges into "india"; the bare term "news" itself is excluded). NO topical/entity filtering — anything that charts counts.
Outputs: lunations.json, chart_terms.json, chart_stats.json.

  python3 analysis/astro1200/build_chart_pool.py
"""
import csv, datetime as dt, io, json, os, re, sys, unicodedata
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROP = sys.argv[1] if len(sys.argv) > 1 else "news"
TOP_PCT = 0.20       # universal selection: the top 20th PERCENTILE (top 20%) of ranked terms
SCORE_FROM = '2008-01-01'  # ranking window = ALL charted lunations (2008-2025)
SIGNS = ['Aries','Taurus','Gemini','Cancer','Leo','Virgo','Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces']


def norm(t):
    t = unicodedata.normalize('NFKD', str(t)).encode('ascii', 'ignore').decode()
    t = re.sub(r"[^a-z0-9 ]+", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def midpoint_sun_sign(d0, d1):
    os.environ.setdefault("KERYKEION_GEONAMES_USERNAME", "dummy")
    from kerykeion import AstrologicalSubject
    mid = d0 + (d1 - d0) / 2
    s = AstrologicalSubject("Q", mid.year, mid.month, mid.day, 12, 0, lat=0, lng=0, tz_str="UTC",
                            zodiac_type="Sidereal", sidereal_mode="USHASHASHI")
    return SIGNS[int(s.sun.abs_pos // 30) % 12]


def parse_lunation(path):
    """-> [(term, section, top_interest_or_0)] for one lunation top-50 CSV, filters applied."""
    rows, section = [], None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("Category:"):
            continue
        if line in ("TOP", "RISING"):
            section = line; continue
        if section is None:
            continue
        parts = list(csv.reader(io.StringIO(line)))[0]
        if len(parts) < 2:
            continue
        t = norm(parts[0])
        t = " ".join(w for w in t.split() if w != "news")   # "india news" -> "india"; bare "news" -> dropped
        if not t or len(t.split()) > 2:
            continue
        v = int(parts[1]) if section == "TOP" and parts[1].strip().isdigit() else 0
        rows.append((t, section, v))
    return rows


def main():
    fms = [dt.date.fromisoformat(f["date"]) for f in json.load(open(os.path.join(HERE, "new_moons.json")))]
    lunations = {}
    terms = defaultdict(lambda: {"lunations": [], "seasons": defaultdict(int),
                                 "season_interest": defaultdict(int), "top_interest": 0})
    for a, b in zip(fms, fms[1:]):
        if not (dt.date(2008, 1, 1) <= a <= dt.date(2025, 12, 31)):
            continue
        p = os.path.join(HERE, f"{PROP}_lunar_{a.isoformat()}.csv")
        if not os.path.exists(p):
            continue
        end = b - dt.timedelta(days=1)
        season = midpoint_sun_sign(a, end)
        rows = parse_lunation(p)
        lunations[a.isoformat()] = {"end": end.isoformat(), "season": season, "rows": len(rows)}
        in_window = a.isoformat() >= SCORE_FROM
        seen = set()
        for t, section, v in rows:
            r = terms[t]
            if t not in seen:                                   # one appearance per lunation
                seen.add(t)
                r["lunations"].append(a.isoformat())
                r["seasons"][season] += 1
            r["season_interest"][season] += v
            r["top_interest"] += v
            if in_window:
                r["window_score"] = r.get("window_score", 0) + v

    n_window = sum(1 for d in lunations if d >= SCORE_FROM)
    out = {}
    for t, r in terms.items():
        mx = max(r["season_interest"].get(s, 0) for s in SIGNS)   # argmax by SUMMED SCORE
        cands = [s for s in SIGNS if r["season_interest"].get(s, 0) == mx]
        if len(cands) > 1:
            mi = max(r["seasons"].get(s, 0) for s in cands)       # tie: appearance count
            cands = [s for s in cands if r["seasons"].get(s, 0) == mi] or cands
        out[t] = {"n_lunations": len(r["lunations"]), "class": cands[0],
                  "seasons": dict(r["seasons"]), "season_score": dict(r["season_interest"]),
                  "top_interest": r["top_interest"],
                  "avg_score": round(r.get("window_score", 0) / n_window, 3) if n_window else 0.0,
                  "lunations": r["lunations"]}

    ranked_all = sorted(out.items(), key=lambda kv: (-kv[1]["avg_score"], -kv[1]["n_lunations"], kv[0]))
    n_sel = max(1, int(round(TOP_PCT * len(ranked_all))))
    selected = dict(ranked_all[:n_sel])
    for t in selected: out[t]["selected"] = True
    json.dump(lunations, open(os.path.join(HERE, f"{PROP}_lunations.json"), "w"), indent=0)
    json.dump(out, open(os.path.join(HERE, f"{PROP}_chart_terms.json"), "w"), indent=0)
    ranked = sorted(out.items(), key=lambda kv: (-kv[1]["avg_score"], -kv[1]["n_lunations"], kv[0]))
    json.dump({"lunations": len(lunations), "unique_terms": len(out), "top_pct": TOP_PCT, "n_selected": len(selected),
               "selected": len(selected),
               "n_ge_2": sum(1 for _, v in out.items() if v["n_lunations"] >= 2),
               "head": [(k, v["avg_score"], v["class"]) for k, v in ranked[:50]]},
              open(os.path.join(HERE, f"{PROP}_chart_stats.json"), "w"), indent=1)
    print(f"[{PROP}] lunations: {len(lunations)} · unique terms: {len(out)} · top {int(TOP_PCT*100)}% -> {len(selected)} selected")
    print("head:", [(k, v["avg_score"], v["class"]) for k, v in ranked[:12]])


if __name__ == "__main__":
    main()
