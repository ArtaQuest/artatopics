#!/usr/bin/env python3
"""Regenerate the PLUGIN disciplines files from the funnel's daily fits — the single source of /fields + house reps.

The funnel writes analysis/houses_daily.json (one genuine daily sidereal fit per finalised keyword). This script
projects each fit into the two plugin data files the backend imports:

  wp-content/plugins/aquest/data/aq-disciplines-add.json   {items:[{key,house,label,sign,score,sig,central,daily,pos,blurb}]}
  wp-content/plugins/aquest/data/aq-disc-trends.json        {key:{ref,kind,query,series}}   (120-pt daily series)

Mapping (canonical, 1:1 by zodiac index):
  zodiac sign  → academic house  (WHAT axis)   and  → style label (HOW-of-the-field, the `sign` column)
  score        = round(rep * 1000)             (rep = (1 - mean/max)·R², the representativeness)
  central      = 1 if the tuning peak sits within the central 20° of its sign (|ref%30 - 15| ≤ 10)
  pos          = 'adj' if the keyword is in the ADJECTIVE pool (_adjectives_pool.py) else 'noun'  ← the WHAT/HOW split

So a discovered ADJECTIVE (e.g. "epic") lands in a house as a HOW-camp field, and a NOUN (e.g. "festival") as a
WHAT-camp field; Topics::house_reps then picks the top-score rep of EACH camp. Run after the funnel finalises new
keywords (build_funnel.merge_and_deploy calls this); the plugin file ships on the next plugin deploy.

  python3 analysis/build_disciplines.py            # regenerate both files
  python3 analysis/build_disciplines.py --check     # print what would change, write nothing
"""
import json, os, sys, csv

DAILY = os.environ.get("AQ_REG", "analysis/_fields_weekly.json")     # registry to build from (merge both tracks via AQ_REG)
DATA_DAILY = os.environ.get("AQ_DATA", "analysis/data_monthly")     # monthly series dir (weekly purged)
OUT_DISC = "wp-content/plugins/aquest/data/aq-disciplines-add.json"
OUT_TRENDS = "wp-content/plugins/aquest/data/aq-disc-trends.json"

ZODIAC = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
          "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
HOUSES = ["psychology", "economics", "communication", "history", "arts", "medicine",
          "politics", "anthropology", "philosophy", "management", "sociology", "theology"]
STYLE_KEYS = ["powerful", "practical", "quick", "calming", "dramatic", "detailed",
              "balanced", "mysterious", "epic", "professional", "innovative", "creative"]

# Adjective vocabulary → the HOW camp. Import lazily so the file is optional in minimal checkouts.
try:
    import importlib.util as _u
    _s = _u.spec_from_file_location("adj", "analysis/_adjectives_pool.py")
    _m = _u.module_from_spec(_s); _s.loader.exec_module(_m); ADJ_SET = _m.ADJ_SET
except Exception:
    ADJ_SET = set()

# Junk-word gate — a stop-word / generic term never becomes a field, even if it lingers in a registry.
try:
    import importlib.util as _u2
    _s2 = _u2.spec_from_file_location("sw", "analysis/_stopwords.py")
    _m2 = _u2.module_from_spec(_s2); _s2.loader.exec_module(_m2); is_junk = _m2.is_junk
except Exception:
    def is_junk(w):
        return False

# Recompute rep with the CURRENT formula (max/2nd-max area × popularity × R²) so the `score` (= rep×1000), which
# BOTH the /fields atlas and the Studio Houses mode rank by, is never stale — a field fit under an older rep
# formula is re-scored here at build time. Optional: falls back to the stored rep in a minimal checkout.
try:
    import importlib.util as _u3
    _s3 = _u3.spec_from_file_location("tf", "analysis/trends_fit.py")
    _m3 = _u3.module_from_spec(_s3); _s3.loader.exec_module(_m3); _rep_score = _m3.rep_score
except Exception:
    _rep_score = None
def cur_rep(r):
    if _rep_score is not None:
        try: return float(_rep_score(r))
        except Exception: pass
    return float(r.get("rep", 0.0))

# RATIONAL house assignment: a field's house is its MEANING, not the arbitrary sidereal sign→house mapping.
#   1. an explicit rational override (analysis/_field_houses.json),
#   2. else the dominant ACADEMIC house of its source topics (analysis/_topic_houses.json — topic → house),
#   3. else, only as a last resort, the fit sign's house.
FIELD_HOUSES = json.load(open("analysis/_field_houses.json")) if os.path.exists("analysis/_field_houses.json") else {}
TOPIC_HOUSE = json.load(open("analysis/_topic_houses.json")) if os.path.exists("analysis/_topic_houses.json") else {}
from collections import Counter
VALID_HOUSES = set(["psychology", "economics", "communication", "history", "arts", "medicine",
                    "politics", "anthropology", "philosophy", "management", "sociology", "theology"])

def rational_house(key, topics, sign_house):
    h = FIELD_HOUSES.get(key.lower())
    if h in VALID_HOUSES:
        return h
    votes = Counter(TOPIC_HOUSE[t] for t in (topics or []) if TOPIC_HOUSE.get(t) in VALID_HOUSES)
    if votes:
        return votes.most_common(1)[0][0]
    return sign_house


def slugify(s):
    return "".join(c if c.isalnum() else "-" for c in s.strip().lower()).strip("-")


def downsample(vals, n=120):
    """Average `vals` into `n` evenly-spaced bins (the compact series the cards plot)."""
    if not vals:
        return []
    if len(vals) <= n:
        return [int(round(v)) for v in vals]
    out = []
    for i in range(n):
        a = i * len(vals) // n
        b = max(a + 1, (i + 1) * len(vals) // n)
        chunk = vals[a:b]
        out.append(int(round(sum(chunk) / len(chunk))))
    return out


def read_series(key, label):
    for name in (key, slugify(label)):
        p = f"{DATA_DAILY}/{name}.csv"
        if os.path.exists(p):
            vals = []
            with open(p) as f:
                for row in csv.DictReader(f):
                    try:
                        vals.append(float(row.get("v", 0)))
                    except (TypeError, ValueError):
                        pass
            return downsample(vals)
    return []


def build():
    reg = json.load(open(DAILY)) if os.path.exists(DAILY) else {}
    items, trends = [], {}
    for key, r in sorted(reg.items()):
        if r.get("res") != "weekly":
            continue
        sign = r.get("sign")
        if sign not in ZODIAC:
            continue
        idx = ZODIAC.index(sign)                  # the fit's sign IS the house — house = sign (1:1)
        label = str(r.get("label") or key)
        ref = float(r.get("ref", 0.0))
        # pos comes from the record (the topic camp it was proposed in); fall back to the adjective pool.
        pos = r.get("pos") if r.get("pos") in ("noun", "adj") else ("adj" if key.lower() in ADJ_SET else "noun")
        items.append({
            "key": key,
            "axis": r.get("axis", "house"),       # 'house'=Professions track · 'topic'=Topics track (for the two pages)
            "house": HOUSES[idx],                 # house = the field's best-fit zodiac sign (the HOUSES slot at the sign index)
            "label": label,
            "sign": STYLE_KEYS[idx],
            "score": int(round(cur_rep(r) * 1000)),   # rank key (rep×1000) — recomputed, never stale
            "sig": int(round(float(r.get("sig", 0.0)))),
            # REP-ELIGIBLE: a field may be the house REPRESENTATIVE only when its primary sign beats the runner-up by
            # > 1.5× area (house_ratio). (Reuses the `central` flag the rep selector already gates on.) Every field is
            # still PUBLISHED in its house — this only decides who may be the title.
            "central": 1 if float(r.get("house_ratio", 0.0)) > 1.5 else 0,
            "ratio": round(float(r.get("house_ratio", 0.0)), 3),
            "daily": 1,
            "pos": pos,
            "freq": int(r.get("freq", 0)),                   # commonness (how many topics proposed it)
            "topics": list(r.get("topics", [])),             # source topics (provenance)
            "blurb": "",
        })
        series = read_series(key, label)
        if series:
            trends[key] = {"ref": f"disc:{key}", "kind": "discipline", "query": label, "series": series}
    return {"items": items}, trends


def main():
    disc, trends = build()
    n_adj = sum(1 for it in disc["items"] if it["pos"] == "adj")
    if "--check" in sys.argv:
        print(f"[build_disciplines] {len(disc['items'])} fields ({n_adj} adj / {len(disc['items']) - n_adj} noun), "
              f"{len(trends)} with series — NOT written (--check)")
        return
    json.dump(disc, open(OUT_DISC, "w"), ensure_ascii=False, indent=1)
    json.dump(trends, open(OUT_TRENDS, "w"), ensure_ascii=False)
    print(f"[build_disciplines] wrote {len(disc['items'])} fields ({n_adj} adj / {len(disc['items']) - n_adj} noun) "
          f"→ {OUT_DISC}; {len(trends)} series → {OUT_TRENDS}")


if __name__ == "__main__":
    main()
