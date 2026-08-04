#!/usr/bin/env python3
"""Astro-1200 — download Google Trends Explore "Top queries" CSVs, one of the FIVE search properties (web/images/news/shop/youtube; arg 1),
Worldwide, LUNAR MONTH BY LUNAR MONTH: each window runs from one NEW MOON to the day before the
next (the platform's Season calendar) (new_moons.json — computed astronomically on the platform ephemeris, sidereal USHASHASHI).
Operator's source of record: https://trends.google.com/explore?geo=Worldwide&gprop=news&date=...

For each lunation: explore (empty query, property=news) -> RELATED_QUERIES widget -> the widget's
CSV export endpoint (the UI's download button) -> analysis/astro1200/news_lunar_<fullmoon-date>.csv
(raw, kept as the provenance artifact). Rides 429s. Resumable (skips saved windows).

  python3 analysis/astro1200/news_topcharts.py            # all lunations 2008..2025
"""
import datetime as dt, importlib.util as u, json, os, sys, time, urllib.parse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

tf = _load("analysis/trends_fit.py", "tf")
HERE = "analysis/astro1200"

def _get(sess, url):
    """GET riding out 429s (the platform ladder)."""
    for wait in (20, 60, 180, 420, 900, None):
        time.sleep(2)
        r = sess.get(url, timeout=25)
        if r.status_code != 429:
            return r
        if wait is None:
            return r
        print(f"    429 — riding {wait}s", flush=True)
        time.sleep(wait)
    return r

def window_csv(sess, d0, d1):
    out = f"{HERE}/{PROP_NAME}_lunar_{d0}.csv"
    if os.path.exists(out):
        return "cached"
    req = {"comparisonItem": [{"keyword": "", "geo": "", "time": f"{d0} {d1}"}],
           "category": 0, "property": PROP_API}
    r1 = _get(sess, "https://trends.google.com/trends/api/explore?hl=en-US&tz=0&req="
              + urllib.parse.quote(json.dumps(req)))
    if r1.status_code != 200:
        return f"explore {r1.status_code}"
    w = json.loads(r1.text[r1.text.find("{"):])["widgets"]
    rq = next((x for x in w if x.get("id") == "RELATED_QUERIES"), None)
    if rq is None:
        return "no RELATED_QUERIES widget"
    r2 = _get(sess, "https://trends.google.com/trends/api/widgetdata/relatedsearches/csv?hl=en-US&tz=0&req="
              + urllib.parse.quote(json.dumps(rq["request"], separators=(",", ":"))) + "&token=" + rq["token"])
    if r2.status_code != 200:
        return f"csv {r2.status_code}"
    open(out, "w").write(r2.text)
    return f"saved ({len(r2.text)} bytes)"

PROPS = {"web": "", "images": "images", "news": "news", "shop": "froogle", "youtube": "youtube"}
PROP_NAME = sys.argv[1] if len(sys.argv) > 1 else "news"
PROP_API = PROPS[PROP_NAME]

def main():
    fms = json.load(open(f"{HERE}/new_moons.json"))
    sess = tf.make_session()
    for a, b in zip(fms, fms[1:]):
        d0 = a["date"]
        if not ("2008-01-01" <= d0 <= "2025-12-31"):
            continue
        d1 = (dt.date.fromisoformat(b["date"]) - dt.timedelta(days=1)).isoformat()
        print(f"  {d0} -> {d1} (sun {a['sun_sign']}): {window_csv(sess, d0, d1)}", flush=True)

if __name__ == "__main__":
    main()
