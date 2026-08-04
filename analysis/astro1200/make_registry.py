#!/usr/bin/env python3
"""Astro4 → prod topics registry. Builds the platform pool + registry from the four study sets:

- key/label: the topic's slug; a topic selected in MORE THAN ONE modality gets per-modality entries
  labelled "<topic> (<modality>)" (key = its slug) so each keeps its own property series and page.
- series: each topic's fetched own-property monthly CSV copied into analysis/data_monthly_astro4/
  (the platform pipeline reads it via the AQ_DATA_MONTHLY override; the shared web cache is untouched).
- rec: the standard platform fit (tf.fit_topic on the clean window) + shares + rep; popularity is the
  topic's CHART score (avg per-lunation TOP value — the study's own popularity measure); system =
  the dataset name (ImageTopics/NewsTopics/ShopTopics/YouTubeTopics); the frozen chart class rides
  along as chart_class. The sinc-gd season overlay comes from reanalyze_topic500_winner.py next.

  python3 analysis/astro1200/make_registry.py
→ analysis/_topics.json + analysis/_topics_weekly.json + analysis/data_monthly_astro4/
"""
import importlib.util as u, json, os, shutil
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
os.environ["AQ_DATA_MONTHLY"] = "analysis/data_monthly_astro4"

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

tf = _load("analysis/trends_fit.py", "tf")
HERE = "analysis/astro1200"
PROPS = {"images": "ImageTopics", "news": "NewsTopics", "shop": "ShopTopics", "youtube": "YouTubeTopics"}
OUTDIR = "analysis/data_monthly_astro4"
os.makedirs(OUTDIR, exist_ok=True)

def main():
    lon = tf.ephemeris()
    # count modalities per topic (to decide qualified labels)
    counts = {}
    sel = {}
    for prop, name in PROPS.items():
        terms = json.load(open(f"{HERE}/{prop}_chart_terms.json"))
        sel[prop] = {t: r for t, r in terms.items() if r.get("selected")}
        for t in sel[prop]:
            counts[t] = counts.get(t, 0) + 1

    pool, reg = {}, {}
    n_fit = n_skip = 0
    for prop, name in PROPS.items():
        short = {"images": "images", "news": "news", "shop": "shopping", "youtube": "youtube"}[prop]
        for t, r in sel[prop].items():
            label = t if counts[t] == 1 else f"{t} ({short})"
            key = tf.slug(label)
            src = f"{HERE}/series_{prop}/{tf.slug(t)}.csv"
            if not os.path.exists(src):
                n_skip += 1; continue
            shutil.copyfile(src, f"{OUTDIR}/{key}.csv")
            _, y = tf.load_y(label)
            if y is None:
                n_skip += 1; continue
            rec = tf.fit_topic(y, lon)
            rec.update(key=key, label=label, res="weekly", pos="noun", freq=0, topics=[],
                       isco="", axis="topic", system=name)
            rec["popularity"] = round(float(r["avg_score"]), 1)     # chart-based popularity
            rec["shares"] = tf.body_shares(rec, lon, y)
            rec["rep"] = tf.rep_score(rec)
            rec["chart_class"] = r["class"]                          # the frozen chart-derived season
            rec["chart_lunations"] = r["n_lunations"]
            reg[key] = rec
            pool[key] = {"axis": "topic", "label": label, "query": t, "system": name,
                         "chart_class": r["class"], "final": 1}
            n_fit += 1
            if n_fit % 200 == 0:
                print(f"  {n_fit} fitted…", flush=True)

    json.dump(pool, open("analysis/_topics.json", "w"), indent=1)
    json.dump(reg, open("analysis/_topics_weekly.json", "w"), indent=0)
    print(f"registry: {n_fit} topics fitted ({n_skip} skipped) → _topics.json/_topics_weekly.json"
          f" · series cache: {OUTDIR}")

if __name__ == "__main__":
    main()
