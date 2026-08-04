#!/usr/bin/env python3
"""AdsTopics → prod topics registry (the prod upgrade, operator order 2026-07-15).

Stage --fit (CPU-parallel, runs alongside the GPU atlas):
  every audit-clean adstopics series is copied into analysis/data_monthly_adstopics/ and fitted
  with the PLATFORM model (tf.fit_topic — the 11-body NNLS + period sweep every live topic page
  renders); recs accumulate in analysis/adstopics/_registry_fits.json (resumable).

Stage --overlay (after the FINAL Gaussian atlas lands):
  each rec's classification is overridden by the atlas of the model of record — sign from the
  optimised Gaussian window direction, class_model="gauss-gd", with tier/r2_val/r2_test/
  season_led/lamps riding along — then the prod registry pair is written:
    analysis/_topics.json          (pool: key -> {axis,label,query,system,final})
    analysis/_topics_weekly.json   (full recs, AQ_REG for build_disciplines + export_research)

  python3 analysis/adstopics/make_registry.py --fit [workers]
  python3 analysis/adstopics/make_registry.py --overlay
"""
import importlib.util as u, json, os, shutil, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
os.environ["AQ_DATA_MONTHLY"] = "analysis/data_monthly_adstopics"

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

OUTDIR = "analysis/data_monthly_adstopics"
FITS = "analysis/adstopics/_registry_fits.json"


def usable_topics():
    tf = _load("analysis/trends_fit.py", "tf")
    grid = pd.DatetimeIndex(tf.GRID)
    i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
    iend = len(grid) - tf.DROP_LAST
    vocab = json.load(open("analysis/adstopics/vocabulary.json"))
    bl = set(json.load(open("analysis/adstopics/blacklist.json")).get("excluded_topics", []))
    out = []
    for t in sorted(vocab):
        if t in bl: continue
        p = f"analysis/adstopics/series/{tf.slug(t)}.csv"
        if not os.path.exists(p): continue
        df = pd.read_csv(p); df["Time"] = pd.to_datetime(df["Time"])
        v = pd.to_numeric(df.drop_duplicates("Time").set_index("Time")["v"].reindex(grid[i0:iend]),
                          errors="coerce")
        if v.notna().sum() < (iend - i0) * 0.5: continue
        y = v.interpolate(limit_direction="both").to_numpy(float)
        if not np.isfinite(y).all() or float(y.max()) <= 0: continue
        out.append(t)
    return out


def _fit_one(t):
    tf = _load("analysis/trends_fit.py", "tf")
    lon = tf.ephemeris()
    key = tf.slug(t)
    src = f"analysis/adstopics/series/{key}.csv"
    dst = f"{OUTDIR}/{key}.csv"
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    _, y = tf.load_y(t)
    if y is None:
        return None
    rec = tf.fit_topic(y, lon)
    rec.update(key=key, label=t, res="weekly", pos="noun", freq=0, topics=[], isco="",
               axis="topic", system="AdsTopics",
               popularity=round(float(np.mean(y)), 1))
    rec["shares"] = tf.body_shares(rec, lon, y)
    rec["rep"] = tf.rep_score(rec)
    return key, rec


def stage_fit(workers):
    os.makedirs(OUTDIR, exist_ok=True)
    topics = usable_topics()
    done = json.load(open(FITS)) if os.path.exists(FITS) else {}
    tf = _load("analysis/trends_fit.py", "tf")
    todo = [t for t in topics if tf.slug(t) not in done]
    print(f"[registry --fit] usable {len(topics)} · fitted {len(done)} · todo {len(todo)} · workers {workers}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_fit_one, todo, chunksize=8)):
            if r is not None:
                done[r[0]] = r[1]
            if (i + 1) % 100 == 0:
                json.dump(done, open(FITS, "w"))
                print(f"  fitted {len(done)}/{len(topics)}", flush=True)
    json.dump(done, open(FITS, "w"))
    print(f"[registry --fit] complete: {len(done)} recs → {FITS}")


def stage_overlay_ts():
    """Classification of record = the AstroTS atlas (operator 2026-07-16): the pure time-series
    model's predicted growth season, for EVERY topic (no fallback needed — the atlas covers the
    entire population)."""
    done = json.load(open(FITS))
    att = pd.read_csv("analysis/adstopics/astro_ts_atlas.csv").set_index("topic")
    ga = pd.read_csv("analysis/adstopics/atlas_topics.csv").set_index("topic")
    vocab = json.load(open("analysis/adstopics/vocabulary.json"))
    from collections import Counter
    pool, reg = {}, {}
    n_ts = 0
    for key, rec in done.items():
        t = rec["label"]
        if t in att.index:
            a = att.loc[t]
            rec["class_model"] = "phasor" if "r2_oos" in att.columns else "astrots"
            if "sign" in att.columns: rec["sign"] = str(a["sign"])
            if "phase" in att.columns: rec["attn_phase"] = float(a["phase"])
            if "peak_month" in att.columns: rec["peak_month"] = str(a["peak_month"])
            if "dominant_planet" in att.columns: rec["dominant_planet"] = str(a["dominant_planet"])
            if "sun_share" in att.columns: rec["sun_share"] = float(a["sun_share"])
            if "dom_share" in att.columns: rec["dom_share"] = float(a["dom_share"])
            if "top_aspect" in att.columns and str(a.get("top_aspect", "")) not in ("", "nan"): rec["top_aspect"] = str(a["top_aspect"])
            if "phase_stable" in att.columns: rec["phase_stable"] = int(a["phase_stable"])
            if "conf" in att.columns: rec["attn_conf"] = float(a["conf"])
            if "arm" in att.columns: rec["pred_arm"] = str(a["arm"])
            if "dir_acc_test" in att.columns: rec["dir_acc_test"] = float(a["dir_acc_test"])
            # PHASOR finalization (operator 2026-07-20): per-topic fit quality, in and out of sample.
            for c in ("r2_full", "r2_ins", "r2_oos", "r2_oos_strict"):
                if c in att.columns: rec[c] = float(a[c])
            n_ts += 1
        if t in ga.index:
            g = ga.loc[t]
            rec["season_led"] = bool(g["season_led"]); rec["tier"] = str(g["tier"])
        # REAL organization (operator 2026-07-16, no made-up naming): the topic's own Google Ads
        # taxonomy placement — majority top-level vertical + its primary full path, verbatim.
        vp = (vocab.get(t) or {}).get("paths") or []
        if vp:
            tops = Counter(p.split("/")[1].strip() for p in vp if p.count("/") >= 1)
            rec["ads_category"] = tops.most_common(1)[0][0]
            rec["ads_path"] = sorted(vp, key=len)[0]
            rec["ads_n_paths"] = len(vp)
        reg[key] = rec
        pool[key] = {"axis": "topic", "label": t, "query": t, "system": "AdsTopics",
                     "season_led": rec.get("season_led", False), "final": 1}
    json.dump(pool, open("analysis/_topics.json", "w"), indent=1)
    json.dump(reg, open("analysis/_topics_weekly.json", "w"), indent=0)
    print(f"[registry --overlay-ts] {len(reg)} recs ({n_ts} astrots-classified)")


def stage_overlay_attn():
    """Classification of record = the ANCHORED ATTENTION atlas (operator 2026-07-16). Gated topics
    carry the ensemble p-hat sign; below-gate topics fall back to the same anchor frame (the
    recency-weighted profile-peak direction) so every topic is classified in ONE coordinate system."""
    done = json.load(open(FITS))
    att = pd.read_csv("analysis/adstopics/attention_atlas.csv").set_index("topic")
    tf = _load("analysis/trends_fit.py", "tf")
    r5 = _load("analysis/topic500_reference_solution.py", "r5")
    ga = pd.read_csv("analysis/adstopics/atlas_topics.csv").set_index("topic")
    grid = pd.DatetimeIndex(tf.GRID)
    i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0]); iend = len(grid) - tf.DROP_LAST
    theta = np.deg2rad(np.arange(12) * 30.0 + 15.0)
    pool, reg = {}, {}
    n_att = n_fb = 0
    for key, rec in done.items():
        t = rec["label"]
        if t in att.index:
            a = att.loc[t]
            rec["class_model"] = "astroattention"
            rec["sign"] = str(a["sign"])
            rec["attn_phase"] = float(a["phase"]); rec["attn_conf"] = float(a["conf"])
            rec["dir_acc_test"] = float(a["dir_acc_test"])
            n_att += 1
        else:
            p = f"analysis/adstopics/series/{key}.csv"
            df = pd.read_csv(p); df["Time"] = pd.to_datetime(df["Time"])
            v = pd.to_numeric(df.drop_duplicates("Time").set_index("Time")["v"]
                              .reindex(grid[i0:iend]), errors="coerce")
            y = v.interpolate(limit_direction="both").to_numpy(float)
            dy = np.diff(y)
            a1 = len(y) - 48 - 1
            ages = (a1 - 1 - np.arange(a1)) / 12.0
            w = 0.8 ** ages
            moy = np.arange(1, len(y)) % 12
            prof = np.array([(dy[:a1][moy[:a1] == m] * w[moy[:a1] == m]).sum() /
                             max(w[moy[:a1] == m].sum(), 1e-9) for m in range(12)])
            pos = np.clip(prof, 0, None)
            ph = float(np.rad2deg(np.arctan2((pos * np.sin(theta)).sum(),
                                             (pos * np.cos(theta)).sum())) % 360.0)
            rec["class_model"] = "astroattention-anchor"
            rec["sign"] = r5.SIGNS[int(ph // 30) % 12]
            rec["attn_phase"] = round(ph, 2)
            n_fb += 1
        if t in ga.index:
            g = ga.loc[t]
            rec["season_led"] = bool(g["season_led"]); rec["tier"] = str(g["tier"])
        # REAL organization (operator 2026-07-16, no made-up naming): the topic's own Google Ads
        # taxonomy placement — majority top-level vertical + its primary full path, verbatim.
        vp = (vocab.get(t) or {}).get("paths") or []
        if vp:
            tops = Counter(p.split("/")[1].strip() for p in vp if p.count("/") >= 1)
            rec["ads_category"] = tops.most_common(1)[0][0]
            rec["ads_path"] = sorted(vp, key=len)[0]
            rec["ads_n_paths"] = len(vp)
        reg[key] = rec
        pool[key] = {"axis": "topic", "label": t, "query": t, "system": "AdsTopics",
                     "season_led": rec.get("season_led", False), "final": 1}
    json.dump(pool, open("analysis/_topics.json", "w"), indent=1)
    json.dump(reg, open("analysis/_topics_weekly.json", "w"), indent=0)
    print(f"[registry --overlay-attn] {len(reg)} recs ({n_att} attention / {n_fb} anchor-fallback)")


def stage_overlay():
    done = json.load(open(FITS))
    atlas = pd.read_csv("analysis/adstopics/atlas_topics.csv").set_index("topic")
    tf = _load("analysis/trends_fit.py", "tf")
    pool, reg = {}, {}
    n_atlas = 0
    for key, rec in done.items():
        t = rec["label"]
        if t in atlas.index:
            a = atlas.loc[t]
            rec["class_model"] = "gauss-gd"
            rec["sign"] = str(a["sign"])
            rec["gauss_phase"] = float(a["phase"])
            rec["gauss_lamps"] = str(a["lamps"])
            rec["r2_val"] = float(a["r2_val"]); rec["r2_test"] = float(a["r2_test"])
            rec["season_led"] = bool(a["season_led"]); rec["tier"] = str(a["tier"])
            n_atlas += 1
        reg[key] = rec
        pool[key] = {"axis": "topic", "label": t, "query": t, "system": "AdsTopics",
                     "season_led": rec.get("season_led", False), "final": 1}
    json.dump(pool, open("analysis/_topics.json", "w"), indent=1)
    json.dump(reg, open("analysis/_topics_weekly.json", "w"), indent=0)
    print(f"[registry --overlay] {len(reg)} recs ({n_atlas} atlas-classified) → _topics.json/_topics_weekly.json")


if __name__ == "__main__":
    if "--fit" in sys.argv:
        w = int(sys.argv[sys.argv.index("--fit") + 1]) if len(sys.argv) > sys.argv.index("--fit") + 1 else 8
        stage_fit(w)
    elif "--overlay-ts" in sys.argv:
        stage_overlay_ts()
    elif "--overlay-attn" in sys.argv:
        stage_overlay_attn()
    elif "--overlay" in sys.argv:
        stage_overlay()
    else:
        print(__doc__)
