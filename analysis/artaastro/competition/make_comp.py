#!/usr/bin/env python3
"""make_comp.py — build the "artaastro" competition from the cleaned GDELT daily/global dataset + the
12-body sidereal ephemeris.

TASK. Predict each of 6 GLOBAL DAILY conflict/cooperation measures from the sidereal (Lahiri) sky.
Features: the 12 sidereal ecliptic longitudes INCLUDING the Moon (degrees).

ANTI-CHEAT = just DE-TIME + SHUFFLE. No date column; rows shuffled; the test `id` is anonymised and
its order is randomised (id independent of date). The target derives from public GDELT, so re-deriving
the date from the sky to look it up is disallowed (rule) — predict from the provided sky only.

METRIC = plain R2 (the platform `r2` scorer): per measure, R2 over the holdout; averaged over the 6.

BASELINE = the single global-phase model (reference_solution.py):
    y_hat = sum_i w_i * sinc(f_i*(x_i - p)) + b   (12 freq f_i, 12 weight w_i, 1 global phase p, bias b),
fit by gradient descent. Its fitted p (-> sidereal sign) and f_i per measure are the reported result.

WINDOWS. Train = clean days 2015-2019 (the modern GDELT regime). Holdout = 2020-01-01 -> present.

EMITS wp-content/plugins/aquest/data/competitions/artaastro/:
    train.zip · test.csv.gz · sample_submission.csv.gz · solution.php (GUARDED) · stats.json ·
    reference_solution.py · starter.ipynb.gz · .htaccess
plus out/artaastro_holdout.csv (private id->date->target).

Run (after clean_data.py + ../build.py):  python3 make_comp.py
"""
import gzip, io, json, os, shutil, sys, time, zipfile
import numpy as np, pandas as pd
import importlib.util as _u

HERE = os.path.dirname(os.path.abspath(__file__)); AST = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(AST)); sys.path.insert(0, HERE)
_rs = _u.spec_from_file_location("ref", os.path.join(HERE, "reference_solution.py"))
ref = _u.module_from_spec(_rs); _rs.loader.exec_module(ref)

SLUG   = "artaastro"
BODIES = ["sun","moon","mercury","venus","mars","jupiter","saturn","uranus","neptune","pluto","chiron","node"]
TOPICS = ["material","conflict","verbal_conf","cooperation","material_coop","violence"]
HOLDOUT_START, TRAIN_START = "2020-01-01", "2013-04-01"   # GDELT daily-feed era (homogeneous; pre-2013 backfill is a different regime). Chart shows full 1979+ history.
CLEAN = os.path.join(HERE, "world_events_daily_clean.csv")
EPH   = os.path.join(AST, "out", "ephemeris_comp.csv")
OUT   = os.path.join(REPO, "wp-content", "plugins", "aquest", "data", "competitions", SLUG)
REFSUB = os.path.join(AST, "out", "artaastro_reference_submission.csv")


def gz(name, text):
    with gzip.open(os.path.join(OUT, name + ".gz"), "wt", newline="") as f:
        f.write(text)


def r2_by_topic(truth, preds):
    per = {}
    for topic, ids in truth.items():
        keys = list(ids); ys = np.array([ids[k] for k in keys], float)
        mean = ys.mean(); sst = float(np.sum((ys - mean) ** 2))
        if sst <= 1e-12: continue
        pm = preds.get(topic, {}); pr = np.array([pm.get(k, mean) for k in keys], float)
        per[topic] = 1.0 - float(np.sum((ys - pr) ** 2)) / sst
    return (float(np.mean(list(per.values()))) if per else None), per


def main():
    t0 = time.time(); rng = np.random.default_rng(20200225)
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(CLEAN, parse_dates=["date"]).merge(
         pd.read_csv(EPH, parse_dates=["date"]), on="date", how="inner").sort_values("date").reset_index(drop=True)
    dates = df["date"]
    sky = np.round(df[BODIES].to_numpy(float), 2)                       # TRUE sidereal longitudes (deg, 2dp)
    train_idx = np.where(((dates >= TRAIN_START) & (dates < HOLDOUT_START)).to_numpy())[0]
    test_idx  = np.where((dates >= HOLDOUT_START).to_numpy())[0]
    print(f"[train {len(train_idx)} days · holdout {len(test_idx)} days · {len(TOPICS)} measures · metric r2]")

    # ── TRAIN: per-measure CSV (12 true longitudes + target), rows shuffled ──
    zbuf = io.BytesIO(); n_train = 0
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for ti, topic in enumerate(TOPICS, 1):
            X = sky[train_idx]; y = df[topic].to_numpy(float)[train_idx]
            arr = np.hstack([X, y.reshape(-1, 1)])
            np.random.default_rng(1500 + ti).shuffle(arr, axis=0)
            buf = io.StringIO(); buf.write(",".join(BODIES + ["target"]) + "\n")
            np.savetxt(buf, arr, fmt=["%.2f"] * len(BODIES) + ["%.6f"], delimiter=",")
            z.writestr(f"train/{topic}.csv", buf.getvalue()); n_train += arr.shape[0]
    open(os.path.join(OUT, "train.zip"), "wb").write(zbuf.getvalue())

    # ── TEST: de-timed (anonymised, shuffled ids) + shuffled rows ──
    solution = {}; holdout = []; test_lines = []; samp_lines = []
    next_id = 1
    for topic in TOPICS:
        yv = df[topic].to_numpy(float)
        ids = list(range(next_id, next_id + len(test_idx))); next_id += len(test_idx)
        order = rng.permutation(len(test_idx))                          # id independent of date
        for j, g in enumerate(test_idx):
            rid = ids[order[j]]; tgt = round(float(yv[g]), 6)
            solution[f"{topic}|{rid}"] = tgt
            holdout.append((topic, rid, dates.iloc[g].strftime("%Y-%m-%d"), tgt))
            test_lines.append(f"{topic},{rid}," + ",".join(f"{v:.2f}" for v in sky[g]))
            samp_lines.append(f"{topic},{rid},0")
    perm = rng.permutation(len(test_lines))
    gz("test.csv", ",".join(["trend","id"] + BODIES) + "\n" + "\n".join(test_lines[i] for i in perm) + "\n")
    gz("sample_submission.csv", "trend,id,target\n" + "\n".join(samp_lines[i] for i in perm) + "\n")
    n_test = len(test_lines)
    pd.DataFrame(holdout, columns=["topic","id","date","target"]).to_csv(os.path.join(AST,"out","artaastro_holdout.csv"), index=False)

    with open(os.path.join(OUT, "solution.php"), "w") as f:
        f.write("<?php\n// artaastro hidden holdout — server-only. Guarded: direct HTTP execution 404s.\n"
                "// { \"trend|id\" => target }: each measure's 2020-> global-daily value.\n"
                "if ( ! defined( 'AQ_COMP_SOLUTION' ) ) { http_response_code( 404 ); exit; }\n"
                "return [\n" + "\n".join(f"\t'{k}' => {v}," for k, v in solution.items()) + "\n];\n")
    open(os.path.join(OUT, ".htaccess"), "w").write('<FilesMatch "\\.(php|json)$">\n  Require all denied\n</FilesMatch>\n')

    # ── BASELINE: the global-phase gradient-descent model → submission + board + phases.json ──
    truth = {}
    for k, v in solution.items():
        tp, rid = k.split("|"); truth.setdefault(tp, {})[int(rid)] = float(v)
    hd = pd.DataFrame(holdout, columns=["topic","id","date","target"])
    ref_lines = ["trend,id,target"]; ref_preds = {}; report = {}
    date_i = {d: i for i, d in enumerate(dates.dt.strftime("%Y-%m-%d"))}
    recent = (dates.iloc[train_idx].to_numpy() >= np.datetime64("2015-01-01"))   # recent era → the LEVEL
    for topic in TOPICS:
        ytr = df[topic].to_numpy(float)[train_idx]
        fit = ref.fit_measure(sky[train_idx], ytr)                               # PATTERN from the FULL history
        # the sky modulation (bias 0): the validated pattern, else flat. Then LEVEL-ANCHOR to the recent era
        # (pre-2013 GDELT has different share levels, so the full-history mean is the wrong level for 2020+).
        raw = {"p": fit["p"], "f": fit["f"], "w": fit["w_raw"], "b": 0.0}   # the actual fit (shown + scored)
        pat = ref.predict(raw, sky)
        level = float(ytr[recent].mean() - pat[train_idx][recent].mean())
        lo, hi = float(np.percentile(ytr[recent], 1)), float(np.percentile(ytr[recent], 99))
        allpred = np.clip(pat + level, lo, hi)
        kt = hd[hd.topic == topic]; pm = {}
        for rid, d in zip(kt.id.to_numpy(), kt.date):
            p = float(allpred[date_i[d]]); ref_lines.append(f"{topic},{int(rid)},{round(p,6)}"); pm[int(rid)] = p
        ref_preds[topic] = pm
        # in-sample R2 on the recent era (where the level is anchored) + monthly actual-vs-fit SERIES for the
        # /astro chart (the model fit ON the data, like the Topics analysis pages).
        pin = allpred[train_idx]; ssr = np.sum((ytr[recent] - ytr[recent].mean())**2)
        in_r2 = 1 - np.sum((ytr[recent] - pin[recent])**2)/ssr if ssr > 0 else float("nan")
        md = pd.DataFrame({"m": dates.dt.strftime("%Y-%m"), "a": df[topic].to_numpy(float), "f": allpred})
        mo = md.groupby("m", sort=True).mean()
        months = mo.index.tolist(); test_from = int(np.searchsorted(np.array(months), "2020-01"))
        report[topic] = {"phase": round(fit["p"], 1), "sign": fit["sign"], "validated": fit["validated"],
                         "val_r2": round(fit["val_r2"], 4), "in_r2": round(float(in_r2), 4),
                         "series": {"months": months, "test_from": test_from,
                                    "actual": [round(float(v), 5) for v in mo["a"]],
                                    "fit": [round(float(v), 5) for v in mo["f"]]},
                         "freqs": [round(float(v), 3) for v in fit["f"]],
                         "weights": [round(float(v), 5) for v in fit["w_raw"]], "bodies": BODIES}
    open(REFSUB, "w").write("\n".join(ref_lines) + "\n")
    board, per = r2_by_topic(truth, ref_preds)
    for topic in TOPICS: report[topic]["holdout_r2"] = round(per.get(topic, float("nan")), 4)
    phases = {"measures": TOPICS, "bodies": BODIES,
              "model": "y = sum_i w_i sinc(f_i (x_i - p)) + b ; one global phase p per measure, 12 sign-centre restarts, time-validated",
              "phases": report}
    json.dump(phases, open(os.path.join(AST, "out", "artaastro_phases.json"), "w"), indent=1)
    json.dump(phases, open(os.path.join(REPO, "artaquest-web", "src", "data", "artaastro-phases.json"), "w"), indent=1)
    print(f"  global-phase baseline (mean R2) = {board:+.4f}")
    for topic in TOPICS: print(f"    {topic:14} R2={per.get(topic, float('nan')):+.4f}")

    stats = {"slug": SLUG, "n_topics": len(TOPICS), "n_train": n_train, "n_test": n_test,
             "n_features": len(BODIES), "n_targets": len(TOPICS), "metric": "r2",
             "target": "global daily conflict/cooperation measure (within-day share)",
             "features": "12 sidereal ecliptic longitudes incl. the Moon (deg); de-timed + shuffled",
             "score": "mean over the 6 measures of the plain R2 over its holdout (0 = predict-the-mean)",
             "holdout": f"{dates[test_idx[0]].date()} -> {dates[test_idx[-1]].date()} ({len(test_idx)} days), "
                        f"de-timed + shuffled; train {TRAIN_START} -> 2019-12-31",
             "topics": TOPICS,
             "baseline": {"model": "global-phase sinc: sum_i w_i sinc(f_i(x_i-p))+b, gradient descent (reference_solution.py)",
                          "expected_public_score": round(board, 4), "expected_full_score": round(board, 4)},
             "ground_truth": "GDELT 1.0 global daily aggregates, entire history (1979->present), CC-BY"}
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=2)
    shutil.copy(os.path.join(HERE, "reference_solution.py"), os.path.join(OUT, "reference_solution.py"))

    base = f"https://artaquest.org/wp-content/uploads/competitions/{SLUG}"
    def c(s): return {"cell_type":"code","metadata":{},"outputs":[],"execution_count":None,"source":s.splitlines(keepends=True)}
    def m(s): return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}
    nb = {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},"language_info":{"name":"python"}},
          "cells":[
        m("# ArtaAstro — does the sky forecast world events?\n\n"
          "Predict 6 global daily conflict measures (from GDELT) from the 12 sidereal longitudes **incl. the "
          "Moon**. Data is de-timed and shuffled. **Metric: plain R² per measure, averaged** (0 = each "
          "measure's mean). The reference baseline is a single **global-phase** model, "
          "`ŷ = Σ wᵢ·sinc(fᵢ(xᵢ−p)) + b`, fit by gradient descent — it reports each measure's phase `p` "
          "(→ a sidereal sign) and per-body frequencies `fᵢ`. Beat it. Predict `trend,id,target`.\n\n"
          f"Competition: https://artaquest.org/competition/?slug={SLUG}\n"),
        c(f"import pandas as pd, io, zipfile, urllib.request\n"
          f"zf=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('{base}/train.zip',timeout=300).read()))\n"
          "train={n.split('/')[-1][:-4]:pd.read_csv(zf.open(n)) for n in zf.namelist() if n.endswith('.csv')}\n"
          f"test=pd.read_csv('{base}/test.csv')\nprint(len(train),'measures ·',len(test),'test rows')"),
        m("## Submit\nUpload `submission.csv` (`trend,id,target`) on the **Submit** tab.\n"),
    ]}
    gz("starter.ipynb", json.dumps(nb, indent=1))

    def sz(p): return os.path.getsize(p) if os.path.isfile(p) else 0
    sizes = {f: sz(os.path.join(OUT, f)) for f in ["train.zip","test.csv.gz","sample_submission.csv.gz","solution.php","stats.json"]}
    print(f"\nmeasures {len(TOPICS)} · train {n_train:,} · test {n_test:,}")
    print("bundle: " + " · ".join(f"{k} {v/1e6:.2f}MB" for k, v in sizes.items()) + f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
