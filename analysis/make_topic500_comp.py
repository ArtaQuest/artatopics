#!/usr/bin/env python3
"""Build the "topic-500" competition dataset — predict a learning topic's worldwide search interest from
the sky, with ONE fixed model class (a global-phase sinc, fit by gradient descent) as the baseline
(operator directive 2026-07-07, the FINAL simplification: NO phase permutation).

THE DESIGN.
  * DATA — each topic's MONTHLY rows, JUST SHUFFLED. Columns = the 11 body phases (sun,mercury,venus,mars,
    jupiter,saturn,uranus,neptune,pluto,chiron,node — sidereal ecliptic longitude 0-360 deg) + `target`
    (0-100 worldwide Google-Trends interest). NO id, NO phase column, NO 72x permutation.
      TRAIN  ~197 months x 500 topics (train.zip, one CSV per topic: the 11 phases + target, rows shuffled)
             the months 2004-01 .. ~2020-05 (the FIRST 80% of the usable window).
      TEST   ~49 months x 500 topics   (test.csv: trend + the 11 phases) — each topic's FUTURE holdout, the
             last 20% of the usable window ~2020-06 .. 2024-06. RECENCY-GUARDED: the most recent clean year
             2024-07 .. 2025-06 is EXCLUDED from the test, and the provisional final year 2025-07 .. 2026-06
             is dropped from the whole set (operator 2026-07-07). The exact indices come from split_index().
      Submit trend + the 11 body phases + target (one value per test row) — the sample_submission shape.
  * MODEL (the baseline) — y = SUM_i w_i * sinc(f_i * (x_i - p)) + b, ONE global phase p, per-body frequency
    f_i, per-body weight w_i, bias b (topic500_reference_solution.py — PYTORCH, one batched GPU-accelerated
    pass, 2026-07-10). Fit by 12 gradient starts, one at each 30 deg sign centre; recency-weighted loss +
    time-blocked validation select the winner (mode="forecast"); a train-objective "atlas" fit on the same
    train months gives the descriptive phase/sign/frequency read-out and the plotted fit line. Imported here
    so the HOSTED reference == the board's baseline entry (both are order-invariant: canonical slow-sky sort).
  * SCORE — a 0-100 board of PLAIN R^2: the scorer folds each submission row's 11 phases to their sky
    fingerprint (each phase minus the Sun's, mod 360 — a unique month key), keys by topic -> the true target,
    per-topic R^2 clamped [0,1], averaged over the 500 topics x100. The model class is fixed; the competition
    is to IMPROVE THE LEARNING ALGORITHM. The topics page reports each topic's fitted PHASE p (-> sign) and its
    per-body FREQUENCIES f_i.

EMITS the deployable plugin bundle wp-content/plugins/aquest/data/competitions/topic-500/:
    train.zip                 500 per-topic CSVs (11 body phases + target — the shuffled train rows)
    test.csv.gz               6,000 rows: trend + the 11 body phases (NO target)
    sample_submission.csv.gz  trend + the 11 body phases + target(=0)
    stats.json · starter.ipynb.gz · reference_solution.py
    solution.php              GUARDED { "topic|fingerprint" => target } — the 12 holdout months per topic,
                              keyed by the sky fingerprint (every phase minus the Sun's, mod 360)
    split.php                 GUARDED public/private halves per topic (fingerprint keys)
    series.php                GUARDED plot bundle: full monthly actuals + the baseline's fit + the 12 test
                              months' fingerprints + each topic's fitted PHASE/SIGN + per-body FREQUENCIES -> Results tab
plus the private topic -> label map to ~/.artaquest-dev/topic500_meta.json (NEVER the repo/uploads) and the
REFERENCE submission (6,000 rows, trend + 11 phases + target) to ~/.artaquest-dev/topic500_reference_submission.csv.

Run:  python3 analysis/make_topic500_comp.py
"""
import gzip, io, json, math, os, random, sys, time, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import importlib.util as _u
_spec = _u.spec_from_file_location("tf", os.path.join(HERE, "trends_fit.py"))
tf = _u.module_from_spec(_spec); _spec.loader.exec_module(tf)
_rspec = _u.spec_from_file_location("ref500", os.path.join(HERE, "topic500_reference_solution.py"))
ref500 = _u.module_from_spec(_rspec); _rspec.loader.exec_module(ref500)

SEED = 1500
N_TOPICS = 500
RECENCY_DROP = 12               # DROP the provisional, recency-biased final year from the WHOLE dataset
# TEST split (operator 2026-07-07, RECENCY-GUARDED): ref500.split_index(clean_end) returns (split, test_end)
# with test_end = clean_end - 12: train = [0, split) (first 80%), TEST = [split, test_end) (last 20% of the
# usable window, a FUTURE holdout), and the last clean year [test_end, clean_end) is EXCLUDED from the test.
POOL = "analysis/_topics_weekly.json"
def _out(slug): return os.path.join(REPO, "wp-content", "plugins", "aquest", "data", "competitions", slug)
BODIES = list(tf.BODIES)        # sun, mercury, …, node  (11; index 0 is the Sun — the fingerprint anchor)
SIGNS = list(tf.SIGNS)
NB = len(BODIES)
SUN_I = BODIES.index("sun")     # 0 — the fingerprint anchor


def fingerprint(sky_row):
    """The sky FINGERPRINT of a month: every body's phase MINUS the Sun's, mod 360 (d0 = 0). Unique per
       clean month, and IDENTICAL to the PHP scorer's sky_fingerprint() — the scorer's month key."""
    s = int(sky_row[SUN_I])
    return tuple(int((int(v) - s) % 360) for v in sky_row)


def fp_str(sky_row):
    return ",".join(str(x) for x in fingerprint(sky_row))


def load_series(label):
    """A topic's monthly series on the FULL tf.GRID (NO DROP_LAST trim — the last 12 CLEAN months are this
       dataset's test targets). Returns a float array of len(GRID) or None."""
    p = f"{tf.DATA_DIR}/{tf.slug(label)}.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p); df["Time"] = pd.to_datetime(df["Time"])
    ser = df.drop_duplicates("Time").set_index("Time")["v"].reindex(pd.DatetimeIndex(tf.GRID))
    ys = pd.to_numeric(ser, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if ys.notna().sum() < len(tf.GRID) * 0.9:
        return None
    y = ys.interpolate(limit_direction="both").ffill().bfill().to_numpy(float)
    return y if np.isfinite(y).all() else None


# The 6 ArtaAstro WORLD-EVENT measures (GDELT global-daily shares, unified into the monthly atlas by
# build_world_measures.py). The artaastro competition predicts THESE from the sky — same model, same board.
MEASURES = ["Material conflict", "Any conflict", "Verbal conflict", "Cooperation", "Material cooperation", "Violence"]

# The two competitions this ONE builder emits (operator 2026-07-07). BOTH use the constrained sinc-GD model,
# the last-20% FUTURE split and the phase-r2 sky-fingerprint board — an identical emit path, so the shared
# PHP scorer treats them the same. Pick the dataset with AQ_COMP (default topic-500).
CFGS = {
    "topic-500": {"slug": "topic-500", "meta": "~/.artaquest-dev/topic500_meta.json",
                  "ref_sub": "~/.artaquest-dev/topic500_reference_submission.csv",
                  "kind": "topic", "kinds": "the 500 most-searched learning topics on Earth",
                  "target": "google-trends monthly interest (0-100 int)"},
    "artaastro": {"slug": "artaastro", "meta": "~/.artaquest-dev/artaastro_meta.json",
                  "ref_sub": "~/.artaquest-dev/artaastro_reference_submission.csv",
                  "kind": "measure", "kinds": "6 global-daily world-event measures (GDELT: conflict, "
                  "cooperation, violence — monthly within-day shares as a percent)",
                  "target": "global world-event share (percent, monthly mean)"},
}


def select_topic500(pool, n_grid):
    """The TOP N_TOPICS learning topics by popularity (desc) → [{key,label,pop,y}]."""
    ranked = sorted(pool.items(), key=lambda kv: (-float(kv[1].get("popularity") or 0.0), kv[0]))
    items, skipped = [], []
    for key, rec in ranked:
        if len(items) >= N_TOPICS:
            break
        label = rec.get("label", key)
        y = load_series(label)
        if y is None or len(y) != n_grid:
            skipped.append(key); continue
        items.append({"key": key, "label": label, "pop": float(rec.get("popularity") or 0.0), "y": y})
    if len(items) < N_TOPICS:
        sys.exit(f"only {len(items)} usable topics ({len(skipped)} skipped: {skipped[:10]}…) — need {N_TOPICS}")
    return items


def select_measures(n_grid):
    """The 6 fixed world-event measures → [{key,label,pop,y}] (build_world_measures.py wrote their CSVs)."""
    items = []
    for label in MEASURES:
        y = load_series(label)
        if y is None or len(y) != n_grid:
            sys.exit(f"missing measure series: {label} — run analysis/build_world_measures.py first")
        items.append({"key": tf.slug(label), "label": label, "pop": 0.0, "y": y})
    return items


def main():
    t0 = time.time()
    which = os.environ.get("AQ_COMP", "topic-500")
    cfg = CFGS[which]
    SLUG = cfg["slug"]; OUT = _out(SLUG)
    META = os.path.expanduser(cfg["meta"]); REF_SUB = os.path.expanduser(cfg["ref_sub"])
    rng = random.Random(SEED)
    pool = json.load(open(POOL))
    lon = tf.ephemeris()
    grid = pd.DatetimeIndex(tf.GRID)
    n_grid = len(grid)
    sky_int = np.stack([np.round(np.asarray(lon[b], float)).astype(int) % 360 for b in BODIES], axis=1)  # n×11 int

    # ── the dataset, anonymised in a SHUFFLED order (topic-NNN — the real label stays private, anti-join) ──
    topics = select_measures(n_grid) if which == "artaastro" else select_topic500(pool, n_grid)
    order = list(range(len(topics))); rng.shuffle(order)
    topics = [topics[i] for i in order]
    for i, t in enumerate(topics):
        t["topic"] = f"topic-{i + 1:03d}"

    # ── the WINDOWS (operator 2026-07-07, RECENCY-GUARDED): drop the raw recency year, then EXCLUDE the last
    #    clean year from the TEST too. train = [0, split) · TEST = [split, test_end) · [test_end, clean_end) guard-dropped ──
    clean_end = n_grid - RECENCY_DROP            # clean series used at all: [0, clean_end)  (.. 2025-06)
    split, test_end = ref500.split_index(clean_end)   # train [0, split) 80% · TEST [split, test_end) last 20% · [test_end, clean_end) guard-dropped
    TEST_MONTHS = test_end - split               # the recency-guarded future-holdout month count
    sky_clean = sky_int[:test_end]               # the in-competition span (the guard year is excluded)
    print(f"[{len(topics)} {cfg['kind']}s · grid {grid[0].date()}…{grid[-1].date()} ({n_grid} mo)"
          f" · DROP recency {grid[clean_end].date()}…{grid[-1].date()} ({RECENCY_DROP})"
          f" · train {grid[0].date()}…{grid[split-1].date()} ({split})"
          f" + FUTURE test {grid[split].date()}…{grid[test_end-1].date()} ({TEST_MONTHS})"
          f" · GUARD-drop {grid[test_end].date()}…{grid[clean_end-1].date()} ({clean_end-test_end})]")

    # The test months' fingerprints are shared across ALL topics (same sky); check they are UNIQUE per month —
    # a collision would make the scorer's per-topic month key ambiguous.
    test_months = list(range(split, test_end))
    test_fps = [fp_str(sky_int[g]) for g in test_months]
    if len(set(test_fps)) != len(test_fps):
        sys.exit("test-month fingerprint collision — the scorer key would be ambiguous")
    print(f"decoder check: {len(set(test_fps))}/{TEST_MONTHS} test-month fingerprints unique")

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(META), exist_ok=True)
    train_months = np.arange(split)

    # ── per-topic train CSVs (11 bodies + target, 246 shuffled rows) → train.zip ──
    print("building train.zip…", flush=True)
    hdr_train = BODIES + ["target"]
    solution = {}                    # canonical map "topic|fingerprint" -> target (the 12 holdout months per topic)
    zbuf = io.BytesIO()
    n_train = 0
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for ti, t in enumerate(topics, 1):
            y_int = np.clip(np.round(t["y"]), 0, 100).astype(int)
            arr = np.column_stack([sky_int[train_months, :], y_int[train_months]]).astype(np.int32)
            np.random.default_rng(SEED + ti).shuffle(arr, axis=0)              # shuffle rows in-file (de-time)
            buf = io.StringIO(); buf.write(",".join(hdr_train) + "\n")
            np.savetxt(buf, arr, fmt="%d", delimiter=",")
            z.writestr(f"train/{t['topic']}.csv", buf.getvalue())
            n_train += arr.shape[0]
            keys = []                                                          # the 12 test months' fingerprints (month order)
            for g in test_months:
                fp = test_fps[g - split]
                keys.append(fp)
                solution[f"{t['topic']}|{fp}"] = int(y_int[g])
            t["keys"] = keys
    with open(os.path.join(OUT, "train.zip"), "wb") as f:
        f.write(zbuf.getvalue())

    def gz(name, text):
        with gzip.open(os.path.join(OUT, name + ".gz"), "wt", newline="") as f:
            f.write(text)

    # ── test.csv + sample_submission.csv (trend + 11 body phases, 12 months per topic) ──
    print("writing test.csv + sample_submission.csv…", flush=True)
    tl = ["trend," + ",".join(BODIES)]
    sl = ["trend," + ",".join(BODIES) + ",target"]
    for t in topics:
        for g in test_months:
            body_str = ",".join(str(int(v)) for v in sky_int[g])
            tl.append(t["topic"] + "," + body_str)
            sl.append(t["topic"] + "," + body_str + ",0")
    gz("test.csv", "\n".join(tl) + "\n")
    gz("sample_submission.csv", "\n".join(sl) + "\n")
    n_test_rows = len(tl) - 1
    del tl, sl

    # ── the GUARDED holdout base map { "topic|fingerprint" => target } ──
    with open(os.path.join(OUT, "solution.php"), "w") as f:
        f.write("<?php\n// topic-500 hidden holdout — server-only. Guarded: direct HTTP execution 404s.\n"
                "// Map { \"topic|fingerprint\" => target }: each topic's 12 future-month search-interest targets,\n"
                "// keyed by the sky fingerprint (every phase minus the Sun's, mod 360) — the scorer folds each\n"
                "// submission row the same way.\n"
                "if ( ! defined( 'AQ_COMP_SOLUTION' ) ) { http_response_code( 404 ); exit; }\n"
                "return [\n" + "\n".join(f"\t'{k}' => {v}," for k, v in solution.items()) + "\n];\n")

    # ── fit the sinc-GD BASELINE (torch, ONE batched pass — GPU-accelerated where available) in BOTH
    #    modes: "forecast" (recency-weighted, val-selected — the board's baseline entry: the reference
    #    submission + the per-topic holdout R²) and "atlas" on the SAME train months (the descriptive
    #    train-objective read-out: phase/sign/frequencies + the plotted fit line). Both see ONLY the
    #    train months — the fit line the results tab shows must never encode the (private) test targets. ──
    print(f"fitting the sinc-GD baseline (torch, batched, device {ref500._device()})…", flush=True)
    Y_train = [np.clip(np.round(t["y"][:split]), 0, 100).astype(float) for t in topics]
    X_train = sky_clean[:split].astype(float)
    par_fc = ref500.fit_many(Y_train, X_train, NB, mode="forecast", progress=True)
    print(f"  forecast fit done · {time.time()-t0:.0f}s — atlas fit…", flush=True)
    par_at = ref500.fit_many(Y_train, X_train, NB, mode="atlas", progress=True)
    print(f"  atlas fit done · {time.time()-t0:.0f}s", flush=True)

    ref_rows = []                    # the reference submission (baseline preds on the test months, all topics)
    baseline_clamps = []             # per-topic test R² clamped [0,1] — the board mirror + the split stratifier
    for i, t in enumerate(topics):
        t["par"] = par_fc[i]
        t["phase"] = ref500.phase_of(par_at[i], NB)     # descriptive (atlas-objective) read-out
        t["sign"] = SIGNS[int(t["phase"] // 30) % 12]
        t["freqs"] = [round(float(v), 4) for v in ref500.freqs_of(par_at[i], NB)]
        # the plotted fit LINE (atlas objective, TRAIN months only, evaluated over the plot window)
        t["fit_clean"] = np.clip(ref500.predict(par_at[i], sky_clean, NB), 0.0, 100.0)
        # the test-month predictions (forecast mode — the reference submission + the board mirror)
        pred_te = np.clip(ref500.predict(par_fc[i], sky_int[np.array(test_months)], NB), 0.0, 100.0)
        yte = np.clip(np.round(t["y"][split:test_end]), 0, 100).astype(float)
        sst = float(np.sum((yte - yte.mean()) ** 2))
        r2 = 1.0 - float(np.sum((yte - pred_te) ** 2)) / sst if sst > 1e-9 else 0.0
        t["r2"] = r2
        baseline_clamps.append(min(1.0, max(0.0, r2)))
        for j, g in enumerate(test_months):
            ref_rows.append((t["topic"], sky_int[g], round(float(pred_te[j]), 2)))

    baseline_board = float(np.mean(baseline_clamps) * 100.0)

    # ── PUBLIC/PRIVATE split PER TOPIC (250 public / 250 private), STRATIFIED by the baseline's per-topic
    #    clamped R² so both halves mirror the full board. Every one of a public topic's 12 months is public;
    #    a private topic's are all private (rank on public, settle on private). ──
    order_topics = sorted(range(len(topics)), key=lambda i: (-baseline_clamps[i], topics[i]["topic"]))
    pub, prv = [], []
    for rank, i in enumerate(order_topics):
        keys = [f"{topics[i]['topic']}|{fp}" for fp in topics[i]["keys"]]
        (pub if rank % 2 == 0 else prv).extend(keys)
    with open(os.path.join(OUT, "split.php"), "w") as f:
        f.write("<?php\n// topic-500 public/private holdout split — server-only, guarded (404s over HTTP).\n"
                "// Keys are \"topic|fingerprint\"; the 500 topics split 250 public / 250 private (a topic's 12\n"
                "// months are ALL in the same half — rank on public topics, settle on private).\n"
                "if ( ! defined( 'AQ_COMP_SOLUTION' ) ) { http_response_code( 404 ); exit; }\n"
                "return [\n"
                "\t'public'  => [ " + ", ".join(f"'{k}'" for k in sorted(pub)) + " ],\n"
                "\t'private' => [ " + ", ".join(f"'{k}'" for k in sorted(prv)) + " ],\n"
                "];\n")

    # sign-distribution GATE stats over the 500 competition topics (the fitted phase -> sign)
    sign_hist = {}
    for t in topics:
        sign_hist[t["sign"]] = sign_hist.get(t["sign"], 0) + 1
    populated = sum(1 for s in SIGNS if sign_hist.get(s, 0) > 0)
    nt = len(topics)
    ent = -sum((c / nt) * math.log2(c / nt + 1e-12) for c in sign_hist.values())
    print(f"\nBASELINE (sinc-GD) plain-R² board — FULL(500) {baseline_board:.4f}/100")
    print(f"fitted-phase sign distribution: {populated}/12 signs · entropy {ent:.3f}/3.585 bits")
    print("  " + " ".join(f"{s[:3]}:{sign_hist.get(s,0)}" for s in SIGNS))

    # ── series.php — the GUARDED plot bundle (actuals + baseline fit + the fitted phase/sign/frequencies) ──
    print("writing series.php…", flush=True)
    months_str = ",".join(d.strftime("%Y-%m") for d in grid[:test_end])
    with open(os.path.join(OUT, "series.php"), "w") as f:
        f.write("<?php\n// topic-500 plot bundle — server-only, guarded (holds the TEST-month actuals; the\n"
                "// results endpoint reveals only the public half while active). Per topic: y = the CLEAN monthly\n"
                "// actuals (csv ints, recency year dropped), fit = the sinc-GD BASELINE's atlas-objective fit\n"
                "// (torch; fit on the TRAIN months ONLY — it must never encode the private test targets —\n"
                "// evaluated over the plot window; csv, 1dp), keys = the test months' sky fingerprints (month\n"
                "// order), sign/peak = the fitted global phase -> its zodiac sign, freqs = the per-body\n"
                "// frequencies f_i (csv, body order sun..node), r2 = the FORECAST baseline's holdout R^2.\n"
                "if ( ! defined( 'AQ_COMP_SOLUTION' ) ) { http_response_code( 404 ); exit; }\n"
                "return [\n"
                f"\t'months' => '{months_str}',\n"
                f"\t'test_from' => {split},\n"
                "\t'topics' => [\n")
        for t in topics:
            tp = t["topic"]
            y_int = np.clip(np.round(t["y"][:test_end]), 0, 100).astype(int)
            y_str = ",".join(str(int(v)) for v in y_int)
            f_str = ",".join(f"{v:.1f}" for v in t["fit_clean"])
            keys_str = ";".join(t["keys"])
            freqs_str = ",".join(f"{v:.4f}" for v in t["freqs"])
            f.write(f"\t\t'{tp}' => [ 'y' => '{y_str}', 'fit' => '{f_str}', 'keys' => '{keys_str}', "
                    f"'sign' => '{t['sign']}', 'peak' => {int(round(t['phase']))}, "
                    f"'freqs' => '{freqs_str}', 'r2' => {round(float(t['r2']), 3)} ],\n")
        f.write("\t],\n];\n")

    # the reference submission (6,000 rows) → ~/.artaquest-dev (NEVER committed)
    print("writing the reference submission…", flush=True)
    with open(REF_SUB, "w") as f:
        f.write("trend," + ",".join(BODIES) + ",target\n")
        for topic, row, v in ref_rows:
            f.write(topic + "," + ",".join(str(int(x)) for x in row) + f",{v}\n")

    # ── stats.json ─
    stats = {"slug": SLUG, "n_topics": len(topics), "n_train": n_train, "n_test": n_test_rows,
             "n_unique_test": len(topics) * TEST_MONTHS, "n_public": len(pub), "n_private": len(prv),
             "n_features": len(BODIES), "n_targets": len(topics), "seed": SEED,
             "layout": f"train.zip: {len(topics)} per-{cfg['kind']} CSVs (the 11 body phases + target; {split} "
                       "months, rows shuffled - NO id, NO phase, NO permutation); test.csv = each "
                       f"{cfg['kind']}'s recency-guarded last-20% FUTURE months ({TEST_MONTHS}; trend + 11 phases) - predict the target for every row",
             "target": cfg["target"],
             "features": "11 sidereal ecliptic phases (0-360 int deg): " + ", ".join(BODIES),
             "score": "0-100 plain-R^2 board: fold each row's 11 phases to their sky fingerprint (each phase "
                      f"minus the Sun's, mod 360), key by {cfg['kind']} -> the true target; per {cfg['kind']} R^2 "
                      "over its future holdout months clamped [0,1]; averaged x100",
             "model": "y = sum_i w_i * sinc(f_i * (x_i - p)) + b — one global phase p (-> the zodiac sign), "
                      "per-body frequency f_i (in [0,20]), weight w_i (>=0, NON-NEGATIVE), bias b; fit by 12 gradient "
                      "starts (one per 30 deg sign centre), keep the best. The model class is FIXED; the "
                      "competition improves the LEARNING ALGORITHM. The baseline algorithm (PyTorch, batched "
                      "full-batch Adam, GPU-accelerated): rows sorted into a canonical near-chronological order "
                      "by the slow-sky clock (order-invariant), RECENCY-WEIGHTED loss (half-life 60 months), a "
                      "time-blocked last-20% validation slice early-checkpoints each start and selects the "
                      "winner, then a brief recency-weighted refine over all train rows.",
             "holdout": f"the FUTURE - each {cfg['kind']}'s months {grid[split].date()}..{grid[test_end-1].date()} ({TEST_MONTHS}); the last clean year {grid[test_end].date()}..{grid[clean_end-1].date()} is excluded from the test (recency guard) and the final {RECENCY_DROP} months are dropped from the dataset",
             "months": f"{grid[0].date()} .. {grid[test_end-1].date()} (recency year + last-year guard dropped); train through {grid[split-1].date()}",
             "baseline": {"model": "global-phase sinc, torch gradient-descent, recency-weighted + "
                                   "time-blocked-validation selected (topic500_reference_solution.py, "
                                   "mode='forecast')",
                          "device": ref500._device(),
                          "reproducibility": "exactly deterministic per device; cross-device (cpu/mps/cuda) "
                                             "float-32 rounding can flip a topic's non-convex phase optimum "
                                             "— the board reproduces to about +-0.2 points",
                          "expected_full_score": round(baseline_board, 4)},
             "sign_distribution": {s: sign_hist.get(s, 0) for s in SIGNS},
             "sign_entropy": round(ent, 3)}
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(META, "w") as f:
        json.dump({t["topic"]: {"label": t["label"], "popularity": t["pop"]} for t in topics}, f, indent=1)

    # ── starter.ipynb ─
    base = f"https://artaquest.org/wp-content/uploads/competitions/{SLUG}"
    def code(src): return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                           "source": src.splitlines(keepends=True)}
    def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "cells": [
        md(f"# {'Topic 500' if which == 'topic-500' else 'ArtaAstro'} — the global-phase sinc starter\n\n"
           f"**{len(topics)} anonymised {cfg['kind']}s** ({cfg['kinds']}). Each {cfg['kind']}'s monthly "
           "series is given as **shuffled** rows: the **11 sidereal body phases** (0-360 deg) "
           f"and the `target`. The test set is each {cfg['kind']}'s **recency-guarded last-20% future months** "
           f"({grid[split].date()} to {grid[test_end-1].date()}) — the raw recency year AND the most recent clean "
           f"year are both held out of the test.\n\n"
           "**The model.** Predict interest with one fixed model class:\n\n"
           "$$y = \\sum_i w_i \\, \\mathrm{sinc}\\big(f_i (x_i - p)\\big) + b$$\n\n"
           "one **global phase** $p$ (its 30 deg sector is the topic's zodiac **sign**), a per-body "
           "**frequency** $f_i$, a weight $w_i$, and a bias $b$. Fit it by **12 gradient starts**, one at each "
           "30 deg **sign centre** (15,45,...,345), and keep the best — an unbiased search over the non-convex "
           "phase — with **non-negative** weights and frequencies and a **circular** phase. The model class is "
           "fixed; **the competition is to improve the learning algorithm** (better optimisation, initialisation, "
           "regularisation).\n\n"
           "**The baseline algorithm (PyTorch).** All topics are fit in ONE batched full-batch-Adam pass — "
           "**GPU-accelerated**: on a free Colab GPU (Runtime → Change runtime type → **GPU**) the full fit takes "
           "about 2 minutes (~15 min on CPU). Rows are first sorted into a canonical near-chronological order by "
           "the slow-sky clock (so the shuffled CSVs fit identically to ordered data); the loss is "
           "recency-weighted (half-life 60 months); a time-blocked last-20% validation slice selects the winning "
           "start; a brief refine then lets the bias track the recent level.\n\n"
           "**Score (0-100).** Per-topic R^2 over the last-20% FUTURE holdout months, clamped [0,1], averaged "
           "x100 (100 = every topic fit perfectly, 0 = predict-the-mean).\n\n"
           "The cell below **reproduces the leaderboard baseline** from the public data alone (exactly "
           "deterministic per device; across device types the non-convex phase can flip on a few topics, so a "
           "cross-device run matches the board to about ±0.2 points).\n\n"
           f"Competition: https://artaquest.org/competition/?slug={SLUG}\n"),
        code("# Reproduce the leaderboard baseline from PUBLIC DATA ONLY. This downloads + runs the hosted\n"
             "# reference solution, which fetches this competition's public data (train.zip + test.csv), fits the\n"
             "# ONE fixed constrained sinc-GD model with PyTorch — NON-NEGATIVE weights, frequencies in [0,20], a\n"
             "# CIRCULAR global phase p, 12 sign-centre gradient starts, recency-weighted loss, time-blocked\n"
             "# validation selection — in ONE batched pass (GPU if available), then writes submission.csv.\n"
             "# Running THIS is what produced the platform's baseline board entry.\n"
             "import subprocess, sys, urllib.request\n"
             "subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'torch', 'numpy', 'pandas'], check=True)\n"
             f"urllib.request.urlretrieve('{base}/reference_solution.py', 'reference_solution.py')\n"
             "import reference_solution as ref\n"
             "ref.main()   # downloads the public data, fits (GPU-accelerated), and writes submission.csv"),
        code("# Inspect the predictions this notebook produced — these are the exact leaderboard-baseline rows.\n"
             "import pandas as pd\n"
             "sub = pd.read_csv('submission.csv')\n"
             "print('submission.csv:', sub.shape[0], 'rows,', sub.shape[1], 'cols')\n"
             "print(sub.head(20).to_string(index=False))"),
        md("## Submit\n\nUpload `submission.csv` (`trend, <11 body phases>, target`) on the **Submit** tab — scored "
           "instantly on the public holdout half. **Ideas to beat the baseline.** The model class is fixed, so the "
           "game is the *fit*: a better global optimiser over the phase, smarter frequency initialisation, "
           "regularising the weights, or an ensemble of restarts. The holdout is the future, so guard against "
           f"overfitting the {split} training months.\n"),
    ]}
    gz("starter.ipynb", json.dumps(nb, indent=1))

    import shutil
    _refdst = os.path.join(OUT, "reference_solution.py")
    shutil.copy(os.path.join(HERE, "topic500_reference_solution.py"), _refdst)
    # Bake THIS competition's public-data BASE URL into the hosted reference solution so, run standalone, it
    # downloads the RIGHT bundle (the vendored file defaults to topic-500). This makes the hosted code
    # reproduce the board from public data alone — the reproducibility requirement (operator 2026-07-07).
    if SLUG != "topic-500":
        _rt = open(_refdst).read().replace('competitions/topic-500', 'competitions/' + SLUG)
        open(_refdst, "w").write(_rt)

    def sz(p): return os.path.getsize(p) if os.path.isfile(p) else 0
    sizes = {f: sz(os.path.join(OUT, f)) for f in ["train.zip", "test.csv.gz", "sample_submission.csv.gz",
                                                   "solution.php", "split.php", "series.php", "stats.json", "starter.ipynb.gz"]}
    print(f"\ntopics {len(topics)} · train rows {n_train:,} · test rows {n_test_rows:,} "
          f"(public {len(pub)}/private {len(prv)})")
    print("bundle sizes: " + " · ".join(f"{k} {v/1e6:.2f}MB" for k, v in sizes.items()))
    print(f"bundle → {OUT}")
    print(f"meta → {META} (PRIVATE) · reference submission → {REF_SUB}")
    print(json.dumps({k: stats[k] for k in ("n_topics", "n_train", "n_test", "n_features", "sign_entropy", "baseline")}))
    label2topic = {t["label"]: t["topic"] for t in topics}
    probes = ["christmas", "halloween", "super bowl", "swimming", "cybersecurity"] if which == "topic-500" \
        else [m.lower() for m in MEASURES]
    print("\nPROBES:")
    for probe in probes:
        hits = [lb for lb in label2topic if lb.lower() == probe] or [lb for lb in label2topic if probe in lb.lower()]
        for lb in hits[:1]:
            t = next(t for t in topics if t["label"] == lb)
            print(f"  {lb:22s} {t['topic']} phase {t['phase']:5.0f}° → {t['sign']:11s} R² {t['r2']:.3f}")


if __name__ == "__main__":
    main()
