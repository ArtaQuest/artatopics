#!/usr/bin/env python3
"""Four chart-derived topic sets against the sky — REPRODUCTION CONTRACT entrypoint.

    python3 reproduce.py <datasets.zip> [--quick N] [--from-cache atlas_signs.csv]

The dataset zip carries ImageTopics.csv, NewsTopics.csv, ShopTopics.csv and YouTubeTopics.csv —
four topic sets selected and CLASSIFIED purely from Google Trends' own lunar-month (new-moon to
new-moon) top-50 charts, 2008-2025, Worldwide: a topic's frozen `class` is the sidereal season in
which its summed chart score is largest (argmax; committed before any series was fetched). This
script fits every topic's own-property monthly interest series with the fixed sinc phase model
(topic500_reference_solution.py, verbatim; mode="atlas": 12 sign-centre starts, deterministic) and
asks whether the model's EMPIRICAL sign — where the series itself peaks in the sidereal year —
agrees with the chart-derived season.

Statistics (all deterministic): per-dataset and pooled exact/adjacent agreement vs 1/12 chance
(exact binomial as the optimistic bound; shared-token semantic clusters + a 10,000-draw fixed-seed
label permutation as the robust test); signed circular offset vs the class sign centre (bias AND
dispersion); rotation null; in-sample R^2 / adjusted R^2 / RMSE; a raw-argmax transparency check
(the fitted phase vs each series' raw monthly-mean peak, and that raw peak scored directly against
the labels); and cross-property consistency for topics charting in more than one property.

Modes: --quick N fits a deterministic every-(T//N)-th subset; --from-cache regenerates every
statistic and figure from a previous atlas_signs.csv in seconds. Declared numbers (expected.json)
come from the full fit on Apple MPS; cross-device float-32 tolerance ~±0.5 pt on headline rates.
Runtime: ~25 min GPU / ~50 min CPU; --from-cache ~10 s.
"""
import io, json, math, os, sys, zipfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import topic500_reference_solution as r5                      # the platform model — verbatim

SIGNS = r5.SIGNS
BODIES = r5.BODIES
GOLD, BLUE, INK = "#E8B923", "#1746DC", "#0C1E32"
DATASETS = ["ImageTopics", "NewsTopics", "ShopTopics", "YouTubeTopics"]
WELLFIT_R2 = 0.30
N_PARAMS = 2 * r5.NB + 2
N_PERM = 10000
TOKEN_STOP = {"video", "songs", "song"}                       # format words too generic to mark one bet


def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def circ_mean_deg(deg):
    rad = np.deg2rad(np.asarray(deg, float))
    C, S = float(np.cos(rad).mean()), float(np.sin(rad).mean())
    return math.degrees(math.atan2(S, C)), math.hypot(C, S)


def log_binom_tail(n, k, p):
    if k <= 0: return 0.0
    logs = []
    lp, lq = math.log(p), math.log(1.0 - p)
    for i in range(k, n + 1):
        logs.append(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    m = max(logs)
    return (m + math.log(sum(math.exp(x - m) for x in logs))) / math.log(10.0)


def semantic_clusters(topics):
    """Shared-token near-duplicate clusters WITHIN one dataset (deterministic union-find)."""
    n = len(topics)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    tok = [set(w for w in t.split() if len(w) >= 4 and w not in TOKEN_STOP) for t in topics]
    for a in range(n):
        for b in range(a + 1, n):
            if tok[a] & tok[b]: union(a, b)
    return [find(i) for i in range(n)]


def load_datasets(path):
    out = {}
    if path.lower().endswith(".zip"):
        zf = zipfile.ZipFile(path)
        names = {os.path.basename(n): n for n in zf.namelist()}
        for ds in DATASETS:
            out[ds] = pd.read_csv(io.BytesIO(zf.read(names[f"{ds}.csv"])))
    else:                                                     # a directory of the four CSVs
        for ds in DATASETS:
            out[ds] = pd.read_csv(os.path.join(path, f"{ds}.csv"))
    return out


def fit_dataset(ds, df, quick):
    topics = list(dict.fromkeys(df["topic"]))
    if quick:
        step = max(1, len(topics) // quick)
        topics = topics[::step][:quick]
    by = {t: g for t, g in df.groupby("topic", sort=False)}
    theory = {t: str(by[t]["class"].iloc[0]) for t in topics}
    Ys = [by[t]["value"].to_numpy(float) for t in topics]
    Xs = [by[t][BODIES].to_numpy(float) for t in topics]
    dev = os.environ.get("AQ_T500_DEVICE") or None
    try:
        par = r5.fit_many(Ys, Xs, mode="atlas", chunk=200, device=dev, progress=True)
    except Exception as e:
        print(f"[{ds}] device fit failed ({e}) — retrying on cpu", flush=True)
        par = r5.fit_many(Ys, Xs, mode="atlas", chunk=200, device="cpu", progress=True)
    rows = []
    for i, t in enumerate(topics):
        y, X = Ys[i], Xs[i]
        pred = r5.predict(par[i], X)
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y - pred) ** 2)) / sst if sst > 1e-9 else 0.0
        nm = len(y)
        adj = 1.0 - (1.0 - r2) * (nm - 1) / (nm - N_PARAMS - 1)
        phase = r5.phase_of(par[i])
        emp = r5.sign_of(par[i])
        ti, ei = SIGNS.index(theory[t]), SIGNS.index(emp)
        d_sign = min((ei - ti) % 12, (ti - ei) % 12)
        off = float(wrap180(phase - (ti * 30 + 15)))
        # raw-argmax transparency: monthly-mean peak of the raw series -> that month's mean sun lon
        mo = pd.to_datetime(by[t]["month"]).dt.month.to_numpy()[:nm]
        mm = [y[mo == m].mean() if (mo == m).any() else -1 for m in range(1, 13)]
        rows.append({"dataset": ds, "topic": t, "class": theory[t], "phase": round(phase, 2),
                     "empirical": emp, "sign_dist": int(d_sign), "offset_deg": round(off, 2),
                     "train_r2": round(r2, 4), "adj_r2": round(adj, 4),
                     "rmse": round(float(np.sqrt(np.mean((y - pred) ** 2))), 3),
                     "raw_peak_month": int(np.argmax(mm)) + 1})
    return pd.DataFrame(rows)


def stats_for(out, sun_by_month):
    n = len(out)
    ti = out["class"].map(SIGNS.index).to_numpy(int)
    ph = out["phase"].to_numpy(float)
    agree = int((out["sign_dist"] == 0).sum())
    adj = int((out["sign_dist"] <= 1).sum())
    offs = out["offset_deg"].to_numpy(float)
    mu, R = circ_mean_deg(offs)
    rot = [float((((out["empirical"].map(SIGNS.index) - ti) % 12) == k).mean() * 100) for k in range(12)]
    # semantic-cluster permutation
    cid = semantic_clusters(list(out["topic"]))
    cdf = out.assign(cid=cid)
    cl = cdf.groupby("cid").agg(label=("class", "first"),
                                emp=("empirical", lambda s: s.mode().iloc[0])).reset_index()
    hits = int((cl["label"] == cl["emp"]).sum())
    rng = np.random.default_rng(0)
    lab = cl["label"].map(SIGNS.index).to_numpy(int)
    emp = cl["emp"].map(SIGNS.index).to_numpy(int)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = (rng.permutation(lab) == emp).sum()
    p_perm = (1 + int((null >= hits).sum())) / (1 + N_PERM)
    z_perm = float((hits - null.mean()) / (null.std() or 1.0))
    # raw peak agreement (transparency)
    raw_lon = out["raw_peak_month"].map(sun_by_month)
    raw_sign = (raw_lon // 30).astype(int) % 12
    raw_agree = float((raw_sign.to_numpy() == ti).mean() * 100)
    raw_d = np.abs(wrap180(ph - raw_lon.to_numpy(float)))
    # permutation for the RAW-PEAK classifier at the same cluster level (seed 1)
    clr = cdf.assign(raw=raw_sign.to_numpy()).groupby("cid").agg(
        label=("class", "first"), raw=("raw", lambda s: int(s.mode().iloc[0]))).reset_index()
    hits_r = int((clr["label"].map(SIGNS.index).to_numpy() == clr["raw"].to_numpy()).sum())
    rng_r = np.random.default_rng(1)
    lab_r = clr["label"].map(SIGNS.index).to_numpy(int)
    null_r = np.empty(N_PERM)
    for i in range(N_PERM):
        null_r[i] = (rng_r.permutation(lab_r) == clr["raw"].to_numpy()).sum()
    raw_perm_p = (1 + int((null_r >= hits_r).sum())) / (1 + N_PERM)
    raw_perm_z = float((hits_r - null_r.mean()) / (null_r.std() or 1.0))
    wf = out[out["train_r2"] >= WELLFIT_R2]
    return {
        "n": n, "agree_pct": round(agree / n * 100, 4), "agree_adjacent_pct": round(adj / n * 100, 4),
        "lift": round((agree / n) * 12, 4),
        "log10_p_binomial_optimistic": round(log_binom_tail(n, agree, 1 / 12), 4),
        "clusters_n": len(cl), "cluster_agree_n": hits,
        "cluster_agree_pct": round(hits / len(cl) * 100, 4),
        "perm_z": round(z_perm, 4), "perm_p": p_perm,
        "mean_signed_offset_deg": round(mu, 4), "offset_R": round(R, 4),
        "mean_abs_offset_deg": round(float(np.abs(offs).mean()), 4),
        "median_abs_offset_deg": round(float(np.median(np.abs(offs))), 4),
        "rotation_max_pct": round(max(rot[1:]), 4), "rotation_max_k": int(1 + int(np.argmax(rot[1:]))),
        "rotation_mean_excl_best_pct": round(float((sum(rot[1:]) - max(rot[1:])) / 10.0), 4),
        "mean_train_r2": round(float(out["train_r2"].mean()), 4),
        "median_train_r2": round(float(out["train_r2"].median()), 4),
        "mean_adj_r2": round(float(out["adj_r2"].mean()), 4),
        "n_wellfit": int(len(wf)),
        "agree_wellfit_pct": round(float((wf["sign_dist"] == 0).mean() * 100), 4) if len(wf) else 0.0,
        "model_vs_rawpeak_median_deg": round(float(np.median(raw_d)), 4),
        "model_vs_rawpeak_within_sign_pct": round(float((raw_d <= 30.0).mean() * 100), 4),
        "rawpeak_agree_pct": round(raw_agree, 4),
        "rawpeak_perm_z": round(raw_perm_z, 4), "rawpeak_perm_p": raw_perm_p,
        "per_class_agree_pct": {s: round(float((out[out["class"] == s]["sign_dist"] == 0).mean() * 100), 2)
                                for s in SIGNS if (out["class"] == s).any()},
    }


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__.splitlines()[2]); sys.exit(2)
    quick = int(args[args.index("--quick") + 1]) if "--quick" in args else 0
    cache = args[args.index("--from-cache") + 1] if "--from-cache" in args else ""
    data = load_datasets(args[0])

    # month -> circular-mean sidereal sun longitude, from the datasets' own sun column
    allrows = pd.concat(data.values(), ignore_index=True)
    moy = pd.to_datetime(allrows["month"]).dt.month.to_numpy()
    sun = allrows["sun"].to_numpy(float)
    sun_by_month = {m: circ_mean_deg(sun[moy == m])[0] % 360.0 for m in range(1, 13)}

    if cache:
        out = pd.read_csv(cache)
    else:
        out = pd.concat([fit_dataset(ds, df, quick) for ds, df in data.items()], ignore_index=True)
        out.to_csv(os.path.join(HERE, "atlas_signs.csv"), index=False)

    res = {"pooled": stats_for(out, sun_by_month)}
    for ds in DATASETS:
        sub = out[out["dataset"] == ds].reset_index(drop=True)
        if len(sub):
            res[ds] = stats_for(sub, sun_by_month)
    # cross-property consistency: same topic in >1 dataset — do the empirical signs agree?
    multi = out.groupby("topic").filter(lambda g: g["dataset"].nunique() > 1)
    pairs = same = 0
    for t, g in multi.groupby("topic"):
        es = list(g["empirical"])
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                pairs += 1; same += (es[i] == es[j])
    res["cross_property"] = {"topics_in_multiple": int(multi["topic"].nunique()),
                             "pairs": pairs, "same_sign_pct": round(same / pairs * 100, 4) if pairs else None,
                             "chance_pct": round(100 / 12, 4)}
    # flat aliases for the pre-flight comparator
    for k in ("agree_pct", "cluster_agree_pct", "mean_signed_offset_deg", "rawpeak_agree_pct",
              "rawpeak_perm_z", "perm_z", "mean_train_r2", "n"):
        res[f"pooled_{k}"] = res["pooled"][k]
    res["cross_same_sign_pct"] = res["cross_property"]["same_sign_pct"]
    for ds in DATASETS:
        if ds in res:
            res[f"{ds}_agree_pct"] = res[ds]["agree_pct"]
            res[f"{ds}_rawpeak_agree_pct"] = res[ds]["rawpeak_agree_pct"]
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))
    print(json.dumps({ds: {kk: res[ds][kk] for kk in ("n", "agree_pct", "perm_z", "perm_p")}
                      for ds in DATASETS if ds in res}, indent=1))
    figures(out)


def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 9, "axes.edgecolor": "#d8deea",
                         "text.color": INK, "axes.labelcolor": INK})
    # fig1 — 2x2 confusion grid (one per dataset)
    f, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, ds in zip(axes.flat, DATASETS):
        sub = out[out["dataset"] == ds]
        M = np.zeros((12, 12))
        for _, r in sub.iterrows():
            M[SIGNS.index(r["class"]), SIGNS.index(r["empirical"])] += 1
        Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
        ax.imshow(Mn, cmap="cividis", vmin=0, vmax=max(0.25, Mn.max()))
        ax.set_xticks(range(12), [s[:3] for s in SIGNS], rotation=90, fontsize=6)
        ax.set_yticks(range(12), [s[:3] for s in SIGNS], fontsize=6)
        for i in range(12):
            ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, edgecolor=GOLD, lw=1.2))
        ax.set_title(f"{ds} (n={len(sub)})", fontsize=10)
    f.suptitle("Chart-season class (rows) vs sinc-model sign (cols) — row-normalised, cividis", y=0.995)
    f.tight_layout(); f.savefig(os.path.join(HERE, "fig1_confusions.png"), dpi=200); plt.close(f)

    # fig2 — agreement per dataset + pooled vs chance
    vals = [float((out[out["dataset"] == ds]["sign_dist"] == 0).mean() * 100) for ds in DATASETS]
    vals.append(float((out["sign_dist"] == 0).mean() * 100))
    labels = [d.replace("Topics", "") for d in DATASETS] + ["Pooled"]
    f, a = plt.subplots(figsize=(7.4, 3.6))
    a.bar(range(len(vals)), vals, color=[BLUE] * len(DATASETS) + [GOLD])
    for i, v in enumerate(vals):
        a.text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
    a.axhline(100 / 12, color=INK, lw=1, ls="--", label="chance (8.3%)")
    a.set_xticks(range(len(vals)), labels)
    a.set_ylabel("exact-sign agreement (%)"); a.legend(frameon=False)
    [a.spines[sp].set_visible(False) for sp in ("top", "right")]
    f.tight_layout(); f.savefig(os.path.join(HERE, "fig2_agreement.png"), dpi=200); plt.close(f)

    # fig3 — pooled signed-offset rose
    offs = out["offset_deg"].to_numpy(float)
    mu, R = circ_mean_deg(offs)
    f = plt.figure(figsize=(6.2, 6.2))
    a = f.add_subplot(111, projection="polar")
    a.set_theta_zero_location("N"); a.set_theta_direction(-1)
    hist, edges = np.histogram(offs, bins=np.arange(-180, 181, 10))
    centers = np.deg2rad(((edges[:-1] + edges[1:]) / 2) % 360.0)
    a.bar(centers, hist, width=np.deg2rad(10), color=BLUE, alpha=.8, edgecolor="white", lw=.4)
    a.annotate("", xy=(np.deg2rad(mu % 360.0), R * hist.max()), xytext=(0, 0),
               arrowprops=dict(color=GOLD, lw=3, arrowstyle="-|>"))
    a.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    a.set_xticklabels([f"{d if d <= 180 else d - 360}°" for d in range(0, 360, 30)], fontsize=8)
    a.set_rlabel_position(202.5); a.tick_params(axis="y", labelsize=7, colors="#6B7A90")
    a.set_title(f"Pooled signed offset: fitted phase − class centre (radial: topics per 10° bin)\n"
                f"gold vector: circular mean {mu:.1f}°, R = {R:.2f}", fontsize=9)
    f.tight_layout(); f.savefig(os.path.join(HERE, "fig3_rose.png"), dpi=200); plt.close(f)

    # fig4 — R2 distribution per dataset (fit quality context)
    f, a = plt.subplots(figsize=(7.4, 3.4))
    for i, ds in enumerate(DATASETS):
        sub = out[out["dataset"] == ds]["train_r2"]
        a.boxplot(sub, positions=[i], widths=0.55, showfliers=False,
                  medianprops=dict(color=GOLD, lw=2))
    a.set_xticks(range(len(DATASETS)), [d.replace("Topics", "") for d in DATASETS])
    a.set_ylabel("in-sample $R^2$ (24-param fit, 210 months)")
    [a.spines[sp].set_visible(False) for sp in ("top", "right")]
    f.tight_layout(); f.savefig(os.path.join(HERE, "fig4_r2.png"), dpi=200); plt.close(f)
    print("figures: fig1_confusions.png fig2_agreement.png fig3_rose.png fig4_r2.png")


if __name__ == "__main__":
    main()
