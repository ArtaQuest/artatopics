#!/usr/bin/env python3
"""
backtest.py — the honest test: does ArtaAstro's a-priori sidereal-Lahiri "event-intensity" signal
have any skill at anticipating real-world conflict, measured against the ENTIRE GDELT history
(global daily aggregates, 1979-present)?

Inputs (produced by build.py / fetch_gdelt.py):
  out/daily_intensity.csv       date, intensity(0-100), ...        (the astrological signal A)
  out/world_conflict_daily.csv  date, n_events, n_q4, ...          (GDELT ground truth)

Ground-truth conflict measures (all SHARE-based, so GDELT's ~1000x growth in raw volume since 1979
cancels out): material-conflict share q4/total, conflict share (q3+q4)/total, -mean Goldstein,
-mean tone. Both A and G are adaptively DETRENDED (365-day centered rolling z-score) so we test the
short-term structure, not a spurious shared trend; day-of-week seasonality is removed from G.

Skill metrics: Spearman rho (levels + detrended), ROC-AUC of A for top-decile conflict days,
precision@k, and a lead/lag cross-correlation over +-60 days. Significance comes from a
CIRCULAR-SHIFT PERMUTATION NULL (5000 random time-shifts of A, preserving its autocorrelation) —
the correct null for two autocorrelated series. We also run the classic Barbault annual-index test
(mean intensity vs mean conflict per year) and report pre-2013 vs 2013+ split stability.

Outputs: RESULTS.md, backtest.json, and PNG plots in out/. Whatever the numbers are, they are
reported straight — no cherry-picking, no post-hoc weight tuning.
"""
import os, sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
RNG = np.random.default_rng(20200225)   # fixed seed (Date.now-free, reproducible)

def load():
    A = pd.read_csv(os.path.join(OUT,"daily_intensity.csv"), parse_dates=["date"]).set_index("date")
    G = pd.read_csv(os.path.join(OUT,"world_conflict_daily.csv"), parse_dates=["date"]).set_index("date")
    G = G[G["n_events"] > 0].copy()
    G["material_share"] = G["n_q4"] / G["n_events"]
    G["conflict_share"] = (G["n_q3"] + G["n_q4"]) / G["n_events"]
    G["neg_goldstein"]  = -(G["sum_goldstein"] / G["n_events"])
    G["neg_tone"]       = -(G["sum_tone"] / G["n_events"])
    return A, G

def detrend(s, win=365):
    """Centered rolling z-score: removes trend/seasonal/regime shifts, keeps short-term anomaly."""
    s = s.astype(float)
    mu = s.rolling(win, center=True, min_periods=win//2).mean()
    sd = s.rolling(win, center=True, min_periods=win//2).std()
    return (s - mu) / sd.replace(0, np.nan)

def deweekday(z):
    z = z.copy()
    dow = z.index.dayofweek
    for d in range(7):
        m = dow == d
        z.loc[m] = z.loc[m] - z.loc[m].mean()
    return z

def circular_perm_pvalue(a, g, stat_fn, obs, n=5000, min_shift=30):
    """p = P(|null stat| >= |obs|); null = circular shifts of a (preserves autocorrelation)."""
    N = len(a); cnt = 0; null = np.empty(n)
    for i in range(n):
        k = RNG.integers(min_shift, N - min_shift)
        null[i] = stat_fn(np.roll(a, k), g)
        if abs(null[i]) >= abs(obs): cnt += 1
    return (cnt + 1) / (n + 1), null

def auc(score, label):
    """Rank-based ROC-AUC (Mann-Whitney)."""
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score)+1)
    pos = label == 1; npos = pos.sum(); nneg = (~pos).sum()
    if npos == 0 or nneg == 0: return np.nan
    return (ranks[pos].sum() - npos*(npos+1)/2) / (npos*nneg)

def analyse(A, G, gcol, label):
    df = pd.DataFrame({"A": A["intensity"], "G": G[gcol]}).dropna()
    Az, Gz = detrend(df["A"]), deweekday(detrend(df["G"]))
    d = pd.DataFrame({"A": df["A"], "G": df["G"], "Az": Az, "Gz": Gz}).dropna()
    a, g, az, gz = d["A"].values, d["G"].values, d["Az"].values, d["Gz"].values

    rho_lvl  = stats.spearmanr(a, g).statistic
    rho_detr = stats.spearmanr(az, gz).statistic
    thr = np.quantile(gz, 0.90); lab = (gz >= thr).astype(int)
    a_auc = auc(az, lab)
    # precision@k where k = number of true high days
    k = int(lab.sum()); topk = np.argsort(az)[-k:]; prec = lab[topk].mean(); base = lab.mean()

    sp = lambda x, y: stats.spearmanr(x, y).statistic
    p_detr, null = circular_perm_pvalue(az, gz, sp, rho_detr)
    p_auc, _     = circular_perm_pvalue(az, lab, lambda x, y: auc(x, y) - 0.5, a_auc - 0.5, n=2000)

    # lead/lag on detrended (positive L = astro leads conflict by L days)
    lags = range(-60, 61)
    xcorr = [np.corrcoef(np.roll(az, L), gz)[0, 1] for L in lags]

    return {
        "measure": label, "gcol": gcol, "n_days": int(len(d)),
        "span": [str(d.index.min().date()), str(d.index.max().date())],
        "spearman_levels": float(rho_lvl), "spearman_detrended": float(rho_detr),
        "perm_p_detrended": float(p_detr),
        "auc_top_decile": float(a_auc), "auc_perm_p": float(p_auc),
        "precision_at_k": float(prec), "base_rate": float(base), "k": k,
        "leadlag_best": {"lag_days": int(list(lags)[int(np.nanargmax(np.abs(xcorr)))]),
                         "corr": float(np.nanmax(np.abs(xcorr)))},
        "_null": null, "_lags": list(lags), "_xcorr": xcorr, "_az": az, "_gz": gz, "_idx": d.index,
    }

def barbault(A, G, gcol):
    df = pd.DataFrame({"A": A["intensity"], "G": G[gcol]}).dropna()
    yr = df.groupby(df.index.year).mean()
    yr = yr[yr.index >= yr.index.min()]  # keep all
    rho = stats.spearmanr(yr["A"], yr["G"]).statistic
    # permute years
    n = 5000; cnt = 0; vals = yr["A"].values; g = yr["G"].values
    for _ in range(n):
        cnt += abs(stats.spearmanr(RNG.permutation(vals), g).statistic) >= abs(rho)
    return {"n_years": int(len(yr)), "spearman": float(rho), "perm_p": float((cnt+1)/(n+1)),
            "_years": yr.index.tolist(), "_A": yr["A"].tolist(), "_G": yr["G"].tolist()}

def split_stability(A, G, gcol):
    out = {}
    for name, lo, hi in [("1979-2012", "1979", "2012"), ("2013-now", "2013", "2100")]:
        a = A.loc[lo:hi]; g = G.loc[lo:hi]
        try:
            r = analyse(a, g, gcol, name)
            out[name] = {"spearman_detrended": r["spearman_detrended"], "perm_p": r["perm_p_detrended"],
                         "auc": r["auc_top_decile"], "n_days": r["n_days"]}
        except Exception as e:
            out[name] = {"error": str(e)}
    return out

def plots(res, bar):
    # 1) lead/lag
    plt.figure(figsize=(7,3.4))
    plt.axhline(0,color="#888",lw=.7); plt.axvline(0,color="#888",lw=.7,ls=":")
    plt.plot(res["_lags"], res["_xcorr"], color="#1746DC")
    plt.title("Lead/lag: astro intensity vs GDELT conflict (detrended)")
    plt.xlabel("astro leads (days) →"); plt.ylabel("correlation"); plt.tight_layout()
    plt.savefig(os.path.join(OUT,"plot_leadlag.png"), dpi=110); plt.close()
    # 2) permutation null
    plt.figure(figsize=(7,3.4))
    plt.hist(res["_null"], bins=60, color="#bbb", edgecolor="none")
    plt.axvline(res["spearman_detrended"], color="#E8B923", lw=2,
                label=f"observed ρ={res['spearman_detrended']:.4f}\np={res['perm_p_detrended']:.3f}")
    plt.title("Permutation null (circular shifts) vs observed"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT,"plot_null.png"), dpi=110); plt.close()
    # 3) Barbault annual overlay
    fig,ax1=plt.subplots(figsize=(8,3.4))
    ax1.plot(bar["_years"], bar["_A"], color="#E8B923", label="astro intensity (annual mean)")
    ax1.set_ylabel("astro intensity", color="#E8B923")
    ax2=ax1.twinx(); ax2.plot(bar["_years"], bar["_G"], color="#1746DC", label="GDELT conflict (annual mean)")
    ax2.set_ylabel("conflict share", color="#1746DC")
    plt.title(f"Barbault annual index  (ρ={bar['spearman']:.2f}, p={bar['perm_p']:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"plot_annual.png"), dpi=110); plt.close()

def write_results_md(result):
    p = result["primary"]; b = result["barbault_annual"]
    def verdict(pval): return "**not significant**" if pval >= 0.05 else "significant (p<0.05)"
    lines = []
    lines.append("# ArtaAstro backtest — results\n")
    lines.append("_Auto-generated by `backtest.py`. The intensity model (`intensity.py`) was frozen "
                 "**before** any GDELT data was loaded and was never tuned against it._\n")
    lines.append(f"**Ground truth:** {result['ground_truth']}  \n"
                 f"**Overlap span:** {result['overlap_span'][0]} → {result['overlap_span'][1]} "
                 f"({p['n_days']:,} days)  \n"
                 f"**Engine:** {result['engine']}\n")
    lines.append("## Headline — does the astrological signal anticipate real conflict?\n")
    lines.append("| Metric | Value | Null p-value | Chance? |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Spearman ρ (detrended) | {p['spearman_detrended']:+.4f} | {p['perm_p_detrended']:.3f} | {verdict(p['perm_p_detrended'])} |")
    lines.append(f"| ROC-AUC (top-decile conflict days) | {p['auc_top_decile']:.4f} | {p['auc_perm_p']:.3f} | {verdict(p['auc_perm_p'])} |")
    lines.append(f"| Precision@k | {p['precision_at_k']:.4f} | (base rate {p['base_rate']:.4f}) | — |")
    lines.append(f"| Spearman ρ (raw levels) | {p['spearman_levels']:+.4f} | — | — |")
    lines.append(f"| Best lead/lag correlation | {p['leadlag_best']['corr']:+.4f} @ {p['leadlag_best']['lag_days']:+d} days | — | — |")
    lines.append(f"| Barbault annual index (ρ over {b['n_years']} yrs) | {b['spearman']:+.4f} | {b['perm_p']:.3f} | {verdict(b['perm_p'])} |\n")
    lines.append("Null = 5,000 circular time-shifts of the astro signal (preserves its autocorrelation). "
                 "AUC 0.5 and ρ 0 are the no-skill baselines.\n")
    lines.append("## Other conflict measures (same test)\n")
    lines.append("| GDELT measure | Spearman ρ (detrended) | null p | AUC |")
    lines.append("|---|---|---|---|")
    for c, o in result["other_measures"].items():
        lines.append(f"| {c} | {o['spearman_detrended']:+.4f} | {o['perm_p_detrended']:.3f} | {o['auc_top_decile']:.4f} |")
    lines.append("\n## Split stability\n")
    lines.append("| Era | Spearman ρ (detrended) | null p | AUC | days |")
    lines.append("|---|---|---|---|---|")
    for era, s in result["split_stability"].items():
        if "error" in s: lines.append(f"| {era} | — | — | — | (n/a) |")
        else: lines.append(f"| {era} | {s['spearman_detrended']:+.4f} | {s['perm_p']:.3f} | {s['auc']:.4f} | {s['n_days']:,} |")
    # honest one-line conclusion
    sig = p["perm_p_detrended"] < 0.05 or p["auc_perm_p"] < 0.05 or b["perm_p"] < 0.05
    lines.append("\n## Conclusion\n")
    if sig:
        lines.append("At least one measure clears the permutation null at p<0.05. This is **not** proof of "
                     "astrological causation — with many measures some small effect can arise — but it is "
                     "reported straight and is worth a closer, pre-registered look. See the effect size "
                     "(ρ and AUC): even where 'significant', the effect is tiny.")
    else:
        lines.append("**No measure beats chance.** Across the entire GDELT history, the a-priori "
                     "sidereal-Lahiri event-intensity signal shows correlation and classification skill "
                     "indistinguishable from a randomly time-shifted copy of itself. Effect sizes are "
                     "near zero (ρ ≈ 0, AUC ≈ 0.5). This is the honest result: the model produces a real, "
                     "precise sky-reading for every day — but it does **not** forecast world events.")
    lines.append("\n![annual](out/plot_annual.png)\n![lead/lag](out/plot_leadlag.png)\n![null](out/plot_null.png)\n")
    open(os.path.join(HERE, "RESULTS.md"), "w").write("\n".join(lines))

def main():
    A, G = load()
    span = (max(A.index.min(), G.index.min()), min(A.index.max(), G.index.max()))
    print(f"overlap: {span[0].date()} .. {span[1].date()}")
    primary = analyse(A, G, "material_share", "material-conflict share (q4/total)")
    others  = {c: {k:v for k,v in analyse(A,G,c,c).items() if not k.startswith("_")}
               for c in ["conflict_share","neg_goldstein","neg_tone"]}
    bar = barbault(A, G, "material_share")
    split = split_stability(A, G, "material_share")
    plots(primary, bar)

    result = {
        "engine": "kerykeion (sidereal Lahiri) + swisseph; a-priori intensity model",
        "ground_truth": "GDELT 1.0 global daily aggregates, entire history",
        "overlap_span": [str(span[0].date()), str(span[1].date())],
        "primary": {k:v for k,v in primary.items() if not k.startswith("_")},
        "other_measures": others, "barbault_annual": {k:v for k,v in bar.items() if not k.startswith("_")},
        "split_stability": split,
    }
    json.dump(result, open(os.path.join(OUT,"backtest.json"),"w"), indent=2)
    write_results_md(result)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
