"""Two checks the 0.5385 needs: (1) does the sky beat the calendar's own seasonality — Sun-only vs
full sky; (2) circular-shift null: shift the label series by a random offset ≥ 1 year (keeps the
label's autocorrelation, breaks any sky-label link) — the honest chance level for these AUCs."""
import os, sys, numpy as np, pandas as pd, importlib.util
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dt_", os.path.expanduser("~/.artaquest-dev/artatopics/analysis/arxivtopics/competition/daily_trend.py"))
src = open(spec.origin).read().split("rows = []")[0]
ns = {"__file__": spec.origin}; exec(compile(src, spec.origin, "exec"), ns)
daily, rel, DAYS, sky_feats, CAL, peaks_and_labels, fit_auc, L, TAU, valid_eph = (ns[k] for k in ("daily","rel","DAYS","sky_feats","CAL","peaks_and_labels","fit_auc","L","TAU","valid_eph"))
rng = np.random.RandomState(0)
out = []
for cat in daily.columns:
    if cat not in rel.index: continue
    start = max(np.searchsorted(DAYS, rel.loc[cat,"reliable_from"].date()), int(np.argmax(valid_eph)))
    x = daily[cat].to_numpy(float)[start:]
    if len(x) < 8*365: continue
    s, peak, lab = peaks_and_labels(x); y = lab.astype(int)
    idx = np.arange(start, start+len(x)); cut = int(len(x)*0.8); tr, te = np.arange(cut), np.arange(cut, len(x))
    Xs = sky_feats(start)[idx]
    Xsun = np.stack([np.cos(L[idx,0]), np.sin(L[idx,0]), np.cos(2*L[idx,0]), np.sin(2*L[idx,0])], 1)   # Sun only = the season
    Xnosun = np.delete(Xs, [0,1, 16,17, 32,33], axis=1)                                                    # sky minus Sun harmonics 1-3
    a_sky,_ = fit_auc(Xs, y, tr, te); a_sun,_ = fit_auc(Xsun, y, tr, te); a_nosun,_ = fit_auc(Xnosun, y, tr, te)
    # circular-shift null: 3 draws
    nulls = []
    for _ in range(3):
        sh = rng.randint(400, len(x)-400); yn = np.roll(y, sh)
        a,_ = fit_auc(Xs, yn, tr, te); nulls.append(a)
    out.append(dict(category=cat, sky=a_sky, sun_only=a_sun, sky_no_sun=a_nosun, null=float(np.nanmean(nulls))))
df = pd.DataFrame(out)
print(f"{len(df)} categories")
for k in ("sky","sun_only","sky_no_sun","null"): print(f"  {k:<12} mean {df[k].mean():.4f} · median {df[k].median():.4f}")
print(f"\n  sky minus null, mean: {(df.sky-df.null).mean():+.4f} · categories where sky > null: {(df.sky>df.null).mean()*100:.0f}%")
print(f"  sky_no_sun minus null: {(df.sky_no_sun-df.null).mean():+.4f}")
print("\n  top 6 by sky:"); print(df.sort_values('sky',ascending=False).head(6).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
df.to_csv(os.path.expanduser("~/.artaquest-dev/artacomp/daily/daily_checks.csv"), index=False)
