#!/usr/bin/env python3
"""THE EXACT PHASOR, SOLVED ANALYTICALLY  —  y(t) = | b + Σᵢ aᵢ · e^{ i(θᵢ(t) − pᵢ) } |²

Inputs: only θᵢ(t), the sidereal (Lahiri) longitudes from kerykeion / Swiss Ephemeris. Unknowns:
b, aᵢ, pᵢ. Solved in closed form — no optimiser, no seed, no restart.

DERIVATION.  Expand the square exactly:
    y = b² + Σᵢ aᵢ²                                                    (constant)
      + Σᵢ 2 b aᵢ cos(θᵢ − pᵢ)                                          (transits)
      + Σᵢ<ₖ 2 aᵢ aₖ cos( (θᵢ − pᵢ) − (θₖ − pₖ) )                        (aspects)
Each cos(θ − p) = cos p · cos θ + sin p · sin θ, so y is LINEAR in the fixed basis
    Φ(t) = [ 1,  cos θᵢ, sin θᵢ  (i=1..B),  cos(θᵢ−θₖ), sin(θᵢ−θₖ)  (i<k) ]
with coefficients that are polynomial in (b, aᵢ, pᵢ):
    c₀        = b² + Σ aᵢ²
    αᵢ, βᵢ    = 2 b aᵢ cos pᵢ ,  2 b aᵢ sin pᵢ
    γᵢₖ, δᵢₖ  = 2 aᵢ aₖ cos(pᵢ−pₖ) ,  2 aᵢ aₖ sin(pᵢ−pₖ)
Step 1 (linear least squares, closed form):  ĉ = (ΦᵀWΦ + λR)⁻¹ ΦᵀW y.
Step 2 (algebraic inversion, closed form):
    pᵢ = atan2(βᵢ, αᵢ)                       — the phase of transit i, exactly
    Mᵢ = √(αᵢ² + βᵢ²) = 2 b aᵢ                — so aᵢ = Mᵢ / (2b) once b is known
    c₀ = b² + Σ Mᵢ²/(4b²)  ⇒  4b⁴ − 4c₀b² + ΣMᵢ² = 0  ⇒  b² = ( c₀ + √(c₀² − ΣMᵢ²) ) / 2
    (the '+' root; the '−' root does NOT reproduce the fitted curve — verified numerically)
Step 3 (consistency check, and the reason this is not just a linear regression): the aspect
coefficients are DETERMINED by step 2 — γᵢₖ must equal 2aᵢaₖcos(pᵢ−pₖ). The 1+2B free transit
coefficients (plus c₀) fix all B(B−1) aspect coefficients. We report how far the freely fitted
aspects sit from the values the phasor implies: that residual is the part of the data the model
form cannot represent. The exact-model prediction uses the projected (b, aᵢ, pᵢ), never the free ĉ.

Identifiability: with B bodies there are 1+2B unknowns and 1+2B transit-side coefficients, so the
system is exactly determined given ĉ; the aspect side is over-determination that tests the form.
The overall sign of each aᵢ is absorbed by pᵢ → pᵢ+π, so aᵢ ≥ 0 by convention.

  python3 analysis/arxivtopics/competition/phasor_exact.py   (self-test, then the daily arXiv fit)
"""
import os, sys, json, datetime as dt
import numpy as np, pandas as pd

def basis(TH):
    """Φ(t): the fixed functions of the sky the square expands into. TH: (T, B) radians."""
    T, B = TH.shape; C = [np.ones(T)]
    for i in range(B): C += [np.cos(TH[:,i]), np.sin(TH[:,i])]
    for i in range(B):
        for k in range(i+1, B): d = TH[:,i]-TH[:,k]; C += [np.cos(d), np.sin(d)]
    return np.stack(C, 1)

def solve_phasor(TH, y, w=None, ridge=0.0, TH_anchor=None, anchor_level=None, anchor_w=0.0):
    """Analytic solution of y = |b + Σ aᵢ e^{i(θᵢ−pᵢ)}|² for b, aᵢ, pᵢ. Returns params + diagnostics.

    Optional HORIZON ANCHOR (the campaign's, +0.17 on the shares task): extra rows Φ(θ over the
    forecast horizon) whose target is the recent level, weighted anchor_w — still one closed-form
    least-squares solve. It holds the forecast near where the series IS, so the arrows carry
    modulation and not a runaway level."""
    T, B = TH.shape; Phi = basis(TH)
    W = np.ones(T) if w is None else np.asarray(w, float)
    R = np.eye(Phi.shape[1]); R[0,0] = 0.0
    A = Phi.T @ (Phi*W[:,None]); rhs = Phi.T @ (W*y)
    if TH_anchor is not None and anchor_w > 0:
        Pa = basis(TH_anchor); A = A + anchor_w * (Pa.T @ Pa); rhs = rhs + anchor_w * Pa.sum(0) * anchor_level
    c = np.linalg.solve(A + ridge*R + 1e-12*np.eye(Phi.shape[1]), rhs)
    c0 = c[0]; alpha = c[1:1+2*B:2]; beta = c[2:2+2*B:2]
    p = np.arctan2(beta, alpha)                          # exact phases
    M = np.sqrt(alpha**2 + beta**2)                      # = 2 b aᵢ
    disc = c0**2 - (M**2).sum()
    feasible = disc >= 0 and c0 > 0
    if not feasible:                                     # data outside the model's reach: project to boundary
        scale = np.sqrt(max(c0, 1e-12)**2 / max((M**2).sum(), 1e-24)); M = M*min(1.0, scale); disc = max(c0**2 - (M**2).sum(), 0.0)
    b = np.sqrt(max((c0 + np.sqrt(disc))/2.0, 1e-18)); a = M/(2*b)
    # step 3: what the phasor IMPLIES for the aspect coefficients vs what the free fit found
    gam_free = c[1+2*B::2]; del_free = c[2+2*B::2]
    gam_imp, del_imp = [], []
    for i in range(B):
        for k in range(i+1, B): gam_imp.append(2*a[i]*a[k]*np.cos(p[i]-p[k])); del_imp.append(2*a[i]*a[k]*np.sin(p[i]-p[k]))
    gam_imp, del_imp = np.array(gam_imp), np.array(del_imp)
    asp_free = np.concatenate([gam_free, del_free]); asp_imp = np.concatenate([gam_imp, del_imp])
    return dict(b=float(b), a=a, p=p, c_free=c, feasible=bool(feasible),
                aspect_residual=float(np.sqrt(((asp_free-asp_imp)**2).sum())/max(np.sqrt((asp_free**2).sum()),1e-12)))

def predict(TH, prm):
    z = prm["b"] + (prm["a"][None,:]*np.exp(1j*(TH - prm["p"][None,:]))).sum(1)
    return np.abs(z)**2

# ═══ SELF-TEST: recover known parameters from data generated by the exact model ═══════════════
if __name__ == "__main__":
    rng = np.random.RandomState(7); B = 8; T = 4000
    b0 = 2.0; a0 = rng.rand(B)*0.6; p0 = rng.rand(B)*2*np.pi
    TH = np.cumsum(rng.rand(T, B)*0.3, 0) % (2*np.pi)              # 8 bodies at unrelated rates
    y0 = predict(TH, dict(b=b0, a=a0, p=p0))
    prm = solve_phasor(TH, y0)
    print("═ self-test on exact synthetic data")
    print(f"  b   true {b0:.6f}  solved {prm['b']:.6f}")
    print(f"  a   max|err| {np.abs(prm['a']-a0).max():.2e}")
    print(f"  p   max|err| {np.abs(((prm['p']-p0+np.pi)%(2*np.pi))-np.pi).max():.2e} rad")
    print(f"  aspect residual (free vs implied): {prm['aspect_residual']:.2e}  ← ~0 means the free fit already IS a phasor")
    y0n = y0 + rng.randn(T)*0.05*y0.std()
    prn = solve_phasor(TH, y0n)
    print(f"  with 5% noise: b {prn['b']:.4f}, max a err {np.abs(prn['a']-a0).max():.3f}, max p err {np.abs(((prn['p']-p0+np.pi)%(2*np.pi))-np.pi).max():.3f} rad, aspect residual {prn['aspect_residual']:.3f}")
    print(f"  prediction R² on the noisy synthetic: {1-((y0n-predict(TH,prn))**2).sum()/((y0n-y0n.mean())**2).sum():.4f}")

    # ═══ THE DAILY ARXIV SERIES, kerykeion sky ═══════════════════════════════════════════════
    D = os.path.expanduser("~/.artaquest-dev/artacomp/daily")
    daily = pd.read_csv(f"{D}/daily.csv", parse_dates=["date"]).set_index("date")
    rel = pd.read_csv(f"{D}/reliable_from.csv", parse_dates=["reliable_from"]).set_index("category")
    DAYS = np.array([d.date() for d in daily.index.to_pydatetime()])
    E = np.load(f"{D}/ephemeris_ker_1991_2026.npz"); e0 = dt.date.fromisoformat(str(E["d0"])); EB = list(E["bodies"])
    off = np.array([(d-e0).days for d in DAYS]); valid = (off >= 0) & (off < E["lon"].shape[0])
    BOD = ["sun","moon","mercury","venus","mars","jupiter","saturn","true_node"]
    SEL = [EB.index(b) for b in BOD]
    LON = np.zeros((len(DAYS), 8)); LON[valid] = E["lon"][off[valid]][:, SEL]; THALL = np.deg2rad(LON)
    from sklearn.metrics import roc_auc_score
    def peaks_labels(x):
        s = pd.Series(x).rolling(7, center=True, min_periods=1).mean().to_numpy()
        med = pd.Series(s).rolling(90, min_periods=30).median().shift(1).to_numpy()
        sd = pd.Series(s-np.nan_to_num(med, nan=s.mean())).rolling(90, min_periods=30).std().shift(1).to_numpy()
        thr = np.nan_to_num(med, nan=np.inf) + np.maximum(0.2*np.nan_to_num(med, nan=0), np.nan_to_num(sd, nan=np.inf))
        n=len(s); peak=np.zeros(n,bool)
        for i in range(3,n-3):
            if s[i] >= s[i-3:i+4].max() and s[i] > thr[i]: peak[i]=True
        lab=np.zeros(n,bool)
        for i in np.where(peak)[0]: lab[max(0,i-7):i]=True
        return lab.astype(int)
    print("\n═ daily arXiv: exact phasor per category, kerykeion sidereal, fit on first 80% of each series")
    print("  y = submissions / trailing-365d level (b carries the level, the arrows the timing); scored on the last 20%")
    rows=[]; PAR={}
    for cat in daily.columns:
        if cat not in rel.index: continue
        start = max(np.searchsorted(DAYS, rel.loc[cat,"reliable_from"].date()), int(np.argmax(valid)))
        x = daily[cat].to_numpy(float)[start:]
        if len(x) < 8*365: continue
        idx = np.arange(start, start+len(x)); cut = int(len(x)*0.8); te = np.arange(cut, len(x))
        lvl = pd.Series(x).rolling(365, min_periods=60).mean().shift(1).bfill().to_numpy(); xr = x/np.maximum(lvl,1e-9)
        TH = THALL[idx]; y = peaks_labels(x)
        prm = solve_phasor(TH[:cut], xr[:cut], ridge=1e-6)
        yh = predict(TH, prm); rise = np.diff(yh, prepend=yh[0])
        # held-out fit quality of the EXACT model on the ratio, vs predicting the train mean ratio
        r2 = 1 - ((xr[te]-yh[te])**2).sum()/((xr[te]-xr[:cut].mean())**2).sum()
        au = roc_auc_score(y[te], yh[te]) if len(set(y[te]))>1 else np.nan
        ar = roc_auc_score(y[te], rise[te]) if len(set(y[te]))>1 else np.nan
        # sun-only exact phasor as the seasonal control
        ps = solve_phasor(TH[:cut][:, [0]], xr[:cut], ridge=1e-6); ys = predict(TH[:, [0]], ps)
        aus = roc_auc_score(y[te], ys[te]) if len(set(y[te]))>1 else np.nan
        rows.append(dict(category=cat, days=len(x), b=prm["b"], feasible=prm["feasible"], aspect_residual=prm["aspect_residual"],
                         r2_heldout=r2, auc_level=au, auc_rise=ar, auc_sun_only=aus,
                         **{f"a_{n}": float(v) for n,v in zip(BOD, prm["a"])}, **{f"p_{n}": float(np.rad2deg(v)%360) for n,v in zip(BOD, prm["p"])}))
        PAR[cat] = dict(b=prm["b"], a=dict(zip(BOD, prm["a"].tolist())), p_deg=dict(zip(BOD, (np.rad2deg(prm["p"])%360).tolist())))
    df = pd.DataFrame(rows); df.to_csv(f"{D}/phasor_exact_results.csv", index=False); json.dump(PAR, open(f"{D}/phasor_exact_params.json","w"), indent=1)
    print(f"  {len(df)} categories · feasible (disc≥0) in {df.feasible.mean()*100:.0f}% · median aspect residual {df.aspect_residual.median():.3f}")
    print(f"  held-out R² of the exact model on the ratio: mean {df.r2_heldout.mean():+.4f} · median {df.r2_heldout.median():+.4f} · >0 in {(df.r2_heldout>0).mean()*100:.0f}%")
    print(f"  peak AUC · level {df.auc_level.mean():.4f} · rise {df.auc_rise.mean():.4f} · Sun-only phasor {df.auc_sun_only.mean():.4f}")
    print(f"  mean amplitudes a_i across categories:"); print("   " + "  ".join(f"{n} {df[f'a_{n}'].mean():.4f}" for n in BOD))
    print(f"  (b ≈ {df.b.mean():.3f}: the ratio's level; arrows are the modulation)")

# ═══ SECOND PASS: only bodies that complete cycles inside the fitting window ═══════════════════
# A body that moves less than one revolution in the fit span (Saturn 29y, the node 18.6y, Jupiter 12y
# on an 8-15y window) has cos/sin near-collinear with the constant: the free fit dumps the TREND onto
# its arrow, disc goes negative, the projection breaks (78% infeasible above). Cycle-complete bodies
# only — Sun, Moon, Mercury, Venus, Mars — are the ones a DAILY phasor can actually resolve.
if __name__ == "__main__":
    print("\n═ second pass: Sun, Moon, Mercury, Venus, Mars only (each completes ≥ several cycles in every window)")
    FAST = [0,1,2,3,4]; rows=[]
    for cat in daily.columns:
        if cat not in rel.index: continue
        start = max(np.searchsorted(DAYS, rel.loc[cat,"reliable_from"].date()), int(np.argmax(valid)))
        x = daily[cat].to_numpy(float)[start:]
        if len(x) < 8*365: continue
        idx = np.arange(start, start+len(x)); cut = int(len(x)*0.8); te = np.arange(cut, len(x))
        lvl = pd.Series(x).rolling(365, min_periods=60).mean().shift(1).bfill().to_numpy(); xr = x/np.maximum(lvl,1e-9)
        TH = THALL[idx][:, FAST]; y = peaks_labels(x)
        prm = solve_phasor(TH[:cut], xr[:cut], ridge=1e-4); yh = predict(TH, prm); rise = np.diff(yh, prepend=yh[0])
        r2 = 1 - ((xr[te]-yh[te])**2).sum()/((xr[te]-xr[:cut].mean())**2).sum()
        au = roc_auc_score(y[te], yh[te]) if len(set(y[te]))>1 else np.nan; ar = roc_auc_score(y[te], rise[te]) if len(set(y[te]))>1 else np.nan
        # circular-shift null for THIS category's exact model (2 draws)
        rng2 = np.random.RandomState(1); nulls=[]
        for _ in range(2):
            sh = rng2.randint(400, len(x)-400); yn = np.roll(y, sh); nulls.append(roc_auc_score(yn[te], yh[te]) if len(set(yn[te]))>1 else np.nan)
        rows.append(dict(category=cat, feasible=prm["feasible"], aspect_residual=prm["aspect_residual"], r2_heldout=r2, auc_level=au, auc_rise=ar, null=float(np.nanmean(nulls)),
                         **{f"a_{BOD[i]}": float(v) for i,v in zip(FAST, prm["a"])}, **{f"p_{BOD[i]}": float(np.rad2deg(v)%360) for i,v in zip(FAST, prm["p"])}, b=prm["b"]))
    df2 = pd.DataFrame(rows); df2.to_csv(f"{D}/phasor_exact_fast5.csv", index=False)
    print(f"  feasible in {df2.feasible.mean()*100:.0f}% · median aspect residual {df2.aspect_residual.median():.3f} · held-out R² mean {df2.r2_heldout.mean():+.4f} median {df2.r2_heldout.median():+.4f} · >0 in {(df2.r2_heldout>0).mean()*100:.0f}%")
    print(f"  peak AUC · level {df2.auc_level.mean():.4f} · rise {df2.auc_rise.mean():.4f} · shift-null {df2.null.mean():.4f}")
    print("  mean amplitudes: " + "  ".join(f"{BOD[i]} {df2[f'a_{BOD[i]}'].mean():.4f}" for i in FAST) + f"  · b {df2.b.mean():.3f}")
    print("  top 5 by held-out R²:"); print(df2.sort_values("r2_heldout", ascending=False).head(5)[["category","r2_heldout","auc_level","auc_rise","a_sun","a_moon","p_sun"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    json.dump({"all8": {"feasible_pct": round(float(df.feasible.mean()*100),1), "r2_median": round(float(df.r2_heldout.median()),4), "auc_level": round(float(df.auc_level.mean()),4), "auc_rise": round(float(df.auc_rise.mean()),4)},
               "fast5": {"feasible_pct": round(float(df2.feasible.mean()*100),1), "r2_median": round(float(df2.r2_heldout.median()),4), "r2_pos_pct": round(float((df2.r2_heldout>0).mean()*100),1), "auc_level": round(float(df2.auc_level.mean()),4), "auc_rise": round(float(df2.auc_rise.mean()),4), "null": round(float(df2.null.mean()),4),
                         "mean_a": {BOD[i]: round(float(df2[f'a_{BOD[i]}'].mean()),4) for i in FAST}}}, open(f"{D}/phasor_exact_summary.json","w"), indent=1)

# ═══ THIRD PASS: L2-regularised exact phasor, all 8 bodies, λ chosen per category on an inner wall ═══
# The closed form is unchanged: ĉ = (ΦᵀWΦ + λR)⁻¹ΦᵀWy with R = I minus the constant, then the exact
# projection to (b, aᵢ, pᵢ). λ picks how hard the slow arrows are shrunk toward zero — chosen on the
# LAST 25% OF THE TRAIN SPAN, so the held-out 20% is untouched. Grid spans six decades.
if __name__ == "__main__":
    print("\n═ third pass: L2 (ridge) exact phasor, all 8 bodies, λ chosen on an inner temporal split")
    LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]; rows=[]
    for cat in daily.columns:
        if cat not in rel.index: continue
        start = max(np.searchsorted(DAYS, rel.loc[cat,"reliable_from"].date()), int(np.argmax(valid)))
        x = daily[cat].to_numpy(float)[start:]
        if len(x) < 8*365: continue
        idx = np.arange(start, start+len(x)); cut = int(len(x)*0.8); te = np.arange(cut, len(x)); k = int(cut*0.75)
        lvl = pd.Series(x).rolling(365, min_periods=60).mean().shift(1).bfill().to_numpy(); xr = x/np.maximum(lvl,1e-9)
        TH = THALL[idx]; y = peaks_labels(x)
        # choose λ on the inner split (fit <k, judge k..cut) by held-out R² of the ratio
        best=None
        for lam in LAMS:
            pr = solve_phasor(TH[:k], xr[:k], ridge=lam); yh = predict(TH, pr)
            r2i = 1 - ((xr[k:cut]-yh[k:cut])**2).sum()/((xr[k:cut]-xr[:k].mean())**2).sum()
            if best is None or r2i > best[0]: best=(r2i, lam)
        lam = best[1]
        prm = solve_phasor(TH[:cut], xr[:cut], ridge=lam); yh = predict(TH, prm); rise = np.diff(yh, prepend=yh[0])
        r2 = 1 - ((xr[te]-yh[te])**2).sum()/((xr[te]-xr[:cut].mean())**2).sum()
        au = roc_auc_score(y[te], yh[te]) if len(set(y[te]))>1 else np.nan; ar = roc_auc_score(y[te], rise[te]) if len(set(y[te]))>1 else np.nan
        rng3 = np.random.RandomState(2); nulls=[]
        for _ in range(2):
            sh = rng3.randint(400, len(x)-400); yn = np.roll(y, sh); nulls.append(roc_auc_score(yn[te], yh[te]) if len(set(yn[te]))>1 else np.nan)
        rows.append(dict(category=cat, lam=lam, inner_r2=best[0], feasible=prm["feasible"], aspect_residual=prm["aspect_residual"], r2_heldout=r2,
                         auc_level=au, auc_rise=ar, null=float(np.nanmean(nulls)), b=prm["b"],
                         **{f"a_{n}": float(v) for n,v in zip(BOD, prm["a"])}, **{f"p_{n}": float(np.rad2deg(v)%360) for n,v in zip(BOD, prm["p"])}))
    df3 = pd.DataFrame(rows); df3.to_csv(f"{D}/phasor_exact_l2.csv", index=False)
    print(f"  λ chosen: " + ", ".join(f"{l:g}×{int((df3.lam==l).sum())}" for l in LAMS if (df3.lam==l).any()))
    print(f"  feasible in {df3.feasible.mean()*100:.0f}% · held-out R² mean {df3.r2_heldout.mean():+.4f} median {df3.r2_heldout.median():+.4f} · >0 in {(df3.r2_heldout>0).mean()*100:.0f}%")
    print(f"  peak AUC · level {df3.auc_level.mean():.4f} · rise {df3.auc_rise.mean():.4f} · shift-null {df3.null.mean():.4f}")
    print("  mean amplitudes: " + "  ".join(f"{n} {df3[f'a_{n}'].mean():.4f}" for n in BOD) + f"  · b {df3.b.mean():.3f}")
    print("  top 6 by held-out R²:"); print(df3.sort_values("r2_heldout", ascending=False).head(6)[["category","lam","r2_heldout","auc_level","auc_rise","a_sun","a_saturn","p_sun"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    S_ = json.load(open(f"{D}/phasor_exact_summary.json"))
    S_["l2_all8"] = {"feasible_pct": round(float(df3.feasible.mean()*100),1), "r2_mean": round(float(df3.r2_heldout.mean()),4), "r2_median": round(float(df3.r2_heldout.median()),4), "r2_pos_pct": round(float((df3.r2_heldout>0).mean()*100),1),
                     "auc_level": round(float(df3.auc_level.mean()),4), "auc_rise": round(float(df3.auc_rise.mean()),4), "null": round(float(df3.null.mean()),4),
                     "lam_hist": {str(l): int((df3.lam==l).sum()) for l in LAMS}, "mean_a": {n: round(float(df3[f'a_{n}'].mean()),4) for n in BOD}}
    json.dump(S_, open(f"{D}/phasor_exact_summary.json","w"), indent=1)
