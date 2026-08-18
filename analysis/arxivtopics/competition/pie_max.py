#!/usr/bin/env python3
"""MAX OUT the year->pie task, KL to the true pie, honest selection (2026-08-18).

Every model gives, for a year t, a distribution p(.|t) over 251 fields. Score = mean KL(true||p).
Selection is on ROLLING INNER WALLS inside train (fit < w, judge w..w+8, for w = 1960,68,76,84),
never on the 1992-2025 held-out years, which are touched exactly once at the end.

Members (all in LOG-space, all built to be shrunk toward climatology):
  clim       the train-window mean pie                                          — the wall
  trend      per-field linear trend in log-share, damped, from the last K years  — the drift ceiling
  sky_rank1  softmax with a SHARED sky signal: b_j + g_j * <w, sky(t)>           — 251+251+D params
  sky_slow   sky_rank1 restricted to Saturn/Uranus/Neptune/Pluto first harmonics
  sky_full   the free per-field softmax of pie_softmax.py, ridge chosen on walls  — the over-fitter
Blend: log p = log clim_or_trend + sum_k lam_k * (log member_k - log clim), lam on the walls.

  python3 analysis/arxivtopics/competition/pie_max.py
"""
import os, sys, json, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import arxiv_fit as af

names, Yv, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
tv = af.META["topic_valid"]
years = np.array([int(y) for y in labels]); J, n = Yv.shape
usable = np.where(tv.sum(0) >= 20)[0]
T = np.arange(usable[0], usable[-1] + 1)
S = np.clip(Yv, 0, None) * tv; S = S / np.maximum(S.sum(0, keepdims=True), 1e-12)
cut = int(len(T)*0.8); TR, TE = T[:cut], T[cut:]
Y0 = int(years[0])
def yi(y): return y - Y0
print(f"train {years[TR[0]]}-{years[TR[-1]]} · test {years[TE[0]]}-{years[TE[-1]]}", flush=True)
EPS = 1e-6
def kl(P, Q):
    """mean over columns of KL(Q||P); P predicted, Q true. columns are years."""
    Pn = np.clip(P, EPS, None); Pn /= Pn.sum(0, keepdims=True)
    Qn = np.clip(Q, 0, None); Qn /= Qn.sum(0, keepdims=True)
    return float(np.mean((Qn * (np.log(np.maximum(Qn, 1e-12)) - np.log(Pn))).sum(0)))
def softmax_cols(Z):
    Z = Z - Z.max(0, keepdims=True); P = np.exp(Z); return P / P.sum(0, keepdims=True)

PAIRS = [(i,k) for i in range(7) for k in range(i+1,7)]
def sky(idx, bodies=range(7), harm=1, pairs=True):
    C = []
    for h in range(1, harm+1):
        for i in bodies: C += [np.cos(h*TH[idx,i]), np.sin(h*TH[idx,i])]
    if pairs:
        for i,k in PAIRS:
            if i in bodies and k in bodies: d = TH[idx,i]-TH[idx,k]; C += [np.cos(d), np.sin(d)]
    return np.stack(C, 1)

# ── members ──────────────────────────────────────────────────────────────────────────────────
def m_clim(fit_idx, pred_idx, K=None):
    src = fit_idx if K is None else fit_idx[-K:]
    c = S[:, src].mean(1); c = c/c.sum()
    return np.repeat(c[:,None], len(pred_idx), 1)
def m_trend(fit_idx, pred_idx, K=20, phi=0.9):
    src = fit_idx[-K:]; L = np.log(np.clip(S[:, src], EPS, None))
    x = np.arange(len(src), dtype=float); xm = x.mean()
    slope = ((x-xm)[None,:]*(L-L.mean(1,keepdims=True))).sum(1)/((x-xm)**2).sum()
    last = L[:, -1]
    out = []
    for h, t in enumerate(pred_idx, start=1):
        step = phi*(1-phi**h)/(1-phi)          # damped cumulative horizon
        out.append(last + slope*step)
    return softmax_cols(np.stack(out, 1))
def m_sky_full(fit_idx, pred_idx, harm, lam, iters=2500, lr=0.5):
    Ftr, Fte = sky(fit_idx, harm=harm), sky(pred_idx, harm=harm)
    mu, sd = Ftr.mean(0), Ftr.std(0)+1e-9; Ftr=(Ftr-mu)/sd; Fte=(Fte-mu)/sd
    Ftr = np.hstack([np.ones((len(fit_idx),1)), Ftr]); Fte = np.hstack([np.ones((len(pred_idx),1)), Fte])
    Str = S[:, fit_idx]; clim = Str.mean(1); clim /= clim.sum()
    B = np.zeros((J, Ftr.shape[1])); B[:,0] = np.log(np.maximum(clim, EPS))
    for _ in range(iters):
        P = softmax_cols(B @ Ftr.T); G = (P - Str) @ Ftr / len(fit_idx); G[:,1:] += lam*B[:,1:]; B -= lr*G
    return softmax_cols(B @ Fte.T)
def m_sky_rank1(fit_idx, pred_idx, bodies=range(7), harm=1, lam=1e-3, iters=3000, lr=0.3):
    """b_j + g_j * s(t),  s(t) = <w, sky(t)>: ONE shared sky signal, per-field gain. Alternating GD."""
    Ftr, Fte = sky(fit_idx, bodies, harm), sky(pred_idx, bodies, harm)
    mu, sd = Ftr.mean(0), Ftr.std(0)+1e-9; Ftr=(Ftr-mu)/sd; Fte=(Fte-mu)/sd
    Str = S[:, fit_idx]; clim = Str.mean(1); clim /= clim.sum()
    b = np.log(np.maximum(clim, EPS)); g = np.zeros(J); w = np.random.RandomState(0).randn(Ftr.shape[1])*0.01
    for _ in range(iters):
        s = Ftr @ w                                   # (T,)
        Z = b[:,None] + g[:,None]*s[None,:]; P = softmax_cols(Z); R = P - Str   # (J,T)
        gb = R.mean(1); gg = (R*s[None,:]).mean(1) + lam*g
        gw = (R*g[:,None]).sum(0) @ Ftr / len(fit_idx) + lam*w
        b -= lr*gb; g -= lr*gg; w -= lr*gw
    return softmax_cols(b[:,None] + g[:,None]*(Fte @ w)[None,:])

# ── walls ─────────────────────────────────────────────────────────────────────────────────────
WALLS = [1960, 1968, 1976, 1984]; H = 8
def wall_sets(w):
    fit = TR[years[TR] < w]; jud = TR[(years[TR] >= w) & (years[TR] < w+H)]
    return fit, jud
def eval_walls(fn):
    sc = []
    for w in WALLS:
        fit, jud = wall_sets(w)
        sc.append(kl(fn(fit, jud), S[:, jud]))
    return float(np.mean(sc)), sc
print("\n— members on the inner walls (mean KL over walls; lower is better):", flush=True)
res = {}
res["clim"] = eval_walls(m_clim); print(f"  clim (all history)         {res['clim'][0]:.4f}", flush=True)
for K in (10, 20, 30):
    res[f"clim{K}"] = eval_walls(lambda f,p: m_clim(f,p,K)); print(f"  clim (last {K}y)            {res[f'clim{K}'][0]:.4f}", flush=True)
for K, phi in ((10,0.8),(20,0.9),(20,0.95),(30,0.9)):
    res[f"trend{K}_{phi}"] = eval_walls(lambda f,p: m_trend(f,p,K,phi)); print(f"  trend K={K} phi={phi}       {res[f'trend{K}_{phi}'][0]:.4f}", flush=True)
for harm, lam in ((1,0.01),(1,0.1),(2,0.1)):
    res[f"skyfull_h{harm}_l{lam}"] = eval_walls(lambda f,p: m_sky_full(f,p,harm,lam)); print(f"  sky_full h{harm} ridge {lam}    {res[f'skyfull_h{harm}_l{lam}'][0]:.4f}", flush=True)
for tag, bod, lam in (("all",range(7),1e-3),("slow",[2,3,4,5],1e-3),("slow_l1e-2",[2,3,4,5],1e-2),("outer",[3,4,5],1e-3)):
    res[f"rank1_{tag}"] = eval_walls(lambda f,p: m_sky_rank1(f,p,bod,1,lam)); print(f"  sky_rank1 {tag:<11}         {res[f'rank1_{tag}'][0]:.4f}", flush=True)

# ── the blend: pick base + shrink weights on the walls ────────────────────────────────────────
best_base = min([k for k in res if k.startswith(("clim","trend"))], key=lambda k: res[k][0])
print(f"\n  best non-sky base on the walls: {best_base} ({res[best_base][0]:.4f})", flush=True)
def base_fn(k):
    if k.startswith("clim"): K = int(k[4:]) if len(k)>4 else None; return lambda f,p: m_clim(f,p,K)
    K, phi = k[5:].split("_"); return lambda f,p: m_trend(f,p,int(K),float(phi))
BASE = base_fn(best_base)
sky_members = {"rank1_slow": lambda f,p: m_sky_rank1(f,p,[2,3,4,5],1,1e-3),
               "rank1_all":  lambda f,p: m_sky_rank1(f,p,range(7),1,1e-3),
               "skyfull_h1": lambda f,p: m_sky_full(f,p,1,0.1)}
# precompute member preds per wall
cache = {w: {} for w in WALLS}
for w in WALLS:
    fit, jud = wall_sets(w); cache[w]["base"] = BASE(fit, jud); cache[w]["clim"] = m_clim(fit, jud)
    for k, fn in sky_members.items(): cache[w][k] = fn(fit, jud)
def blend(w, lams):
    c = cache[w]; L = np.log(np.clip(c["base"], EPS, None))
    for k, lam in lams.items(): L = L + lam*(np.log(np.clip(c[k],EPS,None)) - np.log(np.clip(c["clim"],EPS,None)))
    return softmax_cols(L)
grid = [0.0, 0.25, 0.5, 0.75, 1.0]
best = None
for l1, l2, l3 in itertools.product(grid, grid, grid):
    lams = {"rank1_slow": l1, "rank1_all": l2, "skyfull_h1": l3}
    sc = float(np.mean([kl(blend(w, lams), S[:, wall_sets(w)[1]]) for w in WALLS]))
    if best is None or sc < best[0]: best = (sc, lams)
print(f"  walls choose sky shrink weights {best[1]} → {best[0]:.4f} (base alone {res[best_base][0]:.4f})", flush=True)

# ── ONE shot at 1992-2025 ─────────────────────────────────────────────────────────────────────
print("\n— the held-out years 1992-2025 (KL to the true pie):", flush=True)
fitF = TR
final = {"clim (all)": m_clim(fitF, TE), "best base": BASE(fitF, TE)}
climF = m_clim(fitF, TE); baseF = BASE(fitF, TE)
Lf = np.log(np.clip(baseF, EPS, None))
for k, fn in sky_members.items():
    Pk = fn(fitF, TE); final[k+" alone"] = Pk
    if best[1][k] > 0: Lf = Lf + best[1][k]*(np.log(np.clip(Pk,EPS,None)) - np.log(np.clip(climF,EPS,None)))
final["THE BLEND (wall-selected)"] = softmax_cols(Lf)
out = {}
for k, P in final.items():
    v = kl(P, S[:, TE]); out[k] = round(v, 4); print(f"  {k:<28} {v:.4f}", flush=True)
np.save(os.path.expanduser("~/.artaquest-dev/artacomp/scidist3/pred_blend.npy"), final["THE BLEND (wall-selected)"])
np.save(os.path.expanduser("~/.artaquest-dev/artacomp/scidist3/pred_base.npy"), baseF)
json.dump({"walls": {k: round(v[0],4) for k,v in res.items()}, "best_base": best_base,
           "shrink": best[1], "held_out": out}, open(os.path.expanduser("~/.artaquest-dev/artacomp/scidist3/pie_max.json"),"w"), indent=1)
