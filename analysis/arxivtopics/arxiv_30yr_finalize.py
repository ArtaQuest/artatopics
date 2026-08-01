#!/usr/bin/env python3
"""30-yr FINALIZE ablation (operator 2026-07-24, 2nd pass): brainstorm + test the design choices the
first two sweeps did NOT cover, on the v7 base (rectified cosine-sum · 7 bodies drop-chiron · √N ·
angles-only · √-L1), at the 30-year wall. Novel axes:
  K   learned detector power  max(C,0)^k  — global-learned k (data picks the nonlinearity) · k=1.5/2.5 fixed
  L   robust loss on √-share  — L1 (base) · Huber(δ) · log-cosh   (does L1 still hold 30y out?)
  H2  2nd harmonic on the SLOW bodies only (pluto/neptune/uranus) — non-sinusoidal long arc, no all-body overfit
  T   learned dark threshold  max(C−τ,0)²  (per-topic floor: a field stays dark until C clears τ)
  W   training-window start    {1700 base · 1800 · 1900}  — does recent-only history forecast 1996-2025 better?
  E   SEED ENSEMBLE forecast   — average K seed-fits (parameter-free variance reduction); honest 30-yr AUC
Metric: honest 30-yr wall (pooled AUC + median skill + %>0), per-topic valid-train-mean baseline, seed-median.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/arxiv_30yr_finalize.py
"""
import importlib.util as u, json, os
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
S = _load("analysis/arxivtopics/arxiv_sweep_rosters_factors.py", "S")
import torch as T
dev, ALL = S.dev, S.ALL_BODIES
Y, TV, tot, TH_ALL, Tn, n = S.Y, S.TV, S.tot, S.TH_ALL, S.Tn, S.n
YEARS = [int(y) for y in [c for c in __import__("pandas").read_csv("analysis/citations/citations_received_yearly.csv").columns if c[0].isdigit()]]
W30 = n - 30
BASE = ["mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "node"]   # v7 roster (drop chiron)
SLOW = ["uranus", "neptune", "pluto"]
inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))


def fit(bodies=BASE, kpow=2.0, klearn=False, loss="l1", huber=0.3, h2=(), thresh=False,
        start=1700, wall=W30, seed=7, steps=9000, lr=2e-2, ret_params=False):
    bi = [ALL.index(b) for b in bodies]; nb = len(bi)
    TH = TH_ALL[:, bi]
    h2i = [bodies.index(b) for b in h2 if b in bodies]
    Ytf = np.sqrt(Y)
    T.manual_seed(seed)
    tb = lambda a: T.tensor(a.astype(np.float32), device=dev)
    cT = tb(np.cos(TH).T); sT = tb(np.sin(TH).T)
    c2 = tb(np.cos(2 * TH).T); s2 = tb(np.sin(2 * TH).T)
    tv = TV[:, :wall].astype(np.float32).copy()
    if start > 1700:
        cut = max(0, YEARS.index(start)); tv[:, :cut] = 0.0                 # mask years before `start`
    wy = np.sqrt(np.clip(tot[:wall], 0, None))
    Wm = tv * wy[None]; Wm = Wm / np.maximum(Wm.sum(1, keepdims=True), 1e-9); Wt = tb(Wm)
    vmean = (Ytf[:, :wall] * tv).sum(1) / np.maximum(tv.sum(1), 1.0)
    A0 = np.full((Tn, nb), -2.0, np.float32)
    if "pluto" in bodies: A0[:, bodies.index("pluto")] = inv_sp(np.clip(vmean, 1e-3, None))
    Araw = T.tensor(A0, device=dev, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (Tn, nb, 1)).astype(np.float32) +
                 np.random.RandomState(seed).randn(Tn, nb, 2).astype(np.float32) * 0.01, device=dev, requires_grad=True)
    Bp = T.tensor(vmean.astype(np.float32), device=dev, requires_grad=True)
    params = [Araw, U, Bp]
    A2 = T.tensor(np.full((Tn, len(h2i)), -4.0, np.float32), device=dev, requires_grad=bool(h2i)) if h2i else None
    U2 = T.tensor(np.tile([0.0, 1.0], (Tn, max(len(h2i), 1), 1)).astype(np.float32), device=dev, requires_grad=bool(h2i)) if h2i else None
    if h2i: params += [A2, U2]
    Tau = T.zeros(Tn, device=dev, requires_grad=True) if thresh else None
    if thresh: params += [Tau]
    kr = T.tensor(float(inv_sp(np.array([kpow]))[0]), device=dev, requires_grad=klearn) if klearn else None
    if klearn: params += [kr]
    Yt = tb(Ytf)
    opt = T.optim.Adam(params, lr=lr)

    def forward():
        p = T.atan2(U[:, :, 0], U[:, :, 1]); A = T.nn.functional.softplus(Araw)
        cp, sp = A * T.cos(p), A * T.sin(p)
        C = Bp[:, None] + cp @ cT + sp @ sT
        if h2i:
            idx = T.tensor(h2i, device=dev)
            p2 = T.atan2(U2[:, :, 0], U2[:, :, 1]); A2s = T.nn.functional.softplus(A2)
            C = C + (A2s * T.cos(p2)) @ c2[idx] + (A2s * T.sin(p2)) @ s2[idx]
        if thresh: C = C - Tau[:, None]
        k = T.nn.functional.softplus(kr) if klearn else kpow
        return T.clamp(C, min=1e-4) ** k + 1e-8

    def rootshare(l): return T.sqrt(l + 1e-8)
    best, stall, state = np.inf, 0, None
    for it in range(steps):
        e = rootshare(forward())[:, :wall] - Yt[:, :wall]
        if loss == "l1": pen = e.abs()
        elif loss == "l2": pen = e ** 2
        elif loss == "huber":
            a = e.abs(); pen = T.where(a < huber, 0.5 * e ** 2 / huber, a - 0.5 * huber)
        elif loss == "logcosh": pen = T.log(T.cosh(T.clamp(e, -12, 12)))
        l = (pen * Wt).sum() / Tn
        opt.zero_grad(); l.backward(); opt.step()
        if it % 200 == 199:
            lv = l.item()
            if lv < best - 1e-7: best, stall, state = lv, 0, [x.detach().clone() for x in params]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip(params, state): x.copy_(sv)
        yh = np.clip(forward().cpu().numpy(), 0, None)
    if klearn:
        with T.no_grad(): kval = float(T.nn.functional.softplus(kr))
        return (yh, kval) if ret_params else yh
    return yh


res, KW = {}, {}
def run(tag, **kw):
    yh = fit(**kw)
    if isinstance(yh, tuple): yh, kv = yh; tag = f"{tag} (k={kv:.2f})"
    s, p, a = S.bench(yh, wall=kw.get("wall", W30))
    res.setdefault(tag, {})[kw.get("seed", 7)] = (s, p, a); KW[tag] = {x: v for x, v in kw.items() if x != "seed"}
    print(f"  {tag:34s} AUC {a:+.4f} · skill {s:+.4f} · {p:.1f}%>0", flush=True)

print(f"== 30-yr FINALIZE · wall {W30} · base = relu2/7-body(drop-chiron)/√N/angles ==", flush=True)
run("BASE relu2 k=2")
print("== K) learned / fixed detector power ==", flush=True)
run("K learn-k", klearn=True)
run("K k=1.5", kpow=1.5); run("K k=2.5", kpow=2.5)
print("== L) robust loss ==", flush=True)
run("L huber.3", loss="huber", huber=0.3); run("L huber.6", loss="huber", huber=0.6); run("L logcosh", loss="logcosh")
print("== H2) slow-body 2nd harmonic ==", flush=True)
run("H2 slow(u/n/p)", h2=SLOW); run("H2 pluto+nep", h2=["pluto", "neptune"])
print("== T) learned dark threshold ==", flush=True)
run("T thresh")
print("== W) training-window start ==", flush=True)
run("W 1800", start=1800); run("W 1900", start=1900)

# seed robustness on the top-5 by seed-7 AUC
lead = sorted(res, key=lambda t: -res[t][7][2])[:5]
print("== SEEDS 1,2,3 on top-5 ==", flush=True)
for t in lead:
    for sd in (1, 2, 3): run(t, seed=sd, **KW[t])

# E) SEED ENSEMBLE of the base — average K seed-fits behind the wall, honest 30-yr AUC
print("== E) seed ensemble (base, honest 30-yr) ==", flush=True)
for K in (4, 8):
    yhs = [fit(seed=sd, wall=W30) for sd in range(1, K + 1)]
    ens = np.mean(yhs, 0)
    s, p, a = S.bench(ens, wall=W30)
    res[f"E ensemble-{K}"] = {0: (s, p, a)}; KW[f"E ensemble-{K}"] = {}
    print(f"  ensemble-{K:<2d}                        AUC {a:+.4f} · skill {s:+.4f} · {p:.1f}%>0", flush=True)

med = {t: float(np.median([v[2] for v in res[t].values()])) for t in res}
league = sorted(med, key=lambda t: -med[t])
print("\n  LEAGUE (30-yr AUC, seed-median):", flush=True)
for t in league:
    aucs = sorted(v[2] for v in res[t].values()); sk = float(np.median([v[0] for v in res[t].values()]))
    rng = f"[{aucs[0]:+.4f}..{aucs[-1]:+.4f}]" if len(aucs) > 1 else "(1 seed)      "
    print(f"    {med[t]:+.4f} {rng} · skill {sk:+.4f} · {t}", flush=True)
base = med.get("BASE relu2 k=2", -9)
print(f"\n  BASE (relu2 k=2) seed-median AUC {base:+.4f}; improvements over base by >0.004 (beyond seed noise):", flush=True)
for t in league:
    if med[t] > base + 0.004 and t != "BASE relu2 k=2":
        print(f"    +{med[t]-base:.4f}  {t}", flush=True)
json.dump({"base": base, "league": [(t, med[t]) for t in league]}, open("analysis/arxivtopics/arxiv_30yr_finalize.json", "w"), indent=1)
print("FINALIZEDONE", flush=True)
