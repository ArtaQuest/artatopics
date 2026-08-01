#!/usr/bin/env python3
"""30-yr EVIDENCE-EXPONENT ridge probe (operator "keep improving 30-yr AUC").

The improve pass found N^0.75 year-weighting lifts 30-yr AUC +0.65→+0.69 (√N was the noise-theory
choice; the long test rewards leaning harder onto the high-evidence modern years). Before adoption:
  1  map the ridge w = N^e, e ∈ {0.55..0.95} at the OUTER wall (1996-2025), 3 seeds each;
  2  HONESTY: pick e on the INNER wall (fit ≤1965, score 1966-1995) — adopt only if the inner-selected
     e lands on the same ridge (no test-set shopping);
  3  cross the ridge winner with detector k ∈ {2, 2.5} — do the two levers stack?

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/arxiv_30yr_wexp.py
"""
import importlib.util as u, json, os
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
S = _load("analysis/arxivtopics/arxiv_sweep_rosters_factors.py", "S")
I = _load("analysis/arxivtopics/arxiv_30yr_improve.py", "I30") if False else None  # improve runs its battery on import — reimplement
import torch as T
dev, ALL = S.dev, S.ALL_BODIES
Y, TV, tot, TH_ALL, Tn, n = S.Y, S.TV, S.tot, S.TH_ALL, S.Tn, S.n
W30, WIN = n - 30, n - 60
BASE = ["mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "node"]
inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))


def fit(kpow=2.0, wexp=0.5, wall=W30, seed=7, steps=9000, lr=2e-2):
    bi = [ALL.index(b) for b in BASE]; nb = len(bi)
    TH = TH_ALL[:, bi]; Ytf = np.sqrt(Y)
    T.manual_seed(seed)
    tb = lambda a: T.tensor(a.astype(np.float32), device=dev)
    cT, sT = tb(np.cos(TH).T), tb(np.sin(TH).T)
    tv = TV[:, :wall].astype(np.float32)
    wy = np.clip(tot[:wall], 1e-9, None) ** wexp
    Wm = tv * wy[None]; Wm = Wm / np.maximum(Wm.sum(1, keepdims=True), 1e-9); Wt = tb(Wm)
    vmean = (Ytf[:, :wall] * tv).sum(1) / np.maximum(tv.sum(1), 1.0)
    A0 = np.full((Tn, nb), -2.0, np.float32); A0[:, BASE.index("pluto")] = inv_sp(np.clip(vmean, 1e-3, None))
    Araw = T.tensor(A0, device=dev, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (Tn, nb, 1)).astype(np.float32) +
                 np.random.RandomState(seed).randn(Tn, nb, 2).astype(np.float32) * 0.01, device=dev, requires_grad=True)
    Bp = T.tensor(vmean.astype(np.float32), device=dev, requires_grad=True)
    Yt = tb(Ytf)
    opt = T.optim.Adam([Araw, U, Bp], lr=lr)
    def forward():
        p = T.atan2(U[:, :, 0], U[:, :, 1]); A = T.nn.functional.softplus(Araw)
        C = Bp[:, None] + (A * T.cos(p)) @ cT + (A * T.sin(p)) @ sT
        return T.clamp(C, min=1e-4) ** kpow + 1e-8
    best, stall, state = np.inf, 0, None
    for it in range(steps):
        e = T.sqrt(forward() + 1e-8)[:, :wall] - Yt[:, :wall]
        l = (e.abs() * Wt).sum() / Tn
        opt.zero_grad(); l.backward(); opt.step()
        if it % 200 == 199:
            lv = l.item()
            if lv < best - 1e-7: best, stall, state = lv, 0, [x.detach().clone() for x in [Araw, U, Bp]]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip([Araw, U, Bp], state): x.copy_(sv)
        return np.clip(forward().cpu().numpy(), 0, None)


def bench_at(Yh, wall, lo, hi):
    tvw = TV[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1, keepdims=True) / np.maximum(tvw.sum(1, keepdims=True), 1.0)
    skill = 1.0 - ((Y[:, lo:hi] - Yh[:, lo:hi]) ** 2).sum(1) / np.maximum(((Y[:, lo:hi] - mu) ** 2).sum(1), 1e-6)
    curve = [1.0 - ((Y[:, h] - Yh[:, h]) ** 2).sum() / max(((Y[:, h] - mu[:, 0]) ** 2).sum(), 1e-9) for h in range(lo, hi)]
    return float(np.median(skill)), float((skill > 0).mean() * 100), float(np.mean(curve))

ES = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
print(f"== ridge at the OUTER wall (1996-2025) · 3 seeds ==", flush=True)
outer_med = {}
for e in ES:
    aucs = []
    for sd in (7, 1, 2):
        s, p, a = bench_at(fit(wexp=e, seed=sd), W30, W30, n); aucs.append(a)
    outer_med[e] = float(np.median(aucs))
    print(f"  e={e:.2f}  outer AUC med {outer_med[e]:+.4f} [{min(aucs):+.4f}..{max(aucs):+.4f}]", flush=True)

print(f"== HONESTY: pick e on the INNER wall (fit ≤1965, score 1966-1995) ==", flush=True)
inner = {}
for e in ES + [0.5]:
    s, p, a = bench_at(fit(wexp=e, wall=WIN), WIN, WIN, W30)
    inner[e] = a
    print(f"  e={e:.2f}  inner AUC {a:+.4f}", flush=True)
e_star = max(inner, key=inner.get)
print(f"  INNER-selected e* = {e_star} (inner {inner[e_star]:+.4f}) → outer {outer_med.get(e_star, float('nan')):+.4f}", flush=True)

print(f"== cross: k × e* ==", flush=True)
for k in (2.0, 2.5):
    aucs = []
    for sd in (7, 1, 2):
        s, p, a = bench_at(fit(kpow=k, wexp=e_star, seed=sd), W30, W30, n); aucs.append((s, p, a))
    A_ = [x[2] for x in aucs]
    print(f"  k={k} e={e_star}: AUC med {np.median(A_):+.4f} [{min(A_):+.4f}..{max(A_):+.4f}] · "
          f"skill {np.median([x[0] for x in aucs]):+.4f} · {np.median([x[1] for x in aucs]):.1f}%>0", flush=True)

json.dump({"outer": outer_med, "inner": {str(k): v for k, v in inner.items()}, "e_star": e_star},
          open("analysis/arxivtopics/arxiv_30yr_wexp.json", "w"), indent=1)
print("WEXPDONE", flush=True)
