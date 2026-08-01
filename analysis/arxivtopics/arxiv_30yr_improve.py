#!/usr/bin/env python3
"""30-yr IMPROVE (operator 2026-07-24: "keep brainstorming and improve 30-yr AUC").

Attack the one structural finding of the three prior passes: detector power k trades pooled AUC
(big topics want k≈1) against median skill (typical topics want k≈2-3). NEW levers, all honest:

  A  PER-TOPIC k — each topic picks its own detector curvature k∈{1,1.5,2,2.5,3} on an INNER
     REHEARSAL WALL (fit ≤1965, validate 1966-1995 — no post-1995 data touches selection),
     then ONE refit ≤1995 with the chosen k, scored 1996-2025. Data picks, not a knob.
  B  PER-TOPIC BLEND — same inner-wall selection over w·relu1+(1−w)·relu2, w∈{0,¼,½,¾,1}.
  C  DRIFT ARROW — a secular term d·τ inside C (τ = standardized year): C = b + d·τ + Σaᵢcosφᵢ.
     Stays inside the square → still expands exactly (a "drift transit" 2b·d·τ). +1 param.
  D  EVIDENCE EXPONENT — N^0.25 / N^0.75 around the √N optimum.
  E  PER-TOPIC BEST-RESTART — 4 seeds, each topic keeps its lowest-TRAIN-loss restart (pure
     optimization, no test contact).
  F  Cross the winners; seeds on the top-3. Base to beat: relu2·7-body seed-median +0.637.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/arxiv_30yr_improve.py
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
W30, WIN = n - 30, n - 60                     # outer wall 1996-2025 · inner rehearsal wall 1966-1995
BASE = ["mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "node"]
inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
KS = [1.0, 1.5, 2.0, 2.5, 3.0]


def fit(kpow=2.0, drift=False, wexp=0.5, wall=W30, seed=7, steps=9000, lr=2e-2, want_loss=False):
    bi = [ALL.index(b) for b in BASE]; nb = len(bi)
    TH = TH_ALL[:, bi]
    Ytf = np.sqrt(Y)
    T.manual_seed(seed)
    tb = lambda a: T.tensor(a.astype(np.float32), device=dev)
    cT, sT = tb(np.cos(TH).T), tb(np.sin(TH).T)
    tau = tb(((np.arange(n) - 0.5 * wall) / max(wall, 1) * 2.0)[None, :])   # standardized year, train-centred
    tv = TV[:, :wall].astype(np.float32)
    ww = np.clip(tot[:wall], 1e-9, None)
    wy = ww ** wexp
    Wm = tv * wy[None]; Wm = Wm / np.maximum(Wm.sum(1, keepdims=True), 1e-9); Wt = tb(Wm)
    vmean = (Ytf[:, :wall] * tv).sum(1) / np.maximum(tv.sum(1), 1.0)
    A0 = np.full((Tn, nb), -2.0, np.float32); A0[:, BASE.index("pluto")] = inv_sp(np.clip(vmean, 1e-3, None))
    Araw = T.tensor(A0, device=dev, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (Tn, nb, 1)).astype(np.float32) +
                 np.random.RandomState(seed).randn(Tn, nb, 2).astype(np.float32) * 0.01, device=dev, requires_grad=True)
    Bp = T.tensor(vmean.astype(np.float32), device=dev, requires_grad=True)
    Dr = T.zeros(Tn, device=dev, requires_grad=drift)
    params = [Araw, U, Bp] + ([Dr] if drift else [])
    Yt = tb(Ytf)
    opt = T.optim.Adam(params, lr=lr)

    def forward():
        p = T.atan2(U[:, :, 0], U[:, :, 1]); A = T.nn.functional.softplus(Araw)
        C = Bp[:, None] + (A * T.cos(p)) @ cT + (A * T.sin(p)) @ sT
        if drift: C = C + Dr[:, None] * tau
        return T.clamp(C, min=1e-4) ** kpow + 1e-8

    best, stall, state = np.inf, 0, None
    for it in range(steps):
        e = T.sqrt(forward() + 1e-8)[:, :wall] - Yt[:, :wall]
        l = (e.abs() * Wt).sum() / Tn
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
        if want_loss:
            e = (T.sqrt(T.tensor(yh, device=dev) + 1e-8)[:, :wall] - Yt[:, :wall]).abs() * Wt
            return yh, e.sum(1).cpu().numpy()               # per-topic train loss (for restart pick)
    return yh


def per_topic_skill(Yh, wall, lo, hi):
    """Per-topic held-out SSE over [lo,hi) vs the valid-train-mean baseline fit at `wall`."""
    tvw = TV[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1, keepdims=True) / np.maximum(tvw.sum(1, keepdims=True), 1.0)
    sse = ((Y[:, lo:hi] - Yh[:, lo:hi]) ** 2).sum(1)
    base = np.maximum(((Y[:, lo:hi] - mu) ** 2).sum(1), 1e-9)
    return 1.0 - sse / base

res = {}
def score(tag, Yh, seed=7):
    s, p, a = S.bench(Yh, wall=W30)
    res.setdefault(tag, {})[seed] = (s, p, a)
    print(f"  {tag:34s} AUC {a:+.4f} · skill {s:+.4f} · {p:.1f}%>0", flush=True)
    return a

print(f"== 30-yr IMPROVE · outer wall {W30} (1996-2025) · inner wall {WIN} (1966-1995) ==", flush=True)
print("== phase A/B: fit every k at BOTH walls (seed 7) ==", flush=True)
inner = {k: fit(kpow=k, wall=WIN) for k in KS}              # for selection (≤1965)
outer = {k: fit(kpow=k, wall=W30) for k in KS}              # for prediction (≤1995)
for k in KS: score(f"fixed k={k}", outer[k])

print("== A) per-topic k via inner wall ==", flush=True)
val = {k: per_topic_skill(inner[k], WIN, WIN, W30) for k in KS}   # validated on 1966-1995 ONLY
pick = np.array([max(KS, key=lambda k: val[k][j]) for j in range(Tn)])
Yh = np.stack([outer[pick[j]][j] for j in range(Tn)])
from collections import Counter
print("  k histogram:", dict(sorted(Counter(pick).items())), flush=True)
score("A per-topic k (inner-wall)", Yh)

print("== B) per-topic blend relu1+relu2 via inner wall ==", flush=True)
WS = [0.0, 0.25, 0.5, 0.75, 1.0]
vin = {w: per_topic_skill(w * inner[1.0] + (1 - w) * inner[2.0], WIN, WIN, W30) for w in WS}
wpick = np.array([max(WS, key=lambda w: vin[w][j]) for j in range(Tn)])
Yb = np.stack([(wpick[j] * outer[1.0] + (1 - wpick[j]) * outer[2.0])[j] for j in range(Tn)])
print("  w histogram:", dict(sorted(Counter(wpick).items())), flush=True)
score("B per-topic blend (inner-wall)", Yb)
score("B fixed 50/50 blend", 0.5 * outer[1.0] + 0.5 * outer[2.0])

print("== C) drift arrow ==", flush=True)
score("C drift k=2", fit(kpow=2.0, drift=True))
print("== D) evidence exponent ==", flush=True)
for we in (0.25, 0.75): score(f"D w=N^{we}", fit(wexp=we))
print("== E) per-topic best-restart (4 seeds, by TRAIN loss) ==", flush=True)
fits, losses = zip(*[fit(kpow=2.0, seed=sd, want_loss=True) for sd in (7, 1, 2, 3)])
sel = np.argmin(np.stack(losses), 0)
score("E best-restart k=2", np.stack([fits[sel[j]][j] for j in range(Tn)]))

# F) seeds on the champion strategies (re-derive selection per seed — fully honest per run)
print("== F) seeds on per-topic-k and blend ==", flush=True)
for sd in (1, 2, 3):
    inn = {k: fit(kpow=k, wall=WIN, seed=sd) for k in (1.0, 2.0)}
    out_ = {k: fit(kpow=k, wall=W30, seed=sd) for k in (1.0, 2.0)}
    vi = {w: per_topic_skill(w * inn[1.0] + (1 - w) * inn[2.0], WIN, WIN, W30) for w in WS}
    wp = np.array([max(WS, key=lambda w: vi[w][j]) for j in range(Tn)])
    score("B per-topic blend (inner-wall)", np.stack([(wp[j] * out_[1.0] + (1 - wp[j]) * out_[2.0])[j] for j in range(Tn)]), seed=sd)
    score("B fixed 50/50 blend", 0.5 * out_[1.0] + 0.5 * out_[2.0], seed=sd)
    score("fixed k=1.0", out_[1.0], seed=sd)

med = {t: float(np.median([v[2] for v in res[t].values()])) for t in res}
print("\n  LEAGUE (30-yr AUC, seed-median):", flush=True)
for t in sorted(med, key=lambda t: -med[t]):
    aucs = sorted(v[2] for v in res[t].values()); sk = float(np.median([v[0] for v in res[t].values()]))
    pc = float(np.median([v[1] for v in res[t].values()]))
    rng = f"[{aucs[0]:+.4f}..{aucs[-1]:+.4f}]({len(aucs)}s)" if len(aucs) > 1 else "(1 seed)"
    print(f"    {med[t]:+.4f} {rng:26s} · skill {sk:+.4f} · {pc:.0f}%>0 · {t}", flush=True)
json.dump({t: med[t] for t in med}, open("analysis/arxivtopics/arxiv_30yr_improve.json", "w"), indent=1)
print("IMPROVEDONE", flush=True)
