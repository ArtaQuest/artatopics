#!/usr/bin/env python3
"""30-YEAR-AUC exhaustive ablation & brainstorm (operator 2026-07-24): switch the headline to the
30-year wall (fit ≤1995, forecast 1996-2025 — 30 years of pure ephemeris across the internet + deep-
learning era) and search EVERY design decision for the best 30-yr AUC.

Shares data with arxiv_sweep_rosters_factors (independent yearly record, per-topic non-zero mask, √N).
NEW axes vs the 12-yr sweep — the ones a long horizon actually turns:
  DETECTOR g(C), C=b+Σaᵢcos(θᵢ−pᵢ):  env(C²+S²) · sq(C²) · relu1 max(C,0) · relu1_5 · relu2 max(C,0)²
                                      (record) · relu3 · softplus2 · exp(C)   — the nonlinearity is the
                                      biggest long-horizon lever: spiky detectors overfit, smooth ones
                                      generalise; the rectifier floors long dark stretches.
  ROSTER by SPEED:  record-8 · slow5 (ur/ne/pl/node/chiron) · slow6 (+saturn) · outer4 (ur/ne/pl/chiron)
                    · 7-no-mars · +sun · all-11  — fast bodies (mars 1.9y, jupiter 12y) complete many
                    cycles in 30y; do they help or just average to noise?
  PER-BODY LEAVE-ONE-OUT on record-8 — which body carries the 30-year signal.
  HARMONICS {1 · 1+2 · 1+2+3} · LOSS SPACE {√ · log1p · linear} · DISTANCE {none · 1/r · 1/r²}
  EVIDENCE WEIGHT exponent {0 unweighted · 0.5 √N · 1 N · logN}.
Metric: the honest 30-year wall (skill median / %>0 / pooled AUC over horizons 1..30),
per-topic valid-train-mean baseline. Winner = best SEED-MEDIAN 30-yr AUC.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/arxiv_30yr_ablation.py
"""
import importlib.util as u, itertools, json, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
S = _load("analysis/arxivtopics/arxiv_sweep_rosters_factors.py", "S")
import torch as T
dev, ALL, REC = S.dev, S.ALL_BODIES, S.RECORD
Y, TV, tot, TH_ALL, R_ALL, RDOT, Tn, n = S.Y, S.TV, S.tot, S.TH_ALL, S.R_ALL, S.RDOT, S.Tn, S.n
W30 = n - 30
SLOW5 = ["uranus", "neptune", "pluto", "node", "chiron"]
SLOW6 = ["saturn"] + SLOW5
OUTER4 = ["uranus", "neptune", "pluto", "chiron"]
inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))


def fit(det="relu2", bodies=None, harm=1, space="sqrt", lk="l1", fmode="none", wexp=0.5,
        wall=W30, seed=7, steps=8000, lr=2e-2):
    bods = bodies or REC
    bi = [ALL.index(b) for b in bods]; nb = len(bi)
    TH = TH_ALL[:, bi]; F = S.make_F(bi, fmode)
    tf = {"sqrt": np.sqrt, "log1p": np.log1p, "linear": lambda x: x}[space]
    Ytf = tf(Y)
    T.manual_seed(seed)
    def tb(a): return T.tensor(a.astype(np.float32), device=dev)
    cH = [tb((F * np.cos((h + 1) * TH)).T) for h in range(harm)]
    sH = [tb((F * np.sin((h + 1) * TH)).T) for h in range(harm)]
    tv = TV[:, :wall].astype(np.float32)
    ww = np.clip(tot[:wall], 1e-9, None)
    wy = {0.0: np.ones_like(ww), 0.5: np.sqrt(ww), 1.0: ww, -1.0: np.log1p(ww)}[wexp]
    Wm = tv * wy[None]; Wm = Wm / np.maximum(Wm.sum(1, keepdims=True), 1e-9); Wt = tb(Wm)
    vmean = (Ytf[:, :wall] * tv).sum(1) / np.maximum(tv.sum(1), 1.0)
    A0 = np.full((Tn, nb), -2.0, np.float32)
    if "pluto" in bods: A0[:, bods.index("pluto")] = inv_sp(np.clip(vmean, 1e-3, None))
    Araw = [T.tensor(A0 if h == 0 else np.full((Tn, nb), -4.0, np.float32), device=dev, requires_grad=True)
            for h in range(harm)]
    U = [T.tensor(np.tile([0.0, 1.0], (Tn, nb, 1)).astype(np.float32) +
                  np.random.RandomState(seed + h).randn(Tn, nb, 2).astype(np.float32) * 0.01,
                  device=dev, requires_grad=True) for h in range(harm)]
    Bp = T.tensor(vmean.astype(np.float32), device=dev, requires_grad=True)
    params = Araw + U + [Bp]
    Yt = tb(Ytf)
    opt = T.optim.Adam(params, lr=lr)

    def real_imag():
        C = Bp[:, None].clone(); Sc = T.zeros_like(C)
        for h in range(harm):
            p = T.atan2(U[h][:, :, 0], U[h][:, :, 1]); A = T.nn.functional.softplus(Araw[h])
            cp, sp = A * T.cos(p), A * T.sin(p)
            C = C + cp @ cH[h] + sp @ sH[h]
            Sc = Sc + cp @ sH[h] - sp @ cH[h]
        return C, Sc

    def forward():
        C, Sc = real_imag()
        if det == "env":   return C ** 2 + Sc ** 2 + 1e-8
        if det == "sq":    return C ** 2 + 1e-8
        if det == "relu1": return T.clamp(C, min=1e-4)
        if det == "relu1_5": return T.clamp(C, min=1e-4) ** 1.5 + 1e-8
        if det == "relu2": return T.clamp(C, min=1e-4) ** 2 + 1e-8
        if det == "relu3": return T.clamp(C, min=1e-4) ** 3 + 1e-8
        if det == "softplus2": return T.nn.functional.softplus(C) ** 2 + 1e-8
        if det == "exp":   return T.exp(T.clamp(C, -6, 6))
        raise ValueError(det)

    fl = {"sqrt": lambda l: T.sqrt(l + 1e-8), "log1p": lambda l: T.log1p(T.clamp(l, min=0)),
          "linear": lambda l: l}[space]
    best, stall, state = np.inf, 0, None
    for it in range(steps):
        e = fl(forward())[:, :wall] - Yt[:, :wall]
        ew = (e.abs() if lk == "l1" else e ** 2) * Wt
        loss = ew.sum() / Tn
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            l = loss.item()
            if l < best - 1e-7: best, stall, state = l, 0, [x.detach().clone() for x in params]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip(params, state): x.copy_(sv)
        return np.clip(forward().cpu().numpy(), 0, None)


res = {}; KW = {}
def run(tag, **kw):
    s, p, a = S.bench(fit(**kw), wall=kw.get("wall", W30))
    res.setdefault(tag, {})[kw.get("seed", 7)] = (s, p, a)
    KW[tag] = {x: v for x, v in kw.items() if x != "seed"}
    print(f"  {tag:40s} skill {s:+.4f} ({p:.1f}%>0) · AUC {a:+.4f}", flush=True)
    return a

if __name__ == "__main__":
    print(f"== 30-YEAR ABLATION · {Tn}×{n} · wall {W30} (fit ≤1995, forecast 1996-2025) ==", flush=True)
    print("== A) DETECTOR NONLINEARITY (record-8) ==", flush=True)
    for d in ("env", "sq", "relu1", "relu1_5", "relu2", "relu3", "softplus2", "exp"):
        run(f"A det={d}", det=d)
    print("== B) ROSTER BY SPEED (relu2) ==", flush=True)
    for nm, r in (("slow5", SLOW5), ("slow6", SLOW6), ("outer4", OUTER4), ("no-mars7", [b for b in REC if b != "mars"]),
                  ("+sun9", ["sun"] + REC), ("all11", ALL)):
        run(f"B roster={nm}", bodies=r)
    print("== C) PER-BODY LEAVE-ONE-OUT (relu2, record-8) ==", flush=True)
    for b in REC:
        run(f"C  -{b}", bodies=[x for x in REC if x != b])
    print("== D) HARMONICS (relu2, record-8) ==", flush=True)
    for h in (2, 3):
        run(f"D harm={h}", harm=h)
    print("== E) LOSS SPACE × DISTANCE (relu2, record-8) ==", flush=True)
    for sp in ("log1p", "linear"):
        run(f"E space={sp}", space=sp)
    for fm in ("1/r", "1/r2"):
        run(f"E dist={fm}", fmode=fm)
    print("== F) EVIDENCE WEIGHT (relu2, record-8) ==", flush=True)
    for we, nm in ((0.0, "unweighted"), (1.0, "N"), (-1.0, "logN")):
        run(f"F w={nm}", wexp=we)

    # seed-robustness on the current leaders (add seeds 1,2,3)
    lead = sorted(res, key=lambda t: -res[t][7][2])[:5]
    print(f"== G) SEEDS 1,2,3 on top-5 ==", flush=True)
    for t in lead:
        for sd in (1, 2, 3):
            run(t, seed=sd, **KW[t])

    med = {t: float(np.median([v[2] for v in res[t].values()])) for t in res}
    league = sorted(med, key=lambda t: (-med[t], -float(np.median([v[0] for v in res[t].values()]))))
    print("\n  LEAGUE (30-yr AUC — seed-median where >1 seed):", flush=True)
    for t in league:
        aucs = sorted(v[2] for v in res[t].values()); ns = len(aucs)
        tag = f"med {med[t]:+.4f} [{aucs[0]:+.4f}..{aucs[-1]:+.4f}] ({ns} seeds)" if ns > 1 else f"    {aucs[0]:+.4f}          (1 seed)"
        sk = float(np.median([v[0] for v in res[t].values()]))
        print(f"    {tag} · skill {sk:+.4f} · {t}", flush=True)
    WIN = league[0]
    print(f"\n  WINNER (30-yr): {WIN}  med {med[WIN]:+.4f}  cfg {KW[WIN]}", flush=True)
    json.dump({"winner": WIN, "cfg": KW[WIN], "median30": med[WIN],
               "league": [(t, med[t], KW.get(t, {})) for t in league[:12]]},
              open("analysis/arxivtopics/arxiv_30yr_winner.json", "w"), indent=1, default=str)
    print("A0DONE", flush=True)
