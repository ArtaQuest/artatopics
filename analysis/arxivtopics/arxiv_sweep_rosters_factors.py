#!/usr/bin/env python3
"""ROSTER × DISTANCE-FACTOR × MODEL sweep (operator 2026-07-24) on the independent yearly record.

Base = the model of record: independent per-topic receivers (no pie/tides), yearly 1700-2025,
per-topic non-zero-suffix mask, √N year weights, L1 on √y, angles-only, 8 bodies. This sweep varies:
  ROSTERS  — record 8; add venus / mercury / sun back; all 11; minus mars (7)
  FACTORS  — none · 1/r · 1/r² · 1/r³ (the tidal shape; constant masses are absorbed by aᵢ) ·
             sgn(−ṙ)/r · sgn(−ṙ) alone   (ṙ from centred yearly differences of the ephemeris r)
  MODELS   — record |·|² · cosine-sum (no magnitude) · log1p space · MSE loss · 2nd harmonic (adds
             cos(2θᵢ−p₂ᵢ) terms, +nb params) · tides2×pie reference (cross-sectional, for context)
Metric: the honest 12-year wall (skill median / %>0 / AUC), per-topic valid-train-mean baseline.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/arxiv_sweep_rosters_factors.py
"""
import importlib.util as u, os
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
p2 = _load("analysis/adstopics/astro_phasor2.py", "p2")
import torch as T
dev = "mps" if T.backends.mps.is_available() else "cpu"

ALL_BODIES = [b for b in p2.BODIES if b != "moon"]        # 11: sun..chiron
RECORD = ["mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "node", "chiron"]

mx = pd.read_csv("analysis/citations/citations_received_yearly.csv")
years = [c for c in mx.columns if c[0].isdigit()]
names = list(mx.subfield)
V = mx[years].to_numpy(float)
tot = V.sum(0)
Y = 100.0 * V / np.maximum(tot[None, :], 1.0)
Z = V > 0
TV = np.ones_like(Z, bool)
for i in range(len(years) - 2, -1, -1): TV[:, i] = Z[:, i] & TV[:, i + 1]
Tn, n = Y.shape
W12 = n - 12
Ysq = np.sqrt(Y)
inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))

E = pd.read_csv("analysis/arxivtopics/_ephemeris_yearly.csv").set_index("Time")
E.index = E.index.astype(str)
TH_ALL = np.stack([np.deg2rad(E[f"{b}_lon"].loc[years].to_numpy(float)) for b in ALL_BODIES], 1)
R_ALL = np.stack([E[f"{b}_dist"].loc[years].to_numpy(float) for b in ALL_BODIES], 1)
RDOT = np.gradient(R_ALL, axis=0)                          # centred yearly differences


def make_F(bi, mode):
    R = R_ALL[:, bi]
    if mode == "none": return np.ones_like(R)
    if mode == "sgn":  F = -np.sign(RDOT[:, bi])
    elif mode == "sgn/r": F = -np.sign(RDOT[:, bi]) / R
    else:
        k = {"1/r": 1, "1/r2": 2, "1/r3": 3}[mode]
        F = 1.0 / R ** k
    F = F / np.maximum(np.abs(F).mean(0, keepdims=True), 1e-9)
    for j, b in enumerate([ALL_BODIES[i] for i in bi]):
        if b == "node": F[:, j] = 1.0                      # massless point: no factor
    return F


def fit(bodies=None, fmode="none", space="sqrt", lk="l1", magnitude=True, harm2=False,
        arch="indep", wall=W12, seed=7, steps=8000, lr=2e-2):
    bods = bodies or RECORD
    bi = [ALL_BODIES.index(b) for b in bods]
    TH = TH_ALL[:, bi]; nb = len(bi)
    F = make_F(bi, fmode)
    tf = np.sqrt if space == "sqrt" else np.log1p
    Ytf = tf(Y)
    T.manual_seed(seed)
    cT = T.tensor((F * np.cos(TH)).astype(np.float32).T, device=dev)
    sT = T.tensor((F * np.sin(TH)).astype(np.float32).T, device=dev)
    c2T = T.tensor((F * np.cos(2 * TH)).astype(np.float32).T, device=dev)
    s2T = T.tensor((F * np.sin(2 * TH)).astype(np.float32).T, device=dev)
    tv = TV[:, :wall].astype(np.float32)
    wy = np.sqrt(np.clip(tot[:wall], 0, None))
    Wm = tv * wy[None]; Wm = Wm / np.maximum(Wm.sum(1, keepdims=True), 1e-9)
    Wt = T.tensor(Wm.astype(np.float32), device=dev)
    vmean = (Ytf[:, :wall] * tv).sum(1) / np.maximum(tv.sum(1), 1.0)
    A0 = np.full((Tn, nb), -2.0, np.float32)
    if "pluto" in bods: A0[:, bods.index("pluto")] = inv_sp(vmean)
    Araw = T.tensor(A0, device=dev, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (Tn, nb, 1)).astype(np.float32) +
                 np.random.RandomState(seed).randn(Tn, nb, 2).astype(np.float32) * 0.01, device=dev, requires_grad=True)
    Bp = T.tensor(vmean.astype(np.float32), device=dev, requires_grad=True)
    A2 = T.tensor(np.full((Tn, nb), -4.0, np.float32), device=dev, requires_grad=harm2)
    U2 = T.tensor(np.tile([0.0, 1.0], (Tn, nb, 1)).astype(np.float32), device=dev, requires_grad=harm2)
    nf = {"tides2": 2, "tides1": 1}.get(arch, 0)
    renorm = arch in ("pie", "tides1", "tides2")
    cg = T.tensor(np.full((max(nf, 1), nb), -2.0, np.float32), device=dev, requires_grad=nf > 0)
    qg = T.tensor(np.random.RandomState(seed + 3).randn(max(nf, 1), nb, 2).astype(np.float32) * 0.1 +
                  np.array([0.0, 1.0], np.float32), device=dev, requires_grad=nf > 0)
    lam = T.tensor(np.zeros((Tn, max(nf, 1)), np.float32), device=dev, requires_grad=nf > 0)
    params = [Araw, U, Bp] + ([A2, U2] if harm2 else []) + ([cg, qg, lam] if nf else [])
    Yt = T.tensor(Ytf.astype(np.float32), device=dev)
    opt = T.optim.Adam(params, lr=lr)

    def forward():
        p = T.atan2(U[:, :, 0], U[:, :, 1]); A = T.nn.functional.softplus(Araw)
        cp = A * T.cos(p); sp = A * T.sin(p)
        C = Bp[:, None] + cp @ cT + sp @ sT; S = cp @ sT - sp @ cT
        if harm2:
            p2_ = T.atan2(U2[:, :, 0], U2[:, :, 1]); A2s = T.nn.functional.softplus(A2)
            C = C + (A2s * T.cos(p2_)) @ c2T + (A2s * T.sin(p2_)) @ s2T
            S = S + (A2s * T.cos(p2_)) @ s2T - (A2s * T.sin(p2_)) @ c2T
        if magnitude: lvl = C ** 2 + S ** 2 + 1e-8
        else: lvl = T.clamp(C, min=1e-4) ** 2
        if nf:
            q = T.atan2(qg[:, :, 0], qg[:, :, 1]); c = T.nn.functional.softplus(cg)
            gt = (c * T.cos(q)) @ cT + (c * T.sin(q)) @ sT
            gt = gt - gt[:, :wall].mean(1, keepdim=True)
            lvl = lvl * T.exp(T.clamp(lam @ gt, -3, 3))
        if renorm:
            lvl = 100.0 * lvl / lvl.sum(0, keepdim=True)
        return lvl

    for_loss = (lambda l: T.sqrt(l + 1e-8)) if space == "sqrt" else (lambda l: T.log1p(T.clamp(l, min=0)))
    best, stall, state = np.inf, 0, None
    for it in range(steps):
        e = for_loss(forward())[:, :wall] - Yt[:, :wall]
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
        lvl = forward().cpu().numpy()
    Yh = np.clip(lvl, 0, None)                             # forward() is LEVEL-scale in every space
    if renorm:
        Sb = float(Y[:, :wall].sum(0)[TV[:, :wall].any(0)].mean()) if False else float(Y[:, :wall].sum(0).mean())
        Yh = Yh * (Sb / 100.0)
    return Yh


def bench(Yw, wall=W12):
    tvw = TV[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1, keepdims=True) / np.maximum(tvw.sum(1, keepdims=True), 1.0)
    den = np.maximum(((Y[:, wall:n] - mu) ** 2).sum(1), 1e-6)
    skill = 1.0 - ((Y[:, wall:n] - Yw[:, wall:n]) ** 2).sum(1) / den
    curve = [1.0 - ((Y[:, wall + h] - Yw[:, wall + h]) ** 2).sum() / max(((Y[:, wall + h] - mu[:, 0]) ** 2).sum(), 1e-9)
             for h in range(n - wall)]
    return float(np.median(skill)), float((skill > 0).mean() * 100), float(np.mean(curve))


if __name__ == "__main__":
    res = []
    def run(tag, **kw):
        s, p, a = bench(fit(**kw))
        res.append((tag, s, p, a)); print(f"  {tag:44s} skill {s:+.4f} ({p:.1f}%>0) · AUC {a:+.4f}", flush=True)

    print(f"== SWEEP · independent yearly record · {Tn}×{n} · 12yr wall {W12} ==", flush=True)
    print("== A) ROSTERS (angles-only) ==", flush=True)
    run("A  record 8 (mars..chiron)")
    run("A  7 (no mars)", bodies=[b for b in RECORD if b != "mars"])
    run("A  +venus (9)", bodies=["venus"] + RECORD)
    run("A  +mercury (9)", bodies=["mercury"] + RECORD)
    run("A  +sun (9)", bodies=["sun"] + RECORD)
    run("A  all 11", bodies=ALL_BODIES)
    print("== B) DISTANCE FACTORS (record 8) ==", flush=True)
    for fm in ("1/r", "1/r2", "1/r3", "sgn/r", "sgn"):
        run(f"B  {fm}", fmode=fm)
    print("== C) MODELS (record 8, angles-only) ==", flush=True)
    run("C  cosine-sum (no magnitude)", magnitude=False)
    run("C  log1p space", space="log1p")
    run("C  MSE loss", lk="l2")
    run("C  2nd harmonic (+8 params)", harm2=True)
    run("C  tides2×pie (cross-sectional ref)", arch="tides2")
    print("== D) SEEDS (record) ==", flush=True)
    for sd in (1, 2):
        run(f"D  record seed {sd}", seed=sd)

    print("\n  LEAGUE (by 12yr AUC):", flush=True)
    for tag, s, p, a in sorted(res, key=lambda r: -r[3]):
        print(f"    {a:+.4f} AUC · {s:+.4f} skill · {tag}", flush=True)
    print("SWEEP DONE", flush=True)
