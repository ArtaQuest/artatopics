#!/usr/bin/env python3
"""QUARTERLY ablation program (redone 2026-07-24 on the latest published dataset, snapshot byte-verified
unchanged). Citations-received shares 1900Q1..2026Q1 (pro-rata q0), mid-quarter sky; PRIMARY criterion =
AUC over the 48 held-out quarters (TWELVE years). Default = the model of record: angles-only plain pie.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/arxiv_quarterly_ablations.py
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
BODIES = [b for b in p2.BODIES if b not in {"sun","moon","mercury","venus"}]  # 8 bodies (mars included)
nb = len(BODIES)
MIN_MEAN_WEEKLY = 5.0


def load_quarterly():
    df = pd.read_csv("analysis/citations/citations_received_yearly.csv")
    quarters = [c for c in df.columns if c not in ("subfield_id", "subfield", "field", "domain")]  # years
    names = list(df["subfield"].to_numpy())
    M = df[quarters].to_numpy(float)
    tot = M.sum(0)
    S = 100.0 * M / np.maximum(tot[None, :], 1.0)
    return names, S, quarters, tot > 0


names, Y, quarters, VALID = load_quarterly()
_Z = (pd.read_csv("analysis/citations/citations_received_yearly.csv")[[c for c in pd.read_csv("analysis/citations/citations_received_yearly.csv").columns if c[0].isdigit()]].to_numpy(float) > 0)
TV = np.ones_like(_Z, dtype=bool)
for _i in range(_Z.shape[1] - 2, -1, -1): TV[:, _i] = _Z[:, _i] & TV[:, _i + 1]   # per-topic non-zero suffix
_my = pd.read_csv("analysis/citations/citations_received_yearly.csv")
NTOT = _my[quarters].to_numpy(float).sum(0)            # evidence/YEAR — aligned to Y's columns
Tn, n = Y.shape
E = pd.read_csv("analysis/arxivtopics/_ephemeris_yearly.csv").set_index("Time"); E.index = E.index.astype(str)
TH_ALL = np.stack([np.deg2rad(E[f"{b}_lon"].loc[quarters].to_numpy(float)) for b in BODIES], 1)
R_ALL = np.stack([E[f"{b}_dist"].loc[quarters].to_numpy(float) for b in BODIES], 1)
W6 = n - 12                                            # the TWELVE-year headline wall
Ysq = np.sqrt(Y)
inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
print(f"== YEARLY · {Tn} fields x {n} years ({quarters[0]}..{quarters[-1]}) · 12yr wall {W6} (7 slow bodies, √N) ==", flush=True)


def make_F(R, mode="1/r"):
    if mode == "none": return np.ones_like(R)
    F = 1.0 / R if mode == "1/r" else 1.0 / R ** 2
    F = F / np.abs(F).mean(0, keepdims=True)
    F[:, [BODIES.index("node")]] = 1.0
    return F


def fit(wall=W6, arch="indep", space="sqrt", lk="l1", fmode="none", bodies_idx=None,
        seed=7, steps=8000, lr=2e-2, use_anchor=True, magnitude=True):
    bi = list(range(nb)) if bodies_idx is None else bodies_idx
    TH = TH_ALL[:, bi]; R = R_ALL[:, bi]; nbi = len(bi)
    F = make_F(R, fmode) if fmode != "none" else np.ones_like(R)
    tf = np.sqrt if space == "sqrt" else np.log1p
    itf = (lambda m: m) if space == "sqrt" else (lambda m: np.expm1(m))   # on the model OUTPUT scale
    Ytf = tf(Y)
    T.manual_seed(seed)
    cT = T.tensor((F * np.cos(TH)).astype(np.float32).T, device=dev)
    sT = T.tensor((F * np.sin(TH)).astype(np.float32).T, device=dev)
    A0 = np.full((Tn, nbi), -2.0, np.float32)
    car = bi.index(BODIES.index("pluto")) if BODIES.index("pluto") in bi else 0
    A0[:, car] = inv_sp(Ytf[:, :wall].mean(1))
    Araw = T.tensor(A0, device=dev, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (Tn, nbi, 1)).astype(np.float32) +
                 np.random.RandomState(seed).randn(Tn, nbi, 2).astype(np.float32) * 0.01, device=dev, requires_grad=True)
    Bp = T.tensor(Ytf[:, :wall].mean(1).astype(np.float32), device=dev, requires_grad=use_anchor)
    if not use_anchor:
        with T.no_grad(): Bp.zero_()
    nf = {"indep": 0, "renorm": 0, "pie": 0, "tides1": 1, "tides2": 2, "tides3": 3}[arch]
    cg = T.tensor(np.full((max(nf, 1), nbi), -2.0, np.float32), device=dev, requires_grad=nf > 0)
    qg = T.tensor(np.random.RandomState(seed + 3).randn(max(nf, 1), nbi, 2).astype(np.float32) * 0.1 +
                  np.array([0.0, 1.0], np.float32), device=dev, requires_grad=nf > 0)
    lam = T.tensor(np.zeros((Tn, max(nf, 1)), np.float32), device=dev, requires_grad=nf > 0)
    params = [Araw, U] + ([Bp] if use_anchor else []) + ([cg, qg, lam] if nf > 0 else [])
    Yt = T.tensor(Ytf.astype(np.float32), device=dev)
    opt = T.optim.Adam(params, lr=lr)

    def forward():
        p = T.atan2(U[:, :, 0], U[:, :, 1]); A = T.nn.functional.softplus(Araw)
        cp = A * T.cos(p); sp = A * T.sin(p)
        C = Bp[:, None] + cp @ cT + sp @ sT; S = cp @ sT - sp @ cT
        m = T.sqrt(C ** 2 + S ** 2 + 1e-8) if magnitude else T.clamp(C, min=1e-4)
        lvl = m ** 2 if space == "sqrt" else T.expm1(T.clamp(m, max=20))     # LEVEL scale
        if nf > 0:
            q = T.atan2(qg[:, :, 0], qg[:, :, 1]); c = T.nn.functional.softplus(cg)
            gt = (c * T.cos(q)) @ cT + (c * T.sin(q)) @ sT
            gt = gt - gt[:, :wall].mean(1, keepdim=True)
            lvl = lvl * T.exp(T.clamp(lam @ gt, -3, 3))
        if arch in ("pie", "tides1", "tides2", "tides3"):
            lvl = 100.0 * lvl / lvl.sum(0, keepdim=True)
        return lvl

    for_loss = (lambda l: T.sqrt(l + 1e-8)) if space == "sqrt" else (lambda l: T.log1p(T.clamp(l, min=0)))
    if arch in ("indep", "renorm"):
        Wm = TV[:, :wall].astype(float) * np.sqrt(np.where(VALID[:wall], NTOT[:wall], 0.0))[None]
        Wm = Wm / np.maximum(Wm.sum(1, keepdims=True), 1e-9)      # PER-TOPIC masked √N (record recipe)
        wt = T.tensor(Wm.astype(np.float32), device=dev)
        per_topic = True
    else:
        wq = np.sqrt(np.where(VALID[:wall], NTOT[:wall], 0.0)); wq = wq / max(wq.sum(), 1e-9)
        wt = T.tensor(wq.astype(np.float32), device=dev)
        per_topic = False
    best, stall, state = np.inf, 0, None
    for it in range(steps):
        e = for_loss(forward())[:, :wall] - Yt[:, :wall]
        ew = (e.abs() if lk == "l1" else e ** 2) * (wt if per_topic else wt[None])
        loss = ew.sum() / e.shape[0]
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            l = loss.item()
            if l < best - 1e-7: best, stall, state = l, 0, [x.detach().clone() for x in params]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip(params, state): x.copy_(sv)
        Yh = np.clip(forward().cpu().numpy(), 0, None)
    Sb = float(Y[:, :wall].sum(0)[VALID[:wall]].mean())
    if arch in ("pie", "tides1", "tides2", "tides3"):
        return Yh * (Sb / 100.0)
    if arch == "renorm":
        return Yh / np.maximum(Yh.sum(0, keepdims=True), 1e-9) * Sb
    return Yh                                                    # indep: the share directly


def bench(Yw, wall=W6):
    tvw = TV[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1, keepdims=True) / np.maximum(tvw.sum(1, keepdims=True), 1.0)
    den = np.maximum(((Y[:, wall:n] - mu) ** 2).sum(1), 1e-6)
    skill = 1.0 - ((Y[:, wall:n] - Yw[:, wall:n]) ** 2).sum(1) / den
    curve = [1.0 - ((Y[:, wall+h] - Yw[:, wall+h]) ** 2).sum() / max(((Y[:, wall+h] - mu[:,0]) ** 2).sum(), 1e-9)
             for h in range(n - wall)]
    return float(np.median(skill)), float((skill > 0).mean() * 100), float(np.mean(curve))


res = []
def run(tag, **kw):
    wall = kw.get("wall", W6)
    s, p, a = bench(fit(**kw), wall)
    res.append((tag, s, p, a)); print(f"  {tag:44s} skill {s:+.4f} ({p:.1f}%>0) · AUC {a:+.4f}", flush=True)


print("== 1) ARCHITECTURES (L1-sqrt, angles-only, 11 bodies) ==", flush=True)
for arch in ("indep", "renorm", "pie", "tides1", "tides2", "tides3"):
    run(f"A  {arch}", arch=arch)
print("== 2) LOSS/SPACE (indep) ==", flush=True)
run("L  indep · L1 on log1p", space="log1p")
run("L  indep · MSE on sqrt", lk="l2")
print("== 3) DISTANCE (indep) ==", flush=True)
run("D  indep · 1/r", fmode="1/r")
run("D  indep · 1/r^2", fmode="1/r2")
print("== 4) STRUCTURE (indep) ==", flush=True)
run("S  indep · no magnitude", magnitude=False)
run("S  indep · no anchor", use_anchor=False)
print("== 5) LEAVE-ONE-BODY-OUT (diagnostic; roster stays 11) ==", flush=True)
for i, b in enumerate(BODIES):
    run(f"B  without {b}", bodies_idx=[k for k in range(nb) if k != i])
print("== 6) SEEDS (record recipe) ==", flush=True)
for sd in (1, 2, 3):
    run(f"F  indep seed {sd}", seed=sd)
print("== 7) RECORD at all walls ==", flush=True)
for wname, w in (("4yr", n - 4), ("8yr", n - 8), ("12yr", W6)):
    run(f"W  indep @{wname}", wall=w)

print("\n  LEAGUE (by 12yr AUC):", flush=True)
for tag, s, p, a in sorted([r for r in res if "@4yr" not in r[0] and "@8yr" not in r[0]], key=lambda r: -r[3])[:12]:
    print(f"    {a:+.4f} AUC · {s:+.4f} skill · {tag}", flush=True)
print("QABLATE DONE", flush=True)
