#!/usr/bin/env python3
"""SIX-YEAR LEAGUE: cross-sectional + attention architectures vs the record at wall n-74.
Everything is a function of the sky only (forecastable by construction); pie applied to all."""
import importlib.util as u, os
import numpy as np, pandas as pd

# repo root = three levels up, same as every other script here (was a hardcoded local path)
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
af = _load("analysis/arxivtopics/arxiv_fit.py", "af")
import torch as T
dev = "mps" if T.backends.mps.is_available() else "cpu"
BODIES = af.BODIES
names, Y, labels, _ = af.load_lunar()
Tn, n = Y.shape
E = pd.read_csv("analysis/arxivtopics/_ephemeris_lunar.csv").set_index("Time")
TH = np.stack([np.deg2rad(E[f"{b}_lon"].loc[labels].to_numpy(float)) for b in BODIES], 1)
R = np.stack([E[f"{b}_dist"].loc[labels].to_numpy(float) for b in BODIES], 1)
F = 1.0 / R; F = F / np.abs(F).mean(0, keepdims=True); F[:, [BODIES.index("node")]] = 1.0
nb = len(BODIES)
W6 = n - 74
inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
Ysq = np.sqrt(Y)
YsqT = T.tensor(Ysq.astype(np.float32), device=dev)
cTt = T.tensor((F * np.cos(TH)).astype(np.float32).T, device=dev)
sTt = T.tensor((F * np.sin(TH)).astype(np.float32).T, device=dev)
THt = T.tensor(TH.astype(np.float32), device=dev)          # (n, nb)
Sbar6 = float(Y[:, :W6].sum(0).mean())

def train(params, forward, wall=W6, steps=8000, lr=2e-2):
    opt = T.optim.Adam(params, lr=lr)
    best, stall, state = np.inf, 0, None
    for it in range(steps):
        yh = forward()                                     # LEVEL-scale predictions (rows, n)
        loss = (T.sqrt(T.clamp(yh[:, :wall], min=0) + 1e-8) - YsqT[:, :wall]).abs().mean()
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
    return Yh / np.maximum(Yh.sum(0, keepdims=True), 1e-9) * Sbar6

def bench(Yw, wall=W6):
    mu = Y[:, :wall].mean(1, keepdims=True)
    den = np.maximum(((Y[:, wall:n] - mu) ** 2).sum(1), 1e-6)
    skill = 1.0 - ((Y[:, wall:n] - Yw[:, wall:n]) ** 2).sum(1) / den
    curve = [1.0 - ((Y[:, wall+h] - Yw[:, wall+h]) ** 2).sum() / max(((Y[:, wall+h] - mu[:,0]) ** 2).sum(), 1e-9)
             for h in range(n - wall)]
    return float(np.median(skill)), float((skill > 0).mean() * 100), float(np.mean(curve))

def base_receiver(seed=7, wall=W6):
    T.manual_seed(seed)
    A0 = np.full((Tn, nb), -2.0, np.float32); A0[:, BODIES.index("pluto")] = inv_sp(Ysq[:, :wall].mean(1))
    Araw = T.tensor(A0, device=dev, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (Tn, nb, 1)).astype(np.float32) +
                 np.random.RandomState(seed).randn(Tn, nb, 2).astype(np.float32) * 0.01, device=dev, requires_grad=True)
    Bp = T.tensor(Ysq[:, :wall].mean(1).astype(np.float32), device=dev, requires_grad=True)
    def mfun():
        p = T.atan2(U[:, :, 0], U[:, :, 1]); A = T.nn.functional.softplus(Araw)
        cp = A * T.cos(p); sp = A * T.sin(p)
        C = Bp[:, None] + cp @ cTt + sp @ sTt; S = cp @ sTt - sp @ cTt
        return T.sqrt(C ** 2 + S ** 2 + 1e-8)
    return [Araw, U, Bp], mfun

res = []
def row(tag, Yw):
    s, p, a = bench(Yw); res.append((tag, s, p, a))
    print(f"  {tag:46s} skill {s:+.4f} ({p:.1f}%>0) · AUC {a:+.4f}", flush=True)

print(f"== SIX-YEAR LEAGUE · wall {W6} (74 unseen lunations, 2020-05..2026-06) ==", flush=True)

# X1 record (independent power receiver)
prm, mfun = base_receiver()
row("X1 record: independent power receiver", train(prm, lambda: mfun() ** 2))

# X2 joint sum-divide (bounded cross-section, trained through the pie)
prm2, mfun2 = base_receiver()
row("X2 joint sum-divide (bounded)", train(prm2, lambda: (lambda P: 100.0 * P / P.sum(0, keepdim=True))(mfun2() ** 2)))

# X3 market factor: y_j = m_j^2 * exp(lam_j * g(t)), g = shared sky phasor (one reallocation axis)
prm3, mfun3 = base_receiver()
T.manual_seed(7)
cg = T.tensor(np.full(nb, -2.0, np.float32), device=dev, requires_grad=True)
qg = T.tensor(np.random.RandomState(3).randn(nb, 2).astype(np.float32) * 0.1 + np.array([0., 1.], np.float32), device=dev, requires_grad=True)
lam = T.tensor(np.zeros(Tn, np.float32), device=dev, requires_grad=True)
def fwd3():
    q = T.atan2(qg[:, 0], qg[:, 1]); c = T.nn.functional.softplus(cg)
    g = (c * T.cos(q))[None, :] @ cTt + (c * T.sin(q))[None, :] @ sTt      # (1, n) shared factor
    g = g - g[:, :W6].mean()
    return (mfun3() ** 2) * T.exp(T.clamp(lam[:, None] * g, -3, 3))
row("X3 market factor (shared sky axis + loadings)", train(prm3 + [cg, qg, lam], fwd3))

# X4 aspect-bank attention: shared bank of ALL sky harmonics (11 singles + 55 pairs, cos+sin = 132),
#    per-field low-rank heads (fields attend to shared sky patterns), amplitude head squared.
II, KK = np.triu_indices(nb, 1)
ANG = np.concatenate([TH, TH[:, II] - TH[:, KK]], 1)                       # (n, 66)
BANK = np.concatenate([np.cos(ANG), np.sin(ANG)], 1).astype(np.float32)    # (n, 132)
Bk = T.tensor(BANK.T, device=dev)                                          # (132, n)
for rk, tag in [(8, "X4 aspect-bank attention (rank 8)"), (32, "X4b aspect-bank attention (rank 32)")]:
    T.manual_seed(7)
    Uh = T.tensor(np.random.RandomState(5).randn(Tn, rk).astype(np.float32) * 0.05, device=dev, requires_grad=True)
    Vh = T.tensor(np.random.RandomState(6).randn(rk, 132).astype(np.float32) * 0.05, device=dev, requires_grad=True)
    b0 = T.tensor(Ysq[:, :W6].mean(1).astype(np.float32), device=dev, requires_grad=True)
    def fwd4():
        amp = b0[:, None] + (Uh @ Vh) @ Bk
        return T.clamp(amp, min=0) ** 2
    row(tag, train([Uh, Vh, b0], fwd4))

# X5 record + market factor POST-HOC style but joint: bounded ratio of X3
prm5, mfun5 = base_receiver()
T.manual_seed(7)
cg5 = T.tensor(np.full(nb, -2.0, np.float32), device=dev, requires_grad=True)
qg5 = T.tensor(np.random.RandomState(4).randn(nb, 2).astype(np.float32) * 0.1 + np.array([0., 1.], np.float32), device=dev, requires_grad=True)
lam5 = T.tensor(np.zeros(Tn, np.float32), device=dev, requires_grad=True)
def fwd5():
    q = T.atan2(qg5[:, 0], qg5[:, 1]); c = T.nn.functional.softplus(cg5)
    g = (c * T.cos(q))[None, :] @ cTt + (c * T.sin(q))[None, :] @ sTt
    g = g - g[:, :W6].mean()
    P = (mfun5() ** 2) * T.exp(T.clamp(lam5[:, None] * g, -3, 3))
    return 100.0 * P / P.sum(0, keepdim=True)
row("X5 market factor + bounded pie (joint)", train(prm5 + [cg5, qg5, lam5], fwd5))

print("\n  LEAGUE (by 6yr AUC):", flush=True)
for tag, s, p, a in sorted(res, key=lambda r: -r[3]):
    print(f"    {a:+.4f} AUC · {s:+.4f} skill · {tag}", flush=True)
print("SIXYR DONE", flush=True)
