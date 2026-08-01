#!/usr/bin/env python3
"""CROSS-SECTIONAL model (operator 2026-07-21): the 129 receivers trained JOINTLY as one composition.

Same per-field receiver m_j(t) = |b_j + Σᵢ a_jᵢ·(r̄ᵢ/rᵢ(t))·e^(i(θᵢ(t)−p_jᵢ))|; the week's
prediction is simply every model's output divided by their sum (operator: "just sum the output of
all the models and divide by sum, then do gradient descent"):  ŷ_j(t) = 100·m_j(t)/Σ_k m_k(t)
over exactly the 129 kept fields — the SAME 129×25 parameters as the independent fit, not one more
("no additional params"). The predictions sum to 100 by construction in every week, in-sample and
forecast alike, and ONE MSE over the whole matrix, minimised end-to-end by gradient descent in
PyTorch through the normalization, couples every field: over-predicting one field now costs every
other. Recipe otherwise unchanged — raw shares, a ≥ 0 (softplus), free phases, no penalties, no
tuning; Adam lr 2e-2, plateau early-stop. Two models (deploy n−4, benchmark n−104) + skill(h)/AUC.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/arxiv_xsec.py
"""
import importlib.util as u, os
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
p2 = _load("analysis/adstopics/astro_phasor2.py", "p2")
BODIES = p2.BODIES
MIN_MEAN_COUNT = 5.0
START = "2008-01-07"


def load_weekly():
    df = pd.read_csv("analysis/arxivtopics/weekly_counts.csv")
    wk = [c for c in df.columns if c != "category" and c >= START][:-1]
    tot = np.maximum(df[wk].sum(0).to_numpy(float), 1.0)
    names, Ys = [], []
    for _, r in df.iterrows():
        y = r[wk].to_numpy(float)
        if y.mean() < MIN_MEAN_COUNT: continue
        names.append(r["category"]); Ys.append(100.0 * y / tot)
    return names, np.stack(Ys), wk


def fit_xsec(Yf, TH, F, fit_end, device, seed=7, steps=8000, lr=2e-2):
    """Joint compositional fit over the 129 kept fields. Returns (Yh_softmax (rows, ne), params)."""
    import torch as T
    T.manual_seed(seed)
    rows, n = Yf.shape; ne = TH.shape[0]
    inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
    cT = T.tensor((F * np.cos(TH)).astype(np.float32).T, device=device)   # (12, ne)
    sT = T.tensor((F * np.sin(TH)).astype(np.float32).T, device=device)
    A0 = np.full((rows, 12), -2.0, np.float32); A0[:, 9] = inv_sp(Yf[:, :fit_end].mean(1))
    Araw = T.tensor(A0, device=device, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (rows, 12, 1)).astype(np.float32) +
                 np.random.RandomState(seed).randn(rows, 12, 2).astype(np.float32) * 0.01,
                 device=device, requires_grad=True)
    Bp = T.tensor(Yf[:, :fit_end].mean(1).astype(np.float32), device=device, requires_grad=True)
    params = [Araw, U, Bp]
    Yt = T.tensor(Yf.astype(np.float32), device=device)
    opt = T.optim.Adam(params, lr=lr)

    def forward():
        p = T.atan2(U[:, :, 0], U[:, :, 1])
        A = T.nn.functional.softplus(Araw)
        cp = A * T.cos(p); sp = A * T.sin(p)
        C = Bp[:, None] + cp @ cT + sp @ sT
        S = cp @ sT - sp @ cT
        m = T.sqrt(C ** 2 + S ** 2 + 1e-8)               # (rows, ne) every model's output
        return 100.0 * m / m.sum(0, keepdim=True)        # divide by the sum: columns sum to 100 exactly

    best, stall, state = np.inf, 0, None
    for it in range(steps):
        yh = forward()
        loss = ((yh[:, :fit_end] - Yt[:, :fit_end]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            l = loss.item()
            if l < best - 1e-7: best, stall, state = l, 0, [x.detach().clone() for x in params]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip(params, state): x.copy_(sv)
        Yh = forward()
        out = dict(p=T.atan2(U[:, :, 0], U[:, :, 1]).cpu().numpy(),
                   a=T.nn.functional.softplus(Araw).cpu().numpy(), b=Bp.cpu().numpy())
    return Yh.cpu().numpy(), out


def main():
    dev = "cpu"
    try:
        import torch as T
        if T.backends.mps.is_available(): dev = "mps"
    except Exception: pass
    names, Y, wk = load_weekly()
    Tn, n = Y.shape
    E = pd.read_csv("analysis/arxivtopics/_ephemeris_weekly.csv")
    future = [t for t in E["Time"] if t > wk[-1]][:8]
    weeks_ext = list(wk) + future
    Ei = E.set_index("Time")
    TH = np.stack([np.deg2rad(Ei[f"{b}_lon"].loc[weeks_ext].to_numpy(float)) for b in BODIES], 1)
    R = np.stack([Ei[f"{b}_dist"].loc[weeks_ext].to_numpy(float) for b in BODIES], 1)
    F = 1.0 / R; F = F / np.abs(F[:n]).mean(0, keepdims=True)
    F[:, [BODIES.index("node")]] = 1.0
    WALL = n - 104; DEP = n - 4
    print(f"  XSEC · {Tn} rows x {n} weeks · columns sum to 100 by construction · "
          f"bench<{WALL} · deploy<{DEP} · dev {dev}", flush=True)

    Yd, prm = fit_xsec(Y, TH, F, DEP, dev)
    def r2rows(Yv, Yh):
        tot2 = np.maximum(((Yv - Yv.mean(1, keepdims=True)) ** 2).sum(1), 1e-6)
        return 1.0 - ((Yv - Yh) ** 2).sum(1) / tot2
    r2 = r2rows(Y[:, :DEP], Yd[:Tn, :DEP])
    print(f"  DEPLOY: fit R² {r2.mean():+.4f}/{np.median(r2):+.4f}  (independent was +0.5070/+0.5226)", flush=True)

    Yw, _ = fit_xsec(Y, TH, F, WALL, dev)
    mu = Y[:, :WALL].mean(1, keepdims=True)
    den = np.maximum(((Y[:, WALL:n] - mu) ** 2).sum(1), 1e-6)
    skill = 1.0 - ((Y[:, WALL:n] - Yw[:Tn, WALL:n]) ** 2).sum(1) / den
    curve = [round(float(1.0 - ((Y[:, WALL + h] - Yw[:Tn, WALL + h]) ** 2).sum() /
                   max(((Y[:, WALL + h] - mu[:, 0]) ** 2).sum(), 1e-9)), 4) for h in range(104)]
    auc = float(np.mean(curve))
    dtr = np.maximum(((Y[:, WALL - 104:WALL] - mu) ** 2).sum(1), 1e-6)
    trm = 1.0 - ((Y[:, WALL - 104:WALL] - Yw[:Tn, WALL - 104:WALL]) ** 2).sum(1) / dtr
    print(f"  BENCH: skill {np.median(skill):+.4f} ({(skill > 0).mean()*100:.1f}%>0) · AUC(104wk) {auc:+.4f} · "
          f"train104 {np.median(trm):+.4f}", flush=True)
    print(f"         (independent was: skill +0.7465 (78.3%>0) · AUC +0.8785 · train104 +0.7653)", flush=True)
    sun = np.rad2deg(prm["p"][:Tn, 0]) % 360
    hist = np.bincount((sun // 30).astype(int), minlength=12)
    print("  sun-phase signs: " + " ".join(f"{p2.SIGNS[i][:3]} {hist[i]}" for i in range(12)), flush=True)
    from collections import Counter
    dom = Counter(BODIES[int(k)] for k in prm["a"][:Tn].argmax(1))
    print("  dominant bodies:", dict(dom.most_common(6)), flush=True)
    chk = Yd.sum(0)
    print(f"  column-sum check (deploy, incl. forecast weeks): min {chk.min():.4f} max {chk.max():.4f}", flush=True)
    np.savez_compressed("analysis/arxivtopics/arxiv_xsec_final.npz",
                        a=prm["a"].astype(np.float32), p=prm["p"].astype(np.float32),
                        b=prm["b"].astype(np.float32),
                        yhat_dep=Yd.astype(np.float32), yhat_w=Yw.astype(np.float32),
                        r2=r2.astype(np.float32), r2_ins=trm.astype(np.float32),
                        r2_oos_skill=skill.astype(np.float32),
                        skill_curve=np.array(curve, np.float32), wall=WALL, dep_wall=DEP,
                        ext=len(weeks_ext) - n, weeks=np.array(weeks_ext),
                        bodies=np.array(BODIES), names=np.array(names))
    print("  saved -> analysis/arxivtopics/arxiv_xsec_final.npz", flush=True)
    print("ARXIVXSEC DONE", flush=True)


if __name__ == "__main__":
    main()
