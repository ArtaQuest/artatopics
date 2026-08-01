#!/usr/bin/env python3
"""RE-ABLATION UNDER THE HORIZON ANCHOR — pure 30-year AUC (operator 2026-07-25: "keep competing to
push the 30-yr AUC, don't care about sign information anymore").

WHY THIS SWEEP: almost every structural choice in the deployed model — the roster (chiron dropped),
the detector power (k=2), angles-only, the N^¾ weight, one harmonic — was selected BEFORE the horizon
anchor existed. The anchor changed the model's failure mode completely (it was extrapolation drift,
which the anchor now controls), and this campaign has repeatedly found that the winning recipe flips
when the regime changes. So every one of those choices is re-opened and re-tested under the anchor.

DISCIPLINE (unchanged): everything is selected on WALL_INNER (fit ≤1965, judged 1966-95). Only the
finalists are fitted at WALL_OUTER, once each, at three seeds. 1996+ is never used to choose anything.
CONSTRAINT (operator): at prediction time the model sees ONLY the sky. The citation record enters
during training alone — through the loss, its weights and the anchor constant, never as an input.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/auc_reablate.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import torch as T

DEV = "mps" if T.backends.mps.is_available() else "cpu"
REC = CHAMPION_BODIES                                   # mars ju sa ur ne pl node
SLOW = ["saturn", "uranus", "neptune", "pluto", "node"]


def fit(wall, bodies=None, kpow=2.0, wexp=0.75, lam_h=0.03, anchor_k=5, beta=0.02, tau=0.15,
        fmode="none", harm=1, seed=7, steps=9000, lr=2e-2, ret_phase=False):
    bods = bodies or REC
    bi = [BODIES_ALL.index(b) for b in bods]; nb = len(bi)
    TH = TH_ALL[:, bi]; ne = TH.shape[0]
    R = R_ALL[:, bi]
    if fmode == "none": F = np.ones_like(R)
    else:
        k = {"1/r": 1, "1/r2": 2}[fmode]
        F = 1.0 / R ** k
        F = F / np.maximum(np.abs(F).mean(0, keepdims=True), 1e-9)
    if "node" in bods: F[:, bods.index("node")] = 1.0
    inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
    Ysq = np.sqrt(Y)
    T.manual_seed(seed)
    tb = lambda a: T.tensor(np.asarray(a, np.float32), device=DEV)
    cH = [tb((F * np.cos((h + 1) * TH)).T) for h in range(harm)]
    sH = [tb((F * np.sin((h + 1) * TH)).T) for h in range(harm)]
    tv = train_mask(wall).astype(np.float32)
    wy = np.clip(N[:wall], 0, None) ** wexp
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9); Wt = tb(W)
    Wa = np.zeros_like(W); Wa[:, wall - anchor_k:] = (tv * wy[None])[:, wall - anchor_k:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    m_anchor = tb(((Ysq[:, :wall] * Wa).sum(1))[:, None])
    vmean = (Ysq[:, :wall] * tv).sum(1) / np.maximum(tv.sum(1), 1.0)
    rs = np.random.RandomState(seed)
    A0 = np.full((Tn, nb), -2.0, np.float32)
    if "pluto" in bods: A0[:, bods.index("pluto")] = inv_sp(np.clip(vmean, 1e-3, None))
    Araw = [T.tensor(A0 if h == 0 else np.full((Tn, nb), -4.0, np.float32), device=DEV, requires_grad=True)
            for h in range(harm)]
    U = [T.tensor(np.tile([0.0, 1.0], (Tn, nb, 1)).astype(np.float32) + rs.randn(Tn, nb, 2).astype(np.float32) * 0.01,
                  device=DEV, requires_grad=True) for h in range(harm)]
    Bp = T.tensor(vmean.astype(np.float32), device=DEV, requires_grad=True)
    params = Araw + U + [Bp]
    Yt = tb(Ysq); opt = T.optim.Adam(params, lr=lr)
    hz = min(wall + HORIZON, ne)
    CEN = T.tensor(np.deg2rad(np.arange(12) * 30.0 + 15.0).astype(np.float32), device=DEV)
    PL = bods.index("pluto") if "pluto" in bods else 0

    def fwd(cs, ss):
        C = Bp[:, None].clone()
        for h in range(harm):
            p = T.atan2(U[h][:, :, 0], U[h][:, :, 1]); A = T.nn.functional.softplus(Araw[h])
            C = C + (A * T.cos(p)) @ cs[h] + (A * T.sin(p)) @ ss[h]
        return T.clamp(C, min=1e-4) ** kpow + 1e-8

    best, stall, state = np.inf, 0, None
    for it in range(steps):
        sig = T.sqrt(fwd(cH, sH) + 1e-8)
        per = ((sig[:, :wall] - Yt[:, :wall]).abs() * Wt).sum(1)
        d = (sig[:, wall:hz] - m_anchor) / T.clamp(m_anchor, min=1e-3)
        loss = (per + lam_h * (d ** 2).mean(1)).sum() / Tn
        if beta:
            pp = T.atan2(U[0][:, PL, 0], U[0][:, PL, 1])
            q = T.softmax(T.cos(pp[:, None] - CEN[None, :]) / tau, dim=1)
            Hj = -(q * T.log(q.clamp_min(1e-12))).sum(1).mean(); qb = q.mean(0)
            loss = loss + beta * (Hj + (qb * T.log(qb.clamp_min(1e-12))).sum())
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            lv = loss.item()
            if lv < best - 1e-7: best, stall, state = lv, 0, [x.detach().clone() for x in params]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip(params, state): x.copy_(sv)
        yh = np.clip(fwd(cH, sH).cpu().numpy(), 0, None)
        if ret_phase:
            ph = np.rad2deg(T.atan2(U[0][:, PL, 0], U[0][:, PL, 1]).cpu().numpy()) % 360.0
            return yh, ph
    return yh


CFG, RES = {}, {}
def run(tag, **kw):
    a = evaluate(fit(WALL_INNER, **kw), WALL_INNER)
    RES[tag] = a["auc"]; CFG[tag] = kw
    print(f"  {tag:34s} inner AUC {a['auc']:+.4f} · skill {a['skill']:+.4f} · {a['pct']:.1f}%>0", flush=True)


if __name__ == "__main__":
    print(f"═══ INNER WALL — every pre-anchor choice re-opened (1996+ untouched) ═══", flush=True)
    run("BASE v10 (deployed)")
    print("── A) roster, re-tested now that the anchor controls drift ──", flush=True)
    run("A +chiron (8)", bodies=REC + ["chiron"])
    run("A +sun (8)", bodies=["sun"] + REC)
    run("A +venus (8)", bodies=["venus"] + REC)
    run("A all-11", bodies=BODIES_ALL)
    run("A slow-5 only", bodies=SLOW)
    run("A -mars (6)", bodies=[b for b in REC if b != "mars"])
    print("── B) detector power ──", flush=True)
    for k in (1.5, 2.5, 3.0): run(f"B k={k}", kpow=k)
    print("── C) distance factor ──", flush=True)
    for f in ("1/r", "1/r2"): run(f"C {f}", fmode=f)
    print("── D) evidence weight ──", flush=True)
    for w in (0.6, 0.9, 1.0): run(f"D N^{w}", wexp=w)
    print("── E) the anchor itself ──", flush=True)
    for l in (0.01, 0.08, 0.2): run(f"E lam={l}", lam_h=l)
    for a in (3, 10, 20): run(f"E anchor_k={a}", anchor_k=a)
    print("── F) the KL term (no longer scored — does dropping it help AUC?) ──", flush=True)
    run("F beta=0 (no KL)", beta=0.0)
    print("── G) harmonics ──", flush=True)
    run("G harm=2", harm=2)

    order = sorted(RES, key=lambda t: -RES[t])
    print("\n  INNER LEAGUE:", flush=True)
    for t in order[:10]: print(f"    {RES[t]:+.4f}  {t}", flush=True)
    base = RES["BASE v10 (deployed)"]
    finalists = [t for t in order[:4] if t != "BASE v10 (deployed)"][:3]
    print(f"\n  base {base:+.4f} · finalists → {finalists}", flush=True)

    print("\n═══ OUTER WALL — finalists + base, 3 seeds each, fitted once ═══", flush=True)
    out = {}
    for t in ["BASE v10 (deployed)"] + finalists:
        aucs = [evaluate(fit(WALL_OUTER, seed=sd, **CFG[t]), WALL_OUTER) for sd in (7, 11, 23)]
        med = float(np.median([a["auc"] for a in aucs]))
        out[t] = {"auc_med": med, "spread": [min(a["auc"] for a in aucs), max(a["auc"] for a in aucs)],
                  "skill": float(np.median([a["skill"] for a in aucs])),
                  "pct": float(np.median([a["pct"] for a in aucs])), "cfg": {k: str(v) for k, v in CFG[t].items()}}
        print(f"  {t:34s} OUTER AUC {med:+.4f} [{out[t]['spread'][0]:+.4f}..{out[t]['spread'][1]:+.4f}] · "
              f"skill {out[t]['skill']:+.4f} · {out[t]['pct']:.1f}%>0", flush=True)
    win = max(out, key=lambda t: out[t]["auc_med"])
    print(f"\n  WINNER: {win} → {out[win]['auc_med']:+.4f} (deployed v10 = +0.8193)", flush=True)
    json.dump({"inner": RES, "outer": out, "winner": win},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "auc_reablate.json"), "w"), indent=1)
    print("REABLATEDONE", flush=True)
