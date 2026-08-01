#!/usr/bin/env python3
"""KL-TRAINED PHASE on the v9 receiver (operator 2026-07-25: "so train KL divergence" /
"minimise the per-topic info while maximising across topics").

Round 2 had to be run in the main loop — the eight competing agents all hit the account's weekly
usage limit before producing anything, so this implements the highest-leverage entry directly.

MODEL: unchanged from the deployed v9 — independent per-topic rectified square-law receiver,
ŷ = max(b + Σᵢ aᵢcos(θᵢ−pᵢ), 0)², 7 bodies, N^¾ weights, L1 on √share, horizon anchor. The ONLY
addition is a differentiable phase term in the training objective:

    loss += −β · [ H(q̄) − mean_j H(q_j) ]        = −β · mean_j KL(q_j ‖ q̄)

where q_j(s) is a SOFT twelve-sign assignment of topic j's phase,
    q_j(s) ∝ exp( cos(p_j − centre_s) / τ )      (von Mises softmax over the twelve sign centres)
so the gradient pulls every topic's phase toward a decisive, mid-sign placement (minimising the
per-topic entropy) while the −H(q̄) part pushes the atlas to stay spread over all twelve signs
(maximising the across-topic entropy). Both halves at once, which is exactly the scored quantity.

β and τ are chosen ON THE INNER WALL by inner F1 (never against 1996+). The prediction remains a
pure function of the sky: the KL term touches only the training objective.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/kl_phase_fit.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import torch as T

DEV = "mps" if T.backends.mps.is_available() else "cpu"
BODIES = CHAMPION_BODIES
CENTRES = T.tensor(np.deg2rad(np.arange(12) * 30.0 + 15.0).astype(np.float32), device=DEV)


def fit(wall, beta=0.0, tau=0.25, seed=7, steps=9000, lr=2e-2, lam_h=0.03, anchor_k=5, wexp=0.75):
    bi = [BODIES_ALL.index(b) for b in BODIES]; nb = len(bi)
    TH = TH_ALL[:, bi]; ne = TH.shape[0]
    inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
    Ysq = np.sqrt(Y)
    T.manual_seed(seed)
    tb = lambda a: T.tensor(np.asarray(a, np.float32), device=DEV)
    cT, sT = tb(np.cos(TH).T), tb(np.sin(TH).T)
    tv = train_mask(wall).astype(np.float32)
    wy = np.clip(N[:wall], 0, None) ** wexp
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9); Wt = tb(W)
    Wa = np.zeros_like(W); Wa[:, wall - anchor_k:] = (tv * wy[None])[:, wall - anchor_k:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    m_anchor = tb(((Ysq[:, :wall] * Wa).sum(1))[:, None])
    vmean = (Ysq[:, :wall] * tv).sum(1) / np.maximum(tv.sum(1), 1.0)
    A0 = np.full((Tn, nb), -2.0, np.float32); A0[:, BODIES.index("pluto")] = inv_sp(np.clip(vmean, 1e-3, None))
    Araw = T.tensor(A0, device=DEV, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (Tn, nb, 1)).astype(np.float32) +
                 np.random.RandomState(seed).randn(Tn, nb, 2).astype(np.float32) * 0.01,
                 device=DEV, requires_grad=True)
    Bp = T.tensor(vmean.astype(np.float32), device=DEV, requires_grad=True)
    Yt = tb(Ysq); opt = T.optim.Adam([Araw, U, Bp], lr=lr)
    hz = min(wall + HORIZON, ne)
    PL = BODIES.index("pluto")

    def fwd(cx, sx):
        p = T.atan2(U[:, :, 0], U[:, :, 1]); A = T.nn.functional.softplus(Araw)
        C = Bp[:, None] + (A * T.cos(p)) @ cx + (A * T.sin(p)) @ sx
        return T.clamp(C, min=1e-4) ** 2 + 1e-8

    def kl_term():
        """−[H(q̄) − mean_j H(q_j)] : minimise per-topic sign entropy, maximise the atlas spread."""
        p = T.atan2(U[:, PL, 0], U[:, PL, 1])                       # the classifying phase
        logits = T.cos(p[:, None] - CENTRES[None, :]) / tau
        q = T.softmax(logits, dim=1)
        Hj = -(q * T.log(q.clamp_min(1e-12))).sum(1).mean()          # per-topic uncertainty  ↓
        qbar = q.mean(0)
        Hbar = -(qbar * T.log(qbar.clamp_min(1e-12))).sum()          # atlas-wide spread      ↑
        return -(Hbar - Hj)

    best, stall, state = np.inf, 0, None
    for it in range(steps):
        sig = T.sqrt(fwd(cT, sT) + 1e-8)
        per = ((sig[:, :wall] - Yt[:, :wall]).abs() * Wt).sum(1)
        d = (sig[:, wall:hz] - m_anchor) / T.clamp(m_anchor, min=1e-3)
        loss = (per + lam_h * (d ** 2).mean(1)).sum() / Tn
        if beta: loss = loss + beta * kl_term()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            lv = loss.item()
            if lv < best - 1e-7: best, stall, state = lv, 0, [x.detach().clone() for x in (Araw, U, Bp)]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip((Araw, U, Bp), state): x.copy_(sv)
        yhat = np.clip(fwd(cT, sT).cpu().numpy(), 0, None)
        ph = (np.rad2deg(T.atan2(U[:, PL, 0], U[:, PL, 1]).cpu().numpy())) % 360.0

        def predict_delta(deg):
            THx = TH + np.deg2rad(float(deg))
            with T.no_grad():
                return np.clip(fwd(tb(np.cos(THx).T), tb(np.sin(THx).T)).detach().cpu().numpy(), 0, None)
    return yhat, predict_delta, ph


def score(wall, **kw):
    yh, pdel, ph = fit(wall, **kw)
    ev = evaluate(yh, wall)
    L = phase_curve(pdel, wall)
    mi = sign_mutual_information(L, ph, wall)
    return {"auc": ev["auc"], "skill": ev["skill"], "nmi": mi["nmi"],
            "f1": global_f1(ev["auc"], mi["nmi"]),
            "per_topic_bits": mi["per_topic_bits"], "across_bits": mi["across_topic_bits"],
            "conf": phase_confidence(L, wall)["conf"], "div": sign_diversity(ph),
            "agree": sign_information(L, ph, wall)["sign_agreement"]}, ph


if __name__ == "__main__":
    print("═══ INNER WALL — select β and τ (1996+ never touched) ═══", flush=True)
    base, _ = score(WALL_INNER, beta=0.0)
    print(f"  β=0 (deployed v9)          AUC {base['auc']:+.4f} NMI {base['nmi']:.4f} F1 {base['f1']:.4f} "
          f"(per-topic {base['per_topic_bits']} bits, across {base['across_bits']})", flush=True)
    grid, results = [(b, t) for b in (0.02, 0.1, 0.5) for t in (0.15, 0.35)], {}
    for b, t in grid:
        r, _ = score(WALL_INNER, beta=b, tau=t)
        results[(b, t)] = r
        print(f"  β={b:<5g} τ={t:<5g}            AUC {r['auc']:+.4f} NMI {r['nmi']:.4f} F1 {r['f1']:.4f} "
              f"(per-topic {r['per_topic_bits']}, across {r['across_bits']}, div {r['div']})", flush=True)
    best = max(results, key=lambda k: results[k]["f1"])
    if results[best]["f1"] <= base["f1"]:
        print(f"\n  INNER SAYS: KL training does NOT help (best {results[best]['f1']:.4f} vs β=0 {base['f1']:.4f}).", flush=True)
        chosen = (0.0, 0.25)
    else:
        chosen = best
        print(f"\n  CHOSEN on the inner wall: β={chosen[0]} τ={chosen[1]} (inner F1 {results[best]['f1']:.4f})", flush=True)

    print("\n═══ OUTER WALL — fit ONCE with the frozen configuration ═══", flush=True)
    out, ph = score(WALL_OUTER, beta=chosen[0], tau=chosen[1])
    print(f"  AUC {out['auc']:+.4f} · NMI {out['nmi']:.4f} · F1 {out['f1']:.4f}  (bar: v9 F1 0.4913)", flush=True)
    print(f"  per-topic {out['per_topic_bits']} bits ↓ · across-topic {out['across_bits']} bits ↑ · "
          f"conf {out['conf']} · diversity {out['div']} · agreement {out['agree']}", flush=True)
    phs = [ph] + [fit(WALL_OUTER, beta=chosen[0], tau=chosen[1], seed=s)[2] for s in (11, 23)]
    print(f"  phase stability {phase_stability(phs)} · SIGN reproducibility {sign_agreement_across_seeds(phs)} "
          f"(v9: 0.9991 / 0.7809)", flush=True)
    json.dump({"chosen_beta": chosen[0], "chosen_tau": chosen[1], "inner": results[best] if chosen[0] else base,
               "outer": out, "stability": phase_stability(phs),
               "sign_reproducibility": sign_agreement_across_seeds(phs)},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kl_phase_result.json"), "w"), indent=1)
    print("KLDONE", flush=True)
