#!/usr/bin/env python3
"""SINGLE PHASE PER FIELD vs the current SOTA (operator 2026-07-25: "compare single phase parameter
vs current sota").

SOTA (v10):   every field has its OWN tuning for EVERY body — 7 free phases per field.
                  C_j(t) = b_j + Σᵢ a_jᵢ·cos(θᵢ(t) − p_jᵢ)              15 params/field
SINGLE PHASE: the sky's geometry is SHARED — one global phase per body, estimated from all 251 fields
              at once — and a field differs from the atlas by exactly ONE angle:
                  p_jᵢ = P_ᵢ + φ_j
                  C_j(t) = b_j + Σᵢ a_jᵢ·cos(θᵢ(t) − Pᵢ − φ_j)          9 params/field + 7 shared
              This is the cleanest possible meaning of "the field's phase": one number, pinned by the
              whole record rather than by that field's own thin history. Cross-sectional by construction.

Everything else is held IDENTICAL so the comparison isolates the phase parameterisation: same 7 bodies,
same rectified square-law detector max(C,0)², same N^¾-weighted L1 on √share, same horizon anchor,
same twelve-sign KL term, same walls, same seeds. Selection on WALL_INNER; WALL_OUTER scored once.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/single_phase_fit.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import torch as T

DEV = "mps" if T.backends.mps.is_available() else "cpu"
BODIES = CHAMPION_BODIES
PL = BODIES.index("pluto")
CENTRES = T.tensor(np.deg2rad(np.arange(12) * 30.0 + 15.0).astype(np.float32), device=DEV)


def fit(wall, single=True, beta=0.02, tau=0.15, seed=7, steps=9000, lr=2e-2,
        lam_h=0.03, anchor_k=5, wexp=0.75):
    bi = [BODIES_ALL.index(b) for b in BODIES]; nb = len(bi)
    TH = TH_ALL[:, bi]; ne = TH.shape[0]
    inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
    Ysq = np.sqrt(Y)
    T.manual_seed(seed)
    tb = lambda a: T.tensor(np.asarray(a, np.float32), device=DEV)
    tv = train_mask(wall).astype(np.float32)
    wy = np.clip(N[:wall], 0, None) ** wexp
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9); Wt = tb(W)
    Wa = np.zeros_like(W); Wa[:, wall - anchor_k:] = (tv * wy[None])[:, wall - anchor_k:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    m_anchor = tb(((Ysq[:, :wall] * Wa).sum(1))[:, None])
    vmean = (Ysq[:, :wall] * tv).sum(1) / np.maximum(tv.sum(1), 1.0)

    A0 = np.full((Tn, nb), -2.0, np.float32); A0[:, PL] = inv_sp(np.clip(vmean, 1e-3, None))
    Araw = T.tensor(A0, device=DEV, requires_grad=True)
    Bp = T.tensor(vmean.astype(np.float32), device=DEV, requires_grad=True)
    rs = np.random.RandomState(seed)
    params = [Araw, Bp]
    if single:
        # ONE angle per field (φ_j) + ONE shared global phase per body (P_i)
        Uf = T.tensor(np.tile([0.0, 1.0], (Tn, 1)).astype(np.float32) + rs.randn(Tn, 2).astype(np.float32) * 0.01,
                      device=DEV, requires_grad=True)
        Ug = T.tensor(np.tile([0.0, 1.0], (nb, 1)).astype(np.float32) + rs.randn(nb, 2).astype(np.float32) * 0.01,
                      device=DEV, requires_grad=True)
        params += [Uf, Ug]
    else:
        U = T.tensor(np.tile([0.0, 1.0], (Tn, nb, 1)).astype(np.float32) + rs.randn(Tn, nb, 2).astype(np.float32) * 0.01,
                     device=DEV, requires_grad=True)
        params += [U]
    Yt = tb(Ysq); opt = T.optim.Adam(params, lr=lr)
    hz = min(wall + HORIZON, ne)
    cth, sth = tb(np.cos(TH).T), tb(np.sin(TH).T)          # (nb, ne)

    def phases():
        """Returns the effective per-field, per-body tuning and the field's classifying phase."""
        if single:
            phi = T.atan2(Uf[:, 0], Uf[:, 1])              # (Tn,)  the ONE angle per field
            P = T.atan2(Ug[:, 0], Ug[:, 1])                # (nb,)  shared sky geometry
            p = P[None, :] + phi[:, None]                  # (Tn, nb)
            cls = p[:, PL]                                 # classify by the same quantity as SOTA
        else:
            p = T.atan2(U[:, :, 0], U[:, :, 1])
            cls = p[:, PL]
        return p, cls

    def fwd(cx, sx):
        p, _ = phases()
        A = T.nn.functional.softplus(Araw)
        C = Bp[:, None] + (A * T.cos(p)) @ cx + (A * T.sin(p)) @ sx
        return T.clamp(C, min=1e-4) ** 2 + 1e-8

    def kl_phase():
        _, cls = phases()
        q = T.softmax(T.cos(cls[:, None] - CENTRES[None, :]) / tau, dim=1)
        Hj = -(q * T.log(q.clamp_min(1e-12))).sum(1).mean()
        qb = q.mean(0)
        Hb = -(qb * T.log(qb.clamp_min(1e-12))).sum()
        return -(Hb - Hj)

    best, stall, state = np.inf, 0, None
    for it in range(steps):
        sig = T.sqrt(fwd(cth, sth) + 1e-8)
        per = ((sig[:, :wall] - Yt[:, :wall]).abs() * Wt).sum(1)
        d = (sig[:, wall:hz] - m_anchor) / T.clamp(m_anchor, min=1e-3)
        loss = (per + lam_h * (d ** 2).mean(1)).sum() / Tn
        if beta: loss = loss + beta * kl_phase()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            lv = loss.item()
            if lv < best - 1e-7: best, stall, state = lv, 0, [x.detach().clone() for x in params]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip(params, state): x.copy_(sv)
        yhat = np.clip(fwd(cth, sth).cpu().numpy(), 0, None)
        _, cls = phases()
        ph = np.rad2deg(cls.cpu().numpy()) % 360.0

        def predict_delta(deg):
            THx = TH + np.deg2rad(float(deg))
            with T.no_grad():
                return np.clip(fwd(tb(np.cos(THx).T), tb(np.sin(THx).T)).detach().cpu().numpy(), 0, None)
    npar = Tn * (nb + 1 + (1 if single else nb)) + (nb if single else 0)
    return yhat, predict_delta, ph, npar


def score(wall, **kw):
    yh, pdel, ph, npar = fit(wall, **kw)
    ev = evaluate(yh, wall)
    L = phase_curve(pdel, wall)
    mi = sign_mutual_information(L, ph, wall)
    return {"auc": ev["auc"], "skill": ev["skill"], "pct": ev["pct"], "nmi": mi["nmi"],
            "f1": global_f1(ev["auc"], mi["nmi"]), "per_topic_bits": mi["per_topic_bits"],
            "across_bits": mi["across_topic_bits"], "conf": phase_confidence(L, wall)["conf"],
            "div": sign_diversity(ph), "agree": sign_information(L, ph, wall)["sign_agreement"],
            "params": npar}, ph


if __name__ == "__main__":
    print("═══ INNER WALL — is the constraint worth it? (1996+ untouched) ═══", flush=True)
    for single in (False, True):
        r, _ = score(WALL_INNER, single=single)
        print(f"  {'SINGLE phase' if single else 'SOTA (7 phases)'}: AUC {r['auc']:+.4f} NMI {r['nmi']:.4f} "
              f"F1 {r['f1']:.4f} · params {r['params']} · per-topic {r['per_topic_bits']} across {r['across_bits']}", flush=True)

    print("\n═══ OUTER WALL — 3 seeds each, reported once ═══", flush=True)
    out = {}
    for single in (False, True):
        rows, phs = [], []
        for sd in (7, 11, 23):
            r, ph = score(WALL_OUTER, single=single, seed=sd); rows.append(r); phs.append(ph)
        med = {k: float(np.median([r[k] for r in rows])) for k in
               ("auc", "skill", "pct", "nmi", "f1", "per_topic_bits", "across_bits", "conf", "div", "agree")}
        med["params"] = rows[0]["params"]
        med["stability"] = phase_stability(phs)
        med["sign_repro"] = sign_agreement_across_seeds(phs)
        med["f1_spread"] = [min(r["f1"] for r in rows), max(r["f1"] for r in rows)]
        out["single" if single else "sota"] = med
        tag = "SINGLE phase" if single else "SOTA (7 phases)"
        print(f"  {tag:16s} AUC {med['auc']:+.4f} · NMI {med['nmi']:.4f} · F1 {med['f1']:.4f} "
              f"[{med['f1_spread'][0]:.4f}..{med['f1_spread'][1]:.4f}]", flush=True)
        print(f"  {'':16s} params {med['params']} · per-topic {med['per_topic_bits']:.3f} bits ↓ · "
              f"across {med['across_bits']:.3f} ↑ · conf {med['conf']:.4f}", flush=True)
        print(f"  {'':16s} skill {med['skill']:+.4f} ({med['pct']:.1f}%>0) · diversity {med['div']:.4f} · "
              f"agreement {med['agree']:.4f} · stability {med['stability']:.4f} · sign-repro {med['sign_repro']:.4f}", flush=True)
    s, v = out["single"], out["sota"]
    print(f"\n  VERDICT: single-phase vs SOTA → ΔAUC {s['auc']-v['auc']:+.4f} · ΔNMI {s['nmi']-v['nmi']:+.4f} · "
          f"ΔF1 {s['f1']-v['f1']:+.4f} · Δparams {s['params']-v['params']:+d} · "
          f"Δsign-repro {s['sign_repro']-v['sign_repro']:+.4f}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "single_phase_result.json"), "w"), indent=1)
    print("SINGLEDONE", flush=True)
