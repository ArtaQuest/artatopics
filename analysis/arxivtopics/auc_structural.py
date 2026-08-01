#!/usr/bin/env python3
"""STRUCTURAL push on the 30-year AUC (operator: "keep competing to push the 30-yr AUC, don't care
about sign information anymore ... ensure only astro data is available at test time").

The single-axis re-ablation under the horizon anchor found NOTHING that beats the deployed model
(22 variants: roster incl. re-adding chiron/sun/venus, detector power, distance laws, weight exponent,
anchor strength+memory, harmonics, dropping the KL term). So this tries the two STRUCTURAL changes
that have never been tested against the anchored model:

  1. DIVERSE ENSEMBLE — average structurally different models (different detector power, weight
     exponent, roster, anchor memory). Round 1 showed a SEED ensemble merely ties; a diversity
     ensemble is a different animal. Members are chosen by GREEDY FORWARD SELECTION on WALL_INNER.

  2. LATENT SHARED FACTORS — the cross-sectional model the competition never got to run:
        F_k(t) = Σᵢ c_kᵢ·cos(θᵢ(t) − q_kᵢ)              K global sky factors, fit on ALL 251 fields
        C_j(t) = b_j + Σ_k λ_jk·F_k(t) + Σᵢ a_jᵢ·cos(θᵢ(t) − p_jᵢ)
     Shared factors are estimated from ~250x more data than any single field's private receiver, so
     they should be far better pinned over a 30-year extrapolation. K swept on WALL_INNER.

Both remain PURE FUNCTIONS OF THE SKY at prediction time — the citation record enters only through
the training loss, its weights and the anchor constant.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/auc_structural.py
"""
import os, sys, json, importlib.util as u
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import torch as T

_s = u.spec_from_file_location("reab", os.path.join(os.path.dirname(os.path.abspath(__file__)), "auc_reablate.py"))
reab = u.module_from_spec(_s); _s.loader.exec_module(reab)
DEV = reab.DEV; REC = reab.REC


# ── 1. DIVERSE ENSEMBLE ──────────────────────────────────────────────────────────────────────
POOL = {
    "base":        {},
    "k=1.5":       {"kpow": 1.5},
    "k=2.5":       {"kpow": 2.5},
    "N^0.6":       {"wexp": 0.6},
    "N^0.9":       {"wexp": 0.9},
    "noKL":        {"beta": 0.0},
    "+sun":        {"bodies": ["sun"] + REC},
    "+venus":      {"bodies": ["venus"] + REC},
    "slow5":       {"bodies": reab.SLOW},
    "anchor_k=3":  {"anchor_k": 3},
    "anchor_k=20": {"anchor_k": 20},
    "lam=0.08":    {"lam_h": 0.08},
}


def ensemble_select():
    print("── fitting the pool on WALL_INNER ──", flush=True)
    P = {}
    for nm, kw in POOL.items():
        P[nm] = reab.fit(WALL_INNER, **kw)
        print(f"   {nm:12s} inner AUC {evaluate(P[nm], WALL_INNER)['auc']:+.4f}", flush=True)
    chosen, cur, best = [], None, -9.0
    print("── greedy forward selection (inner AUC of the AVERAGE) ──", flush=True)
    for _ in range(len(POOL)):
        cand, cand_auc = None, best
        for nm in POOL:
            if nm in chosen: continue
            mix = P[nm] if cur is None else (cur * len(chosen) + P[nm]) / (len(chosen) + 1)
            a = evaluate(mix, WALL_INNER)["auc"]
            if a > cand_auc + 1e-5: cand, cand_auc, cand_mix = nm, a, mix
        if cand is None: break
        chosen.append(cand); cur = cand_mix; best = cand_auc
        print(f"   + {cand:12s} → inner AUC {best:+.4f}  ({len(chosen)} members)", flush=True)
    return chosen, best


# ── 2. LATENT SHARED FACTORS ─────────────────────────────────────────────────────────────────
def fit_factors(wall, K=4, bodies=None, kpow=2.0, wexp=0.75, lam_h=0.03, anchor_k=5,
                seed=7, steps=9000, lr=2e-2, private=True):
    bods = bodies or REC
    bi = [BODIES_ALL.index(b) for b in bods]; nb = len(bi)
    TH = TH_ALL[:, bi]; ne = TH.shape[0]
    inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
    Ysq = np.sqrt(Y)
    T.manual_seed(seed)
    tb = lambda a: T.tensor(np.asarray(a, np.float32), device=DEV)
    cT, sT = tb(np.cos(TH).T), tb(np.sin(TH).T)                      # (nb, ne)
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
    Araw = T.tensor(A0, device=DEV, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (Tn, nb, 1)).astype(np.float32) + rs.randn(Tn, nb, 2).astype(np.float32) * 0.01,
                 device=DEV, requires_grad=True)
    Bp = T.tensor(vmean.astype(np.float32), device=DEV, requires_grad=True)
    Craw = T.tensor(np.full((K, nb), -2.0, np.float32), device=DEV, requires_grad=True)   # factor amps
    Q = T.tensor(np.tile([0.0, 1.0], (K, nb, 1)).astype(np.float32) + rs.randn(K, nb, 2).astype(np.float32) * 0.1,
                 device=DEV, requires_grad=True)                                          # factor phases
    Lam = T.tensor((rs.randn(Tn, K) * 0.01).astype(np.float32), device=DEV, requires_grad=True)  # loadings
    params = [Araw, U, Bp, Craw, Q, Lam] if private else [Bp, Craw, Q, Lam]
    Yt = tb(Ysq); opt = T.optim.Adam(params, lr=lr)
    hz = min(wall + HORIZON, ne)

    def fwd(cx, sx):
        q = T.atan2(Q[:, :, 0], Q[:, :, 1]); c = T.nn.functional.softplus(Craw)
        Fk = (c * T.cos(q)) @ cx + (c * T.sin(q)) @ sx                # (K, ne) shared sky factors
        C = Bp[:, None] + Lam @ Fk
        if private:
            p = T.atan2(U[:, :, 0], U[:, :, 1]); A = T.nn.functional.softplus(Araw)
            C = C + (A * T.cos(p)) @ cx + (A * T.sin(p)) @ sx
        return T.clamp(C, min=1e-4) ** kpow + 1e-8

    best, stall, state = np.inf, 0, None
    for it in range(steps):
        sig = T.sqrt(fwd(cT, sT) + 1e-8)
        per = ((sig[:, :wall] - Yt[:, :wall]).abs() * Wt).sum(1)
        d = (sig[:, wall:hz] - m_anchor) / T.clamp(m_anchor, min=1e-3)
        loss = (per + lam_h * (d ** 2).mean(1)).sum() / Tn
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            lv = loss.item()
            if lv < best - 1e-7: best, stall, state = lv, 0, [x.detach().clone() for x in params]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip(params, state): x.copy_(sv)
        return np.clip(fwd(cT, sT).cpu().numpy(), 0, None)


if __name__ == "__main__":
    print("═══ 1. DIVERSE ENSEMBLE — selected on WALL_INNER ═══", flush=True)
    members, inner_ens = ensemble_select()
    print(f"  chosen: {members}  (inner AUC {inner_ens:+.4f} vs base +0.8929)", flush=True)

    print("\n═══ 2. LATENT SHARED FACTORS — K swept on WALL_INNER ═══", flush=True)
    kres = {}
    for K in (2, 4, 8, 16):
        a = evaluate(fit_factors(WALL_INNER, K=K), WALL_INNER)
        kres[K] = a["auc"]
        print(f"  K={K:<3d} (+private) inner AUC {a['auc']:+.4f}", flush=True)
    a = evaluate(fit_factors(WALL_INNER, K=8, private=False), WALL_INNER)
    kres["8-nopriv"] = a["auc"]
    print(f"  K=8 factors ONLY (no private receiver) inner AUC {a['auc']:+.4f}", flush=True)
    bestK = max((k for k in kres if k != "8-nopriv"), key=lambda k: kres[k])
    print(f"  best K = {bestK} (inner {kres[bestK]:+.4f})", flush=True)

    print("\n═══ OUTER WALL — the two finalists + the deployed base, 3 seeds, fitted once ═══", flush=True)
    out = {}
    base = [evaluate(reab.fit(WALL_OUTER, seed=s), WALL_OUTER) for s in (7, 11, 23)]
    out["deployed v10"] = {"auc": float(np.median([a["auc"] for a in base])),
                           "skill": float(np.median([a["skill"] for a in base])),
                           "pct": float(np.median([a["pct"] for a in base]))}
    ens = []
    for s in (7, 11, 23):
        mix = np.mean([reab.fit(WALL_OUTER, seed=s, **POOL[m]) for m in members], 0)
        ens.append(evaluate(mix, WALL_OUTER))
    out["diverse ensemble"] = {"auc": float(np.median([a["auc"] for a in ens])),
                              "skill": float(np.median([a["skill"] for a in ens])),
                              "pct": float(np.median([a["pct"] for a in ens])), "members": members}
    fac = [evaluate(fit_factors(WALL_OUTER, K=bestK, seed=s), WALL_OUTER) for s in (7, 11, 23)]
    out[f"latent factors K={bestK}"] = {"auc": float(np.median([a["auc"] for a in fac])),
                                        "skill": float(np.median([a["skill"] for a in fac])),
                                        "pct": float(np.median([a["pct"] for a in fac]))}
    for k, v in out.items():
        print(f"  {k:24s} OUTER AUC {v['auc']:+.4f} · skill {v['skill']:+.4f} · {v['pct']:.1f}%>0", flush=True)
    win = max(out, key=lambda k: out[k]["auc"])
    print(f"\n  WINNER: {win} → {out[win]['auc']:+.4f}", flush=True)
    json.dump({"inner_ensemble": inner_ens, "members": members, "K_sweep": {str(k): v for k, v in kres.items()},
               "outer": out, "winner": win},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "auc_structural.json"), "w"), indent=1)
    print("STRUCTDONE", flush=True)
