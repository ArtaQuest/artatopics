#!/usr/bin/env python3
"""ONE MULTI-HEAD MODEL OVER THE WHOLE DISTRIBUTION (operator 2026-08-04):

    y_j(t) = ( b_j + Σᵢ w_ij · sin θᵢ(t) )²          one head per topic, everything fit at once
    p_j(t) = y_j(t) / Σ_k y_k(t)                      the predicted DISTRIBUTION of topics given the date

trained against the empirical share distribution q_j(t) with a cross-entropy objective — and with the
optimum delivered ANALYTICALLY, which needs one honest sentence of mathematics before any code:

  CROSS-ENTROPY OVER THIS CLASS HAS NO UNIQUE CLOSED FORM. Negating any single head's weights leaves
  every p_j(t) unchanged, so CE has 2^251 global optima and saddle ridges between them — "the"
  analytic CE solution cannot exist. What DOES exist, exactly and uniquely, is the global optimum of
  the AMPLITUDE form of the same family: with amplitudes a_j(t) = u_j · z(t) linear in the features,

      min_U  Σ_t n_t Σ_j ( √q_j(t) − a_j(t) )²                                (CONVEX ⇒ global)

  whose normal equations share ONE Gram matrix G = Σ_t n_t z_t z_tᵀ across all 251 heads:

      U* = S Zᵀ N Z (ZᵀNZ)⁻¹      — one d×d inverse fits the entire model.

  On normalised rows this objective equals 2−2·Bhattacharyya(q,p) — the fidelity member of the CE
  family, and CE's own second-order approximation around the optimum. So the procedure is: solve the
  amplitude problem in closed form (the canonical +√q branch), then run a DETERMINISTIC full-batch
  polish of true CE from that start and MEASURE the gap. If the polish moves nothing, the analytic
  solution simply is the CE optimum that matters, demonstrated rather than asserted.

Features are the DATE and nothing else: z(t) = [1, sin θᵢ(t)] for the seven bodies (as specified);
the sin+cos variant is reported beside it because sin alone fixes every head's phase reference, and
the comparison is one extra solve. No anchor, no per-topic mask games: before a topic emerges its
true share IS zero, and a distribution model is supposed to say so.

Scored on the same honest walls as everything else in this repository, against the same bars.

  python3 analysis/arxivtopics/multihead_ce.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import arxiv_fit as af

names, Yv, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
J, n = Yv.shape
ne = TH.shape[0]
Q = Yv / 100.0                                       # empirical distribution: Σ_j q_j(t) = 1 (≈)
NW = np.clip(af.META["evidence"], 0, None) ** 0.75   # the campaign's rehearsal-selected year weight


def features(kind):
    if kind == "sin":
        Z = np.concatenate([np.ones((ne, 1)), np.sin(TH)], 1)
    else:
        Z = np.concatenate([np.ones((ne, 1)), np.sin(TH), np.cos(TH)], 1)
    return Z


def solve(Z, wall, anchored=True):
    """THE ANALYTIC GLOBAL OPTIMUM of the amplitude objective — optionally with the horizon anchor
    folded in as extra rows, exactly as the per-topic record model does it. The anchor is a LINEAR
    constraint (each head's own forecast-window amplitude pulled toward the level √q̄_j it held over
    the last ANCHOR_K training years, weighted λ/m_j² so it is scale-free), so the problem stays
    convex and closed-form; the per-head weight just makes the Gram per-head — 251 batched d×d
    solves instead of one shared inverse. λ = the campaign's 0.03, unchanged, chosen by the same
    rehearsal that chose it for the record model — nothing new is tuned here."""
    d = Z.shape[1]
    # NORMALISED weights, the record model's convention — λ's rehearsal calibration assumes the data
    # term sums to 1. Unnormalised, the data Gram out-masses the anchor rows ~10⁴:1 and the anchor
    # silently becomes a no-op (measured: identical to four decimals before this line existed).
    Zt = Z[:wall]; w = NW[:wall] / NW[:wall].sum()
    S = np.sqrt(Q[:, :wall])
    G0 = Zt.T @ (Zt * w[:, None]) + 1e-10 * np.eye(d)
    B0 = (S * w[None, :]) @ Zt                             # (J, d)
    if not anchored:
        return np.linalg.solve(G0, B0.T).T
    hz = min(wall + af.HORIZON, ne)
    Za = Z[wall:hz]
    Ga = Za.T @ Za                                         # shared anchor Gram shape
    tail = S[:, max(0, wall - af.ANCHOR_K):wall]
    m = np.maximum(tail.mean(1), 1e-4)                     # each head's recent amplitude level
    aw = af.LAM_HORIZON / (m ** 2) / max(hz - wall, 1)     # scale-free, per head
    G = G0[None, :, :] + aw[:, None, None] * Ga[None, :, :]
    B = B0 + (aw * m)[:, None] * Za.sum(0)[None, :]
    return np.linalg.solve(G, B[:, :, None])[:, :, 0]      # (J, d) batched


def predict(U, Z):
    A = U @ Z.T                                            # (J, ne)
    Y = A ** 2
    return 100.0 * Y / np.maximum(Y.sum(0, keepdims=True), 1e-12), A


def ce(U, Z, lo, hi):
    """Cross-entropy of the model distribution against the empirical one, per year-weighted."""
    P, _ = predict(U, Z)
    p = np.clip(P / 100.0, 1e-12, None)
    w = NW[lo:hi]
    return float(-(Q[:, lo:hi] * np.log(p[:, lo:hi])).sum(0) @ w / w.sum())


def polish(U, Z, wall, iters=300):
    """Deterministic full-batch CE descent with backtracking — measures how far the analytic
    amplitude optimum sits from the true CE optimum. No seed, no stochasticity."""
    Zt = Z[:wall]; w = NW[:wall]; w = w / w.sum()
    U = U.copy(); step = 1e-3
    def loss(U):
        A = U @ Zt.T; Y = A ** 2 + 1e-12
        P = Y / Y.sum(0, keepdims=True)
        return float(-(Q[:, :wall] * np.log(P)).sum(0) @ w), A, P
    L, A, P = loss(U)
    for it in range(iters):
        r2 = (A ** 2 + 1e-12).sum(0, keepdims=True)
        # dL/dA = -2w( q/a − a·(Σq)/r² );  Σ_j q_jt ≈ 1
        gA = -2.0 * w[None, :] * (Q[:, :wall] / np.where(np.abs(A) > 1e-9, A, 1e-9)
                                  - A * Q[:, :wall].sum(0, keepdims=True) / r2)
        gU = gA @ Zt
        while step > 1e-12:
            Ln, An, Pn = loss(U - step * gU)
            if Ln < L: break
            step *= 0.5
        if step <= 1e-12: break
        U = U - step * gU; L, A, P = Ln, An, Pn; step *= 1.3
    return U, L


def kl_at(P, wall, hi=None):
    hi = min(wall + af.HORIZON, n) if hi is None else hi
    w = NW[wall:hi]; w = w / w.sum()
    Hq = float(-(np.where(Q[:, wall:hi] > 0, Q[:, wall:hi] * np.log(np.clip(Q[:, wall:hi], 1e-12, None)), 0)).sum(0) @ w)
    pd = np.clip(P[:, wall:hi] / 100.0, 1e-12, None); pd = pd / pd.sum(0, keepdims=True)
    return float(-(Q[:, wall:hi] * np.log(pd)).sum(0) @ w) - Hq


def kl_pers(wall):
    hi = min(wall + af.HORIZON, n)
    return kl_at(np.repeat(Yv[:, wall - 1:wall], n - wall, 1) if False else
                 np.concatenate([np.zeros((J, wall)), np.repeat(Yv[:, wall - 1:wall], hi - wall, 1)], 1), wall, hi)


def dist_metrics(P, wall):
    """THE NATIVE SCORE for a date→distribution model: held-out cross-entropy and its decomposition
    CE = H(q) + KL(q‖p). H(q) is the floor no model can beat; the KL is what the model owes. The two
    honest baselines are the train-mean distribution and carrying the last training year forward."""
    hi = n
    w = NW[wall:hi]; w = w / w.sum()
    Hq = float(-(np.where(Q[:, wall:hi] > 0, Q[:, wall:hi] * np.log(np.clip(Q[:, wall:hi], 1e-12, None)), 0)).sum(0) @ w)
    def ce_of(Pd):
        p = np.clip(Pd, 1e-12, None); p = p / p.sum(0, keepdims=True)
        return float(-(Q[:, wall:hi] * np.log(p)).sum(0) @ w)
    model = ce_of(P[:, wall:hi] / 100.0)
    tw = NW[:wall]
    mean_d = (Q[:, :wall] * tw[None, :]).sum(1, keepdims=True) / tw.sum()
    pers_d = Q[:, wall - 1:wall]
    return {"H_q": Hq, "model": model, "train_mean": ce_of(np.repeat(mean_d, hi - wall, 1)),
            "persistence": ce_of(np.repeat(pers_d, hi - wall, 1))}


# ── the TUNED-HEAD variant (operator 2026-08-04): y_j = (b_j + Σᵢ w_ij·sin(θᵢ − p_j))² ──────────
# Each head gains ONE tuning phase shared by its seven bodies — exactly the restriction the AUC
# campaign measured as optimal (one continuous phase + signed per-body weights), now in the
# distribution setting. Still analytic: with p_j fixed, sin(θᵢ−p_j) = cosp·sinθᵢ − sinp·cosθᵢ is
# linear in the same features, so a 1° HALF-circle sweep (p+180° ≡ negate w — the same gauge as the
# record model) with the anchored closed-form solve at each candidate is the exact global optimum to
# grid resolution. NO rectifier, per the spec: the amplitude is squared directly, so a negative
# projection still radiates — the one structural difference from the record receiver.
GRID = np.deg2rad(np.arange(0.0, 180.0, 1.0))


def solve_tuned(wall):
    ST, CT = np.sin(TH), np.cos(TH)                         # (ne, 7)
    d = 1 + TH.shape[1]; NG = len(GRID)
    w = NW[:wall] / NW[:wall].sum()
    S = np.sqrt(Q[:, :wall])
    hz = min(wall + af.HORIZON, ne)
    tail = S[:, max(0, wall - af.ANCHOR_K):wall]
    m = np.maximum(tail.mean(1), 1e-4)
    aw = af.LAM_HORIZON / (m ** 2) / max(hz - wall, 1)      # (J,) scale-free anchor weights
    # features at every grid tuning: Z[g,t,:] = [1, sin(θᵢ(t) − p_g)]
    Z = np.empty((NG, ne, d)); Z[:, :, 0] = 1.0
    Z[:, :, 1:] = np.cos(GRID)[:, None, None] * ST[None] - np.sin(GRID)[:, None, None] * CT[None]
    Zt, Za = Z[:, :wall], Z[:, wall:hz]
    Gd = np.einsum('gtp,t,gtq->gpq', Zt, w, Zt) + 1e-10 * np.eye(d)[None]
    Ga = np.einsum('gtp,gtq->gpq', Za, Za)
    Bd = np.einsum('jt,t,gtp->jgp', S, w, Zt)               # (J, NG, d)
    Sa = Za.sum(1)                                          # (NG, d)
    A = np.zeros((J, ne)); P_j = np.zeros(J, int)
    for j in range(J):
        Gj = Gd + aw[j] * Ga
        bj = Bd[j] + aw[j] * m[j] * Sa
        c = np.linalg.solve(Gj, bj[:, :, None])[:, :, 0]    # (NG, d)
        fit = np.einsum('gtp,gp->gt', Zt, c)
        r = ((fit - S[j, None, :]) ** 2) @ w + aw[j] * ((np.einsum('gtp,gp->gt', Za, c) - m[j]) ** 2).sum(1)
        g = int(np.argmin(r)); P_j[j] = g
        A[j] = Z[g] @ c[g]
    Yh = A ** 2
    return 100.0 * Yh / np.maximum(Yh.sum(0, keepdims=True), 1e-12), A, P_j


def auc_at(Yh, wall):
    tv = af.META["topic_valid"]; tvw = tv[:, :wall].astype(float)
    mu = (Yv[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    return float(np.mean([1.0 - ((Yv[:, wall + h] - Yh[:, wall + h]) ** 2).sum() /
                max(((Yv[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(30)]))


def main():
    WALLS = list(range(n - 63, n - 29, 3))
    out = {}
    print(f"═══ MULTI-HEAD DISTRIBUTION MODEL · {J} heads · one Gram inverse per wall ═══", flush=True)
    for kind in ("sin", "sincos"):
      for anchored in (False, True):
        Z = features(kind); d = Z.shape[1]
        t0 = time.time()
        aucs, aucs_pol = [], []
        for w in WALLS:
            U = solve(Z, w, anchored)
            P, _ = predict(U, Z)
            aucs.append(auc_at(P, w))
            Up, _ = polish(U, Z, w)
            Pp, _ = predict(Up, Z)
            aucs_pol.append(auc_at(Pp, w))
        wall = n - 30
        U = solve(Z, wall, anchored)
        ce_an_tr, ce_an_te = ce(U, Z, 0, wall), ce(U, Z, wall, n)
        Up, _ = polish(U, Z, wall)
        ce_po_tr, ce_po_te = ce(Up, Z, 0, wall), ce(Up, Z, wall, n)
        a = np.array(aucs); ap = np.array(aucs_pol)
        P96, _ = predict(solve(Z, wall, anchored), Z)
        dm = dist_metrics(P96, wall)
        print(f"\n  z(t) = [1, {'sinθ' if kind=='sin' else 'sinθ, cosθ'}]{' + anchor rows' if anchored else ''} "
              f"· {J}×{d} = {J*d} params · {time.time()-t0:.1f}s for 12 origins", flush=True)
        print(f"    DISTRIBUTION (the goal) · held-out 1996-2025: CE model {dm['model']:.4f} · "
              f"persistence {dm['persistence']:.4f} · train-mean {dm['train_mean']:.4f} · floor H(q) {dm['H_q']:.4f}", flush=True)
        print(f"      KL(q‖·): model {dm['model']-dm['H_q']:.4f} · persistence {dm['persistence']-dm['H_q']:.4f} · "
              f"train-mean {dm['train_mean']-dm['H_q']:.4f}", flush=True)
        print(f"    analytic solve   12-origin mean {a.mean():+.4f} · 1996 {a[-1]:+.4f}", flush=True)
        print(f"    + CE polish      12-origin mean {ap.mean():+.4f} · 1996 {ap[-1]:+.4f} "
              f"(Δ {ap.mean()-a.mean():+.4f})", flush=True)
        print(f"    CE at 1996 wall  analytic train {ce_an_tr:.4f} test {ce_an_te:.4f} · "
              f"polished train {ce_po_tr:.4f} test {ce_po_te:.4f}", flush=True)
        out[f"{kind}{'_anchored' if anchored else ''}"] = {"params": J * d, "dist_1996": {k: round(v, 4) for k, v in dm.items()}, "auc": [round(v, 4) for v in aucs],
                     "auc_polished": [round(v, 4) for v in aucs_pol],
                     "mean": round(float(a.mean()), 4), "mean_polished": round(float(ap.mean()), 4),
                     "auc_1996": round(float(a[-1]), 4), "auc_1996_polished": round(float(ap[-1]), 4),
                     "ce_1996": {"analytic": [round(ce_an_tr, 4), round(ce_an_te, 4)],
                                  "polished": [round(ce_po_tr, 4), round(ce_po_te, 4)]}}
    # ── THE TUNED-HEAD MODEL across the twelve origins ──────────────────────────────────────
    print(f"\n  y_j = (b_j + Σ w_ij·sin(θᵢ − p_j))² · one tuning per head · {J}×9 = {J*9} params", flush=True)
    t0 = time.time()
    tuned_auc, tuned_kl = [], []
    for w in WALLS:
        P, _, _ = solve_tuned(w)
        tuned_auc.append(auc_at(P, w)); tuned_kl.append(kl_at(P, w))
    ta, tk = np.array(tuned_auc), np.array(tuned_kl)
    print(f"    12-origin share-AUC mean {ta.mean():+.4f} · 1996 {ta[-1]:+.4f}   [{time.time()-t0:.0f}s]", flush=True)
    print(f"    12-origin KL mean {tk.mean():.4f}   per-origin " + " ".join(f"{v:.3f}" for v in tk), flush=True)
    P96, A96, PJ = solve_tuned(n - 30)
    dm96 = dist_metrics(P96, n - 30)
    print(f"    1996 held-out: CE {dm96['model']:.4f} · KL {dm96['model']-dm96['H_q']:.4f} "
          f"(persistence {dm96['persistence']-dm96['H_q']:.4f})", flush=True)
    occ = np.bincount((np.degrees(GRID[PJ]).round().astype(int) % 180) // 15, minlength=12)
    print(f"    tuning occupancy (15° bins over the half-circle): " + " ".join(map(str, occ)), flush=True)
    out["tuned_head"] = {"params": J * 9, "auc": [round(v, 4) for v in tuned_auc],
                         "auc_mean": round(float(ta.mean()), 4), "auc_1996": round(float(ta[-1]), 4),
                         "kl": [round(v, 4) for v in tuned_kl], "kl_mean": round(float(tk.mean()), 4),
                         "dist_1996": {k: round(v, 4) for k, v in dm96.items()}}

    # ── the fair native comparison: KL over ALL twelve origins, and the per-topic record model
    #    renormalised into a distribution (the strongest date→distribution predictor available) ──
    Zsc = features("sincos")
    kl_model, kl_p, kl_rec = [], [], []
    for w in WALLS:
        U = solve(Zsc, w, True)
        P, _ = predict(U, Zsc)
        kl_model.append(kl_at(P, w))
        kl_p.append(kl_pers(w))
        Prec, _ = af.fit_final(Yv, TH, w)
        kl_rec.append(kl_at(100.0 * Prec / np.maximum(Prec.sum(0, keepdims=True), 1e-12), w))
    print(f"\n  KL(q‖p) OVER ALL TWELVE ORIGINS (the goal's native score, lower is better):", flush=True)
    print(f"    multi-head anchored (sincos)   mean {np.mean(kl_model):.4f}   per-origin " +
          " ".join(f"{v:.3f}" for v in kl_model), flush=True)
    print(f"    persistence distribution       mean {np.mean(kl_p):.4f}   per-origin " +
          " ".join(f"{v:.3f}" for v in kl_p), flush=True)
    print(f"    per-topic record, renormalised mean {np.mean(kl_rec):.4f}   per-origin " +
          " ".join(f"{v:.3f}" for v in kl_rec), flush=True)
    out["kl_12origin"] = {"multihead_anchored_sincos": [round(v, 4) for v in kl_model],
                          "persistence": [round(v, 4) for v in kl_p],
                          "record_renormalised": [round(v, 4) for v in kl_rec]}
    print(f"\n    reference AUC: per-topic record +0.8751 (12-origin) / +0.7990 (1996) · "
          f"persistence +0.8511 / +0.7344", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "multihead_ce.json"), "w"), indent=1)
    print("MHDONE", flush=True)


if __name__ == "__main__":
    main()
