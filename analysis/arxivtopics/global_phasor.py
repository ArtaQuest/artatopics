#!/usr/bin/env python3
"""THE GLOBAL PHASOR, CLOSED FORM VIA THE SQUARE (operator 2026-08-07, final):

    y_j(t) = | b + SUM_i a_i * exp( i * (theta_i(t) - p_ji) ) |^2

The square is the whole trick. Expanding the modulus squared:

    y = b^2 + SUM_i a_i^2                                        (a constant)
      + SUM_i 2 b a_i * cos( theta_i(t) - p_ji )                 (TRANSITS: each planet past the topic's point)
      + SUM_{i<k} 2 a_i a_k * cos( (theta_i - theta_k) - (p_ji - p_jk) )   (ASPECTS: planet pairs)

which is LINEAR in 57 fixed features of the date: {1, cos theta_i, sin theta_i, cos(theta_i-theta_k),
sin(theta_i-theta_k)}. So the fit is one weighted least-squares solve per topic -- closed form, no
optimiser, no seed -- exactly like the record model's machinery, and the expansion is the
astrological reading itself: a baseline, seven transit terms, twenty-one aspect terms.

TWO STAGES, BOTH CLOSED FORM.
  STAGE 1 (the relaxation): per topic, WLS of its normalised share on the 57 features, with the
  horizon anchor as extra rows and a RIDGE on the non-constant features. The ridge is not taste: the
  slow features are nearly constant inside a 296-year window (the neptune-pluto separation has a
  ~493-year period), so the unridged design is collinear -- its predictions are fine but its
  COEFFICIENTS are arbitrary splits, and stage 2 reads the coefficients. The strength is chosen on
  the nine early origins only, by the exact model's own score.
  STAGE 2 (the exact structure): the operator's spec makes b and a_i GLOBAL and leaves each topic
  only its phases. Recover that from stage 1 without iteration: per topic and body, the transit
  coefficients (alpha_i, beta_i) give the phase p_ji = atan2(beta_i, alpha_i) and a magnitude
  M_ji = sqrt(alpha^2+beta^2) = 2 b a_i; pool M_i = median_j M_ji; the pooled constant C0 = b^2 +
  SUM a_i^2 with a_i = M_i/(2b) yields the quartic b^4 - C0 b^2 + SUM M_i^2/4 = 0, solved in closed
  form (the larger root -- the level dominates the arrows). Rebuild the EXACT model from (b, a, P)
  and score it. Each topic is then fully characterised by its seven phases -- its signs.

SCALE, stated plainly: global amplitudes cannot span the two orders of magnitude between fields'
shares, so each topic is fitted on y normalised by the square of its own training-mean level (a
measured statistic behind each wall, never a parameter). Prediction = level^2 * model, clipped at 0
(the relaxation may dip negative; the exact form cannot).

  python3 analysis/arxivtopics/global_phasor.py            # twelve origins + headline
  python3 analysis/arxivtopics/global_phasor.py headline   # the CI gate
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import arxiv_fit as af

names, Y, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
n = Y.shape[1]; ne = TH.shape[0]; nb = TH.shape[1]; Tn = Y.shape[0]
WALLS = list(range(n - 63, n - 29, 3))
PAIRS = [(i, k) for i in range(nb) for k in range(i + 1, nb)]

# the 57 features of the date, built once
F = [np.ones(ne)]
for i in range(nb): F += [np.cos(TH[:, i]), np.sin(TH[:, i])]
for i, k in PAIRS:
    D = TH[:, i] - TH[:, k]; F += [np.cos(D), np.sin(D)]
F = np.stack(F, 1)                                               # (ne, 57)
NF = F.shape[1]


def prep(wall):
    tv = af.META["topic_valid"][:, :wall].astype(float)
    wy = np.clip(af.META["evidence"][:wall], 0, None) ** 0.75
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    Wa = np.zeros_like(W); Wa[:, wall - af.ANCHOR_K:] = (tv * wy[None])[:, wall - af.ANCHOR_K:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    lvl2 = np.maximum((Y[:, :wall] * W).sum(1), 1e-8)             # mean share level (y scale)
    anc2 = np.maximum((Y[:, :wall] * Wa).sum(1), 1e-8)            # anchor level (y scale)
    return W, lvl2, anc2, min(wall + af.HORIZON, ne)


def fit_wall(wall, ridge=0.01):
    W, lvl2, anc2, hz = prep(wall)
    Yn = Y / lvl2[:, None]                                        # normalised share
    mt = anc2 / lvl2                                              # normalised anchor target
    Ft, Fa = F[:wall], F[wall:hz]
    FaS = Fa.T @ Fa; FaB = Fa.sum(0)
    Rg = np.eye(NF); Rg[0, 0] = 0.0                               # never shrink the constant
    coef = np.zeros((Tn, NF))
    for j in range(Tn):
        aw = af.LAM_HORIZON / (mt[j] ** 2) / max(hz - wall, 1)
        A = Ft.T @ (Ft * W[j][:, None]) + aw * FaS + ridge * Rg + 1e-8 * np.eye(NF)
        b = Ft.T @ (W[j] * Yn[j, :wall]) + aw * FaB * mt[j]
        coef[j] = np.linalg.solve(A, b)
    P_relax = np.clip(coef @ F.T, 0, None) * lvl2[:, None]        # stage 1: the 57-param relaxation

    # ── stage 2: project onto the EXACT global-phasor structure, all in closed form ──
    alpha = coef[:, 1:1 + 2 * nb:2]; beta = coef[:, 2:2 + 2 * nb:2]     # transit coefficients (Tn, 7)
    Pji = np.arctan2(beta, alpha)                                 # each topic's phases — its signs
    M = np.median(np.sqrt(alpha ** 2 + beta ** 2), 0)             # pooled 2*b*a_i per body
    C0 = float(np.median(coef[:, 0]))                             # pooled b^2 + SUM a_i^2
    C0 = max(C0, 1e-6)
    disc = C0 ** 2 - (M ** 2).sum()
    if disc < 0:                                                  # infeasible split: project onto the
        M = M * np.sqrt(0.5) * C0 / np.sqrt((M ** 2).sum())       # boundary (level = arrows in energy)
        disc = 0.0
    b2 = (C0 + np.sqrt(disc)) / 2.0
    bg = np.sqrt(max(b2, 1e-9)); ag = M / (2 * bg)

    # ── stage 2b (operator, confirmed spec): y_j = |b_j + SUM_i a_i e^{i(theta_i - p_ij)}|^2 —
    # b PER TOPIC, arrows global. The first attempt split the transit magnitudes by rank-1 SVD and
    # collapsed (ridge-biased magnitudes are not cleanly rank-1; unit-vector scaling threw most
    # levels far from 1 and the reconstruction exploded to -2 AUC — kept in git history). The robust
    # closed form: hold the global arrows and the phases from stage 2 fixed; then for each topic the
    # weighted objective is a QUARTIC in b_j alone, its derivative a CUBIC, solved exactly by
    # np.roots — the real nonnegative root with the lowest objective. Deterministic, no scan.
    Dm2 = TH[None, :, :] - Pji[:, None, :]
    ReS = (np.cos(Dm2) * ag[None, None, :]).sum(2)                # (Tn, ne)
    ImS = (np.sin(Dm2) * ag[None, None, :]).sum(2)
    Q = ReS ** 2 + ImS ** 2
    bj = np.zeros(Tn)
    for j in range(Tn):
        w = W[j]; hzs = slice(wall, hz)
        awj = af.LAM_HORIZON / (mt[j] ** 2) / max(hz - wall, 1)
        # objective: sum_t v_t (g_t - b^2 - 2 b R_t)^2 over train (g = y_norm - Q) and anchor rows
        v = np.concatenate([w, np.full(hz - wall, awj)])
        g = np.concatenate([Yn[j, :wall] - Q[j, :wall], mt[j] ** 2 - Q[j, hzs]])
        R = np.concatenate([ReS[j, :wall], ReS[j, hzs]])
        # d/db: sum v (g - b^2 - 2bR)(b + R) = 0  ->  cubic  c3 b^3 + c2 b^2 + c1 b + c0 = 0
        S0 = v.sum(); S1 = (v * R).sum(); S2 = (v * R * R).sum()
        G0 = (v * g).sum(); G1 = (v * g * R).sum()
        c3 = -S0; c2 = -3 * S1; c1 = G0 - 2 * S2; c0 = G1
        roots = np.roots([c3, c2, c1, c0])
        cand = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
        if not cand: cand = [float(np.sqrt(max(np.median(g), 1e-6)))]
        obj = lambda b: float((v * (g - b * b - 2 * b * R) ** 2).sum())
        bj[j] = min(cand, key=obj)
    R2 = bj[:, None] + ReS
    P_btopic = (R2 * R2 + ImS ** 2) * lvl2[:, None]

    # ── stage 2c (operator 2026-08-08): y_j = |b_j + A_j SUM_i a_i e^{i(theta_i - p_ij)}|^2 —
    # a per-field GAIN A_j on the shared spectrum, beside the per-field level. For fixed arrows and
    # phases the objective is quartic in b_j and in A_j separately, so coordinate descent with an
    # EXACT cubic root at each step (np.roots), initialised at (b_j from stage 2b, A_j = 1).
    # Deterministic; the arrow scale gauge is fixed by the pooled global spectrum.
    Qs = ReS ** 2 + ImS ** 2
    bj2 = bj.copy(); Aj = np.ones(Tn)
    for j in range(Tn):
        awj = af.LAM_HORIZON / (mt[j] ** 2) / max(hz - wall, 1)
        v = np.concatenate([W[j], np.full(hz - wall, awj)])
        gg = np.concatenate([Yn[j, :wall], np.full(hz - wall, mt[j] ** 2)])
        R = np.concatenate([ReS[j, :wall], ReS[j, wall:hz]])
        Q = np.concatenate([Qs[j, :wall], Qs[j, wall:hz]])
        b, A = float(bj[j]), 1.0
        obj = lambda b_, A_: float((v * (gg - b_ * b_ - 2 * b_ * A_ * R - A_ * A_ * Q) ** 2).sum())
        for _ in range(8):
            # exact b-step (A fixed): P0 = g - A^2 Q
            P0 = gg - A * A * Q
            S0 = v.sum(); S1 = (v * R).sum(); S2 = (v * R * R).sum()
            SP0 = (v * P0).sum(); SP0R = (v * P0 * R).sum()
            roots = np.roots([-S0, -3 * A * S1, SP0 - 2 * A * A * S2, A * SP0R])
            cand = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real > 0] or [b]
            b = min(cand, key=lambda x: obj(x, A))
            # exact A-step (b fixed): h = g - b^2
            h = gg - b * b
            c3 = -(v * Q * Q).sum(); c2 = -3 * b * (v * R * Q).sum()
            c1 = (v * h * Q).sum() - 2 * b * b * (v * R * R).sum()
            c0 = b * (v * h * R).sum()
            roots = np.roots([c3, c2, c1, c0])
            cand = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real > 0] or [A]
            A = min(cand, key=lambda x: obj(b, x))
        bj2[j], Aj[j] = b, A
    R3 = bj2[:, None] + Aj[:, None] * ReS
    P_gain = (R3 * R3 + (Aj[:, None] * ImS) ** 2) * lvl2[:, None]
    Dm = TH[None, :, :] - Pji[:, None, :]
    R = bg + (np.cos(Dm) * ag[None, None, :]).sum(2)
    Im = (np.sin(Dm) * ag[None, None, :]).sum(2)
    P_exact = (R * R + Im * Im) * lvl2[:, None]
    return P_relax, P_exact, P_btopic, P_gain, bg, ag, bj, Aj, Pji


def auc_at(Yh, wall):
    tv = af.META["topic_valid"]; tvw = tv[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    hi = min(wall + af.HORIZON, n)
    return float(np.mean([1.0 - ((Y[:, wall + h] - Yh[:, wall + h]) ** 2).sum() /
                max(((Y[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(hi - wall)]))


RIDGE_GRID = [1e-3, 1e-2, 1e-1, 1.0]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "headline":
        Pr, Pe, Pb, Pg, *_ = fit_wall(n - 30)
        print(f"GLOBAL PHASOR headline: relaxation {auc_at(Pr, n-30):+.4f} · exact {auc_at(Pe, n-30):+.4f}"
              f" · b-per-topic {auc_at(Pb, n-30):+.4f} · gain {auc_at(Pg, n-30):+.4f}", flush=True)
        return
    print("═══ GLOBAL PHASOR, CLOSED FORM · twelve origins ═══", flush=True)
    t0 = time.time()
    # ridge chosen on the NINE EARLY origins by the EXACT model's score; 1990/93/96 never consulted
    best, sel = None, {}
    for rg in RIDGE_GRID:
        e = [auc_at(fit_wall(w, rg)[1], w) for w in WALLS[:9]]
        sel[rg] = float(np.mean(e))
        print(f"    ridge={rg:g}  early(9) exact {sel[rg]:+.4f}", flush=True)
        if best is None or sel[rg] > sel[best]: best = rg
    print(f"  CHOSEN ON EARLY ORIGINS: ridge = {best:g}", flush=True)
    rel, exa, btp, gan = [], [], [], []
    glob = None
    for w in WALLS:
        Pr, Pe, Pb, Pg, bg, ag, bj, Aj, Pji = fit_wall(w, best)
        btp.append(auc_at(Pb, w)); gan.append(auc_at(Pg, w))
        rel.append(auc_at(Pr, w)); exa.append(auc_at(Pe, w))
        if w == WALLS[-1]: glob = (bg, ag, bj, Aj, Pji)
    rel, exa, btp, gan = np.array(rel), np.array(exa), np.array(btp), np.array(gan)
    rec = np.array([auc_at(af.fit_final(Y, TH, w)[0], w) for w in WALLS])
    print(f"  [{time.time()-t0:.0f}s]", flush=True)
    print(f"  relaxation (57p/topic)        mean {rel.mean():+.4f} · 1996 {rel[-1]:+.4f}   " + " ".join(f"{v:+.3f}" for v in rel), flush=True)
    print(f"  exact global (7p/topic + 8g)  mean {exa.mean():+.4f} · 1996 {exa[-1]:+.4f}   " + " ".join(f"{v:+.3f}" for v in exa), flush=True)
    print(f"  b-per-topic (8p/t + 7 glob)   mean {btp.mean():+.4f} · 1996 {btp[-1]:+.4f}   " + " ".join(f"{v:+.3f}" for v in btp), flush=True)
    print(f"  + gain A_j (9p/t + 7 glob)    mean {gan.mean():+.4f} · 1996 {gan[-1]:+.4f}   " + " ".join(f"{v:+.3f}" for v in gan), flush=True)
    print(f"  record (9p/topic)             mean {rec.mean():+.4f} · 1996 {rec[-1]:+.4f}   " + " ".join(f"{v:+.3f}" for v in rec), flush=True)
    bg, ag, bj, Aj, Pji = glob
    print(f"  global level b = {bg:.3f} · arrow lengths: " + " ".join(f"{bd[:3]} {v:.3f}" for bd, v in zip(af.BODIES, ag)), flush=True)
    json.dump({"walls": [labels[w] for w in WALLS],
               "relaxation": [round(float(v), 4) for v in rel],
               "exact_global": [round(float(v), 4) for v in exa],
               "record": [round(float(v), 4) for v in rec],
               "b_per_topic": [round(float(v), 4) for v in btp],
               "gain": [round(float(v), 4) for v in gan],
               "gain_range": [round(float(np.percentile(Aj, q)), 3) for q in (5, 50, 95)],
               "b_topic_range": [round(float(np.percentile(bj, q)), 3) for q in (5, 50, 95)],
               "global_b": round(float(bg), 4),
               "global_a": {bd: round(float(v), 4) for bd, v in zip(af.BODIES, ag)},
               "ridge": best, "ridge_selection": {str(k): round(v, 4) for k, v in sel.items()},
               "params": {"relaxation_per_topic": NF, "exact_per_topic": nb, "exact_global": 1 + nb}},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_phasor.json"), "w"), indent=1)
    print("GPDONE", flush=True)


if __name__ == "__main__":
    main()
