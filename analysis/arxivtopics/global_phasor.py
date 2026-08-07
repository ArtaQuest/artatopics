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
    Dm = TH[None, :, :] - Pji[:, None, :]
    R = bg + (np.cos(Dm) * ag[None, None, :]).sum(2)
    Im = (np.sin(Dm) * ag[None, None, :]).sum(2)
    P_exact = (R * R + Im * Im) * lvl2[:, None]
    return P_relax, P_exact, bg, ag, Pji


def auc_at(Yh, wall):
    tv = af.META["topic_valid"]; tvw = tv[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    hi = min(wall + af.HORIZON, n)
    return float(np.mean([1.0 - ((Y[:, wall + h] - Yh[:, wall + h]) ** 2).sum() /
                max(((Y[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(hi - wall)]))


RIDGE_GRID = [1e-3, 1e-2, 1e-1, 1.0]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "headline":
        Pr, Pe, *_ = fit_wall(n - 30)
        print(f"GLOBAL PHASOR headline: relaxation {auc_at(Pr, n-30):+.4f} · exact {auc_at(Pe, n-30):+.4f}", flush=True)
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
    rel, exa = [], []
    glob = None
    for w in WALLS:
        Pr, Pe, bg, ag, Pji = fit_wall(w, best)
        rel.append(auc_at(Pr, w)); exa.append(auc_at(Pe, w))
        if w == WALLS[-1]: glob = (bg, ag, Pji)
    rel, exa = np.array(rel), np.array(exa)
    rec = np.array([auc_at(af.fit_final(Y, TH, w)[0], w) for w in WALLS])
    print(f"  [{time.time()-t0:.0f}s]", flush=True)
    print(f"  relaxation (57p/topic)        mean {rel.mean():+.4f} · 1996 {rel[-1]:+.4f}   " + " ".join(f"{v:+.3f}" for v in rel), flush=True)
    print(f"  exact global (7p/topic + 8g)  mean {exa.mean():+.4f} · 1996 {exa[-1]:+.4f}   " + " ".join(f"{v:+.3f}" for v in exa), flush=True)
    print(f"  record (9p/topic)             mean {rec.mean():+.4f} · 1996 {rec[-1]:+.4f}   " + " ".join(f"{v:+.3f}" for v in rec), flush=True)
    bg, ag, Pji = glob
    print(f"  global level b = {bg:.3f} · arrow lengths: " + " ".join(f"{bd[:3]} {v:.3f}" for bd, v in zip(af.BODIES, ag)), flush=True)
    json.dump({"walls": [labels[w] for w in WALLS],
               "relaxation": [round(float(v), 4) for v in rel],
               "exact_global": [round(float(v), 4) for v in exa],
               "record": [round(float(v), 4) for v in rec],
               "global_b": round(float(bg), 4),
               "global_a": {bd: round(float(v), 4) for bd, v in zip(af.BODIES, ag)},
               "ridge": best, "ridge_selection": {str(k): round(v, 4) for k, v in sel.items()},
               "params": {"relaxation_per_topic": NF, "exact_per_topic": nb, "exact_global": 1 + nb}},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_phasor.json"), "w"), indent=1)
    print("GPDONE", flush=True)


if __name__ == "__main__":
    main()
