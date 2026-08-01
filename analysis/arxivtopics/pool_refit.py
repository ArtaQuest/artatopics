#!/usr/bin/env python3
"""PARTIAL POOLING DONE PROPERLY -- a penalised REFIT, not a post-hoc average.

pool_shrink.py shrinks the ALREADY-FITTED arrows toward a pooled spectrum. That is honest but weak:
the tuning phi_j was chosen for the FREE arrows and never gets to move, so at strong shrinkage the
model is a good spectrum read at the wrong angle. Here the prior is INSIDE the fit:

    minimise   SUM_t w_jt (b + SUM_i a_i cos(theta_i(t)-phi) - sqrt(y_jt))^2
             + the thirty-year horizon anchor rows (unchanged)
             + tau * || a - t_j ||^2                          <-- the pooled prior

still a closed-form weighted least-squares at every one of the 180 candidate tunings: the prior is
tau*I added to the normal matrix and tau*t_j added to the right-hand side. phi IS re-selected under
the prior, the fit stays exhaustive and deterministic, and tau=0 must reproduce the model of record
bit-for-bit (asserted below).

FAST BECAUSE THE PRIOR ONLY MOVES ONE EIGENVALUE SET. Partition the normal matrix by level/arrows;
the arrow block's Schur complement S = A11 - a01 a01'/a00 does NOT depend on tau or on the target, so
one batched eigendecomposition per wall (251 topics x 180 tunings) makes EVERY tau a pair of 7x7
matvecs. The whole tau x prior x level x round grid then costs about one extra sweep.

FIVE PRIORS (t_j), at every pooling level that means anything:
    zero      t_j = 0                       -- ridge control: does ANY regularisation help?
    signed    t_j = s_j e_j g               -- shared SIGNED spectrum (180-degree gauge resolved)
    mag       t_j = s_j sign(a_j) m         -- shared MAGNITUDES, per-topic SIGNS
    magabs    t_j = sign(a_j) m_abs         -- shared magnitudes, no per-topic rescale
    atom      t_j = s_j g_k(j)              -- a DICTIONARY of K learned spectra (dict_spectrum.py)
  levels: global · domain (4 OpenAlex domains) · field (26 OpenAlex fields) · cluster (K k-lines)

EFFECTIVE PARAMETERS. Partial pooling does not delete a parameter, it costs a fraction of one. The
honest count is the ridge trace df = tr((A + tau*D)^-1 A) over the eight fitted numbers, plus 1 for
the tuning. Reported beside the nominal count at every tau.

SELECTION: tau, prior, level and the number of target-update rounds are chosen on the FIRST NINE
origins (1963..1987) ONLY, then applied unchanged to 1990/1993/1996.

  python3 analysis/arxivtopics/pool_refit.py
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arxiv_fit as af                                            # noqa: E402
from pool_shrink import auc_at, build_targets                     # noqa: E402
from dict_spectrum import klines, wall_weights                    # noqa: E402

OUT = os.path.join(HERE, "pool_refit_result.json")
KDICT = 12                    # dictionary size -- carried over from dict_spectrum.py's select-9 pick
TAUS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
MODES = ["zero", "signed", "mag", "magabs", "atom"]
ROUNDS = 2                    # target-update rounds; 0/1/2 all recorded, the winner is selected on 9


def cache_wall(Y, TH, fit_end):
    """Everything independent of tau and of the target: per-topic normal equations at all 180 tunings,
    already partitioned and eigendecomposed on the arrow block."""
    Tn, ne, nb = Y.shape[0], TH.shape[0], TH.shape[1]
    W, Ysq, MJ, hz, aw = wall_weights(Y, TH, fit_end)
    G = af.GRID; NG = len(G)
    X = np.empty((NG, ne, 1 + nb)); X[:, :, 0] = 1.0
    for i in range(nb):
        X[:, :, 1 + i] = np.cos(TH[:, i][None, :] - G[:, None])
    Xt, Xa = X[:, :fit_end], X[:, fit_end:hz]
    XaS = np.einsum('gtp,gtq->gpq', Xa, Xa); XaB = np.einsum('gtp->gp', Xa)
    I = 1e-8 * np.eye(1 + nb)[None]
    AM = np.empty((Tn, NG, 1 + nb, 1 + nb)); BV = np.empty((Tn, NG, 1 + nb))
    for j in range(Tn):
        AM[j] = np.einsum('gtp,t,gtq->gpq', Xt, W[j], Xt) + aw[j] * XaS + I
        BV[j] = np.einsum('gtp,t->gp', Xt, W[j] * Ysq[j, :fit_end]) + aw[j] * XaB * MJ[j]
    a00 = AM[:, :, 0, 0]                       # (Tn,NG)
    a01 = AM[:, :, 0, 1:]                      # (Tn,NG,nb)
    S = AM[:, :, 1:, 1:] - a01[..., :, None] * a01[..., None, :] / a00[..., None, None]
    L, Q = np.linalg.eigh(S)                   # S = Q diag(L) Q'   -- tau shifts L only
    return dict(X=X, AM=AM, BV=BV, a00=a00, a01=a01, L=L, Q=Q, fit_end=fit_end, nb=nb, Tn=Tn)


def solve_pen(cw, target, tau):
    """Penalised sweep. J(c) = c'Mc - 2c'v + const, M = A + tau*D (D = diag(0,1..1)), v = b + tau*[0;t];
    at the optimum J = -c.v, so the tuning is argmax(c.v) and no residual pass is needed."""
    v0 = cw["BV"][:, :, 0]                                        # (Tn,NG)
    v1 = cw["BV"][:, :, 1:]
    if tau > 0:
        v1 = v1 + tau * target[:, None, :]
    u = v1 - cw["a01"] * (v0 / cw["a00"])[..., None]
    qu = np.einsum('jgab,jga->jgb', cw["Q"], u)                   # Q' u
    x = np.einsum('jgab,jgb->jga', cw["Q"], qu / (cw["L"] + tau))
    b = (v0 - np.einsum('jga,jga->jg', cw["a01"], x)) / cw["a00"]
    score = b * v0 + np.einsum('jga,jga->jg', x, v1)
    gi = np.argmax(score, axis=1)
    ar = np.arange(cw["Tn"])
    c = np.concatenate([b[ar, gi][:, None], x[ar, gi]], 1)
    yh = np.maximum(np.einsum('jtp,jp->jt', cw["X"][gi], c), 0.0) ** 2
    return yh, c, gi


def edf(cw, gi, tau):
    """ridge trace df = tr((A+tau D)^-1 A) over the eight fitted numbers, at the selected tuning."""
    ar = np.arange(cw["Tn"])
    A = cw["AM"][ar, gi]
    D = np.zeros((cw["nb"] + 1, cw["nb"] + 1)); D[1:, 1:] = np.eye(cw["nb"])
    M = A + tau * D
    return float(np.mean([np.trace(np.linalg.solve(M[j], A[j])) for j in range(A.shape[0])]))


def main():
    t_all = time.time()
    names, Y, labels, future = af.load_lunar()
    TH, R = af.sky_lunar(labels + future)
    n = Y.shape[1]; Tn, nb = Y.shape[0], TH.shape[1]
    WALLS = list(range(n - 63, n - 29, 3))
    SEL, HELD = WALLS[:9], WALLS[9:]
    yr = lambda w: int(labels[w]); w96 = WALLS[-1]

    doms = np.unique([af.META["domain"][nm] for nm in names], return_inverse=True)[1]
    flds = np.unique([af.META["field"][nm] for nm in names], return_inverse=True)[1]
    LEVELS = {"global": np.zeros(Tn, int), "domain": doms, "field": flds}
    COMBOS = ([("zero", "global")] +
              [(m, l) for m in ("signed", "mag", "magabs") for l in LEVELS] +
              [("atom", "cluster")])

    auc = {}                                   # (mode, level, rounds, tau) -> {wall: auc}
    dfs = {}
    yh_ref96 = None
    print(f"  {Tn} topics · {len(WALLS)} walls · {len(TAUS)} taus · {len(COMBOS)} priors · "
          f"{ROUNDS+1} round settings", flush=True)
    for w in WALLS:
        t0 = time.time()
        cw = cache_wall(Y, TH, w)
        yh0, c0, gi0 = solve_pen(cw, np.zeros((Tn, nb)), 0.0)
        a0 = auc_at(Y, yh0, w)
        if w == w96:
            yh_ref96 = yh0
        A0 = c0[:, 1:]
        U0 = A0 / np.maximum(np.linalg.norm(A0, axis=1), 1e-12)[:, None]
        clus = klines(U0, KDICT)[1]
        grp = dict(LEVELS); grp["cluster"] = clus
        for mode, lvl in COMBOS:
            tmode = "signed" if mode == "atom" else mode
            for tau in TAUS:
                A = A0
                for rd in range(ROUNDS + 1):
                    if tau == 0.0:
                        yh = yh0
                    else:
                        tg = np.zeros_like(A) if mode == "zero" else build_targets(A, grp[lvl], tmode)[0]
                        yh, c, gi = solve_pen(cw, tg, tau)
                        A = c[:, 1:]
                    auc.setdefault((mode, lvl, rd, tau), {})[w] = auc_at(Y, yh, w) if tau > 0 else a0
                    if w == w96:
                        dfs[(mode, lvl, rd, tau)] = edf(cw, gi0 if tau == 0.0 else gi, tau)
                    if mode == "zero" or tau == 0.0:
                        for r2 in range(rd + 1, ROUNDS + 1):        # nothing to update
                            auc.setdefault((mode, lvl, r2, tau), {})[w] = auc[(mode, lvl, rd, tau)][w]
                            if w == w96:
                                dfs[(mode, lvl, r2, tau)] = dfs[(mode, lvl, rd, tau)]
                        break
        print(f"  wall {yr(w)}  base AUC {a0:+.4f}  [{time.time()-t0:.0f}s]", flush=True)
        del cw

    # ---- correctness
    yh_f, _ = af.fit_final(Y, TH, w96)
    dev = float(np.abs(yh_f - yh_ref96).max())
    print(f"\n  CHECK tau=0 == fit_final : max|dy| = {dev:.3e}", flush=True)
    assert dev < 1e-10

    base = auc[("zero", "global", 0, 0.0)]
    b_sel = float(np.mean([base[w] for w in SEL]))
    b_held = float(np.mean([base[w] for w in HELD]))
    b_all = float(np.mean([base[w] for w in WALLS]))
    print(f"  BASELINE (tau=0): select9 {b_sel:+.4f} · held3 {b_held:+.4f} · all12 {b_all:+.4f} · "
          f"1996 {base[w96]:+.4f}", flush=True)
    assert abs(b_all - 0.8751) < 5e-4 and abs(base[w96] - 0.7990) < 5e-4

    rows = []
    for (mode, lvl, rd, tau), a in auc.items():
        rows.append(dict(mode=mode, level=lvl, rounds=rd, tau=tau,
                         sel=float(np.mean([a[w] for w in SEL])),
                         held=float(np.mean([a[w] for w in HELD])),
                         all=float(np.mean([a[w] for w in WALLS])), w1996=a[w96],
                         edf=round(dfs[(mode, lvl, rd, tau)] + 1.0, 3),
                         per_wall={yr(w): round(a[w], 4) for w in WALLS}))

    print("\n  TAU CURVES (select-9 mean; best round per cell):", flush=True)
    print("  " + "prior".ljust(18) + "".join(f"{t:>8g}" for t in TAUS), flush=True)
    for mode, lvl in COMBOS:
        line = [max(r["sel"] for r in rows if r["mode"] == mode and r["level"] == lvl
                    and r["tau"] == t) for t in TAUS]
        print("  " + f"{mode}/{lvl}".ljust(18) + "".join(f"{v:+8.4f}" for v in line), flush=True)

    print("\n  EFFECTIVE PARAMS / TOPIC (ridge trace df + tuning; nominal 9):", flush=True)
    print("  " + "prior".ljust(18) + "".join(f"{t:>8g}" for t in TAUS), flush=True)
    for mode, lvl in COMBOS:
        line = [np.mean([r["edf"] for r in rows if r["mode"] == mode and r["level"] == lvl
                         and r["tau"] == t]) for t in TAUS]
        print("  " + f"{mode}/{lvl}".ljust(18) + "".join(f"{v:8.2f}" for v in line), flush=True)

    best = max(rows, key=lambda r: r["sel"])
    print(f"\n  SELECTED ON THE FIRST NINE ORIGINS ONLY: prior={best['mode']}/{best['level']} "
          f"rounds={best['rounds']} tau={best['tau']:g} · effective params/topic {best['edf']:.2f}",
          flush=True)
    for tag, k, bl in (("select9", "sel", b_sel), ("HELD3", "held", b_held),
                       ("all12", "all", b_all), ("1996", "w1996", base[w96])):
        print(f"    {tag:8s} {best[k]:+.4f}  (baseline {bl:+.4f}, delta {best[k]-bl:+.4f})", flush=True)
    print(f"    per wall {best['per_wall']}", flush=True)

    print("\n  BEST PER PRIOR (selected on 9, reported on all):", flush=True)
    for mode, lvl in COMBOS:
        b = max([r for r in rows if r["mode"] == mode and r["level"] == lvl], key=lambda r: r["sel"])
        print(f"    {mode:7s}/{lvl:8s} rd={b['rounds']} tau={b['tau']:<6g} · sel {b['sel']:+.4f} "
              f"held {b['held']:+.4f} all {b['all']:+.4f} 1996 {b['w1996']:+.4f} · edf {b['edf']:.2f}",
              flush=True)

    print("\n  STRONG PRIOR (tau=100) -- the near-fully-pooled end:", flush=True)
    for mode, lvl in COMBOS:
        b = max([r for r in rows if r["mode"] == mode and r["level"] == lvl and r["tau"] == 100.0],
                key=lambda r: r["sel"])
        print(f"    {mode:7s}/{lvl:8s} rd={b['rounds']} · sel {b['sel']:+.4f} held {b['held']:+.4f} "
              f"all {b['all']:+.4f} 1996 {b['w1996']:+.4f} · edf {b['edf']:.2f}", flush=True)

    wall_s = time.time() - t_all
    res = dict(model="penalised refit toward a pooled spectrum (phi re-selected under the prior)",
               baseline=dict(sel=b_sel, held=b_held, all=b_all, w1996=base[w96],
                             params_per_topic=9, params_total=9 * Tn),
               taus=TAUS, walls=[yr(w) for w in WALLS], select=[yr(w) for w in SEL],
               held=[yr(w) for w in HELD], kdict=KDICT, grid=rows, selected=best,
               deterministic=True, wall_clock_s=round(wall_s, 1))
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  wall clock {wall_s:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
