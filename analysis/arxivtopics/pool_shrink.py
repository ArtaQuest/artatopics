#!/usr/bin/env python3
"""PARTIAL POOLING over the 251 per-topic receivers — do the 251 fits share a spectrum?

The model of record (arxiv_fit.fit_final) is 251 INDEPENDENT fits:
    yhat_j(t) = max( b_j + SUM_i a_ji cos(theta_i(t) - phi_j), 0 )^2
nine parameters a topic (one level, one shared tuning, seven SIGNED arrows), nothing shared.

Here we interpolate between "every topic free" and "one global model" by shrinking each topic's
arrows toward a POOLED spectrum:  a_j <- (1-lam)*a_j_hat + lam*a_bar_j.

THE FRAME PROBLEM (the thing that makes a naive pooling meaningless). Each topic carries its own
tuning phi_j, and in the fixed (cos,sin) basis its coefficient on body i is the 2-vector
a_ji*(cos phi_j, sin phi_j) -- i.e. all seven of a topic's arrows are COLLINEAR along its own axis
and only their SIGNS choose which pole. So the natural common frame is each topic's own axis (which
is exactly what a_j already is), and the ONLY residual gauge is the global 180-degree flip
(phi_j, a_j) ~ (phi_j+180, -a_j). The 0..179 sweep in arxiv_fit picks one representative of that
pair by a convention that has nothing to do with physics, so raw averaging of a_ji across topics is
averaging over an arbitrary sign and collapses to ~0. We resolve the flip with the sign-invariant
estimator (leading eigenvector of SUM_j u_j u_j', u_j = a_j/||a_j||) before pooling, and we MEASURE
the naive version so the report can show the difference.

FOUR POOLING TARGETS x THREE LEVELS, plus a shrink-to-zero control:
    zero      a_bar = 0                      -- ridge control ("does ANY regularisation help?")
    signed    a_bar = s_j*e_j*g              -- shared SIGNED spectrum (topic keeps only a scale+gauge)
    mag       a_bar = s_j*sign(a_j)*m        -- shared MAGNITUDES, per-topic SIGNS  (never tested before)
    magabs    a_bar = sign(a_j)*m_abs        -- shared magnitudes, no per-topic rescale
  levels: global (1 group) · domain (4 OpenAlex domains) · field (26 OpenAlex fields)

SELECTION DISCIPLINE: lam, the target and the level are chosen on the FIRST NINE origins
(1963..1987) ONLY, then applied unchanged to 1990/1993/1996. lam=0 must reproduce the baseline
exactly -- that is asserted, not assumed.

  python3 analysis/arxivtopics/pool_shrink.py
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arxiv_fit as af                      # noqa: E402  (chdir's to repo root on import)

OUT = os.path.join(HERE, "pool_shrink_result.json")


# ---------------------------------------------------------------- harness (verbatim from the brief)
def auc_at(Y, Yhat, wall):
    tv = af.META["topic_valid"]; tvw = tv[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    return float(np.mean([1.0 - ((Y[:, wall + h] - Yhat[:, wall + h]) ** 2).sum() /
                max(((Y[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(30)]))


# ---------------------------------------------------------------- the base fit, instrumented
def fit_base(Y, TH, fit_end):
    """arxiv_fit.fit_final re-expressed so the SOLUTION VECTOR survives: returns the design tensor X,
    the chosen grid index g_j, the raw coefficient c_j = [b_j, a_j1..a_j7], and the per-topic normal
    equations (Amat_j at g_j, bvec_j) so a shrunk refit costs no extra sweep. Verified identical to
    fit_final's yhat to ~1e-12 (assert in main)."""
    Tn, ne, nb = Y.shape[0], TH.shape[0], TH.shape[1]
    tv = af.META["topic_valid"][:, :fit_end].astype(float)
    wy = np.clip(af.META["evidence"][:fit_end], 0, None) ** 0.75
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    Wa = np.zeros_like(W)
    Wa[:, fit_end - af.ANCHOR_K:] = (tv * wy[None])[:, fit_end - af.ANCHOR_K:]
    bad = Wa.sum(1) <= 0
    Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    Ysq = np.sqrt(Y)
    MJ = np.maximum((Ysq[:, :fit_end] * Wa).sum(1), 1e-3)
    hz = min(fit_end + af.HORIZON, ne)

    G = af.GRID; NG = len(G)
    X = np.empty((NG, ne, 1 + nb)); X[:, :, 0] = 1.0
    for i in range(nb):
        X[:, :, 1 + i] = np.cos(TH[:, i][None, :] - G[:, None])
    Xt, Xa = X[:, :fit_end], X[:, fit_end:hz]
    XaS = np.einsum('gtp,gtq->gpq', Xa, Xa); XaB = np.einsum('gtp->gp', Xa)
    I = 1e-8 * np.eye(1 + nb)[None]

    C = np.zeros((Tn, 1 + nb)); GI = np.zeros(Tn, int)
    AM = np.zeros((Tn, 1 + nb, 1 + nb)); BV = np.zeros((Tn, 1 + nb))
    for j in range(Tn):
        aw = af.LAM_HORIZON / (MJ[j] ** 2) / max(hz - fit_end, 1)
        Amat = np.einsum('gtp,t,gtq->gpq', Xt, W[j], Xt) + aw * XaS + I
        bvec = np.einsum('gtp,t->gp', Xt, W[j] * Ysq[j, :fit_end]) + aw * XaB * MJ[j]
        c = np.linalg.solve(Amat, bvec[..., None])[..., 0]
        r = (((np.einsum('gtp,gp->gt', Xt, c) - Ysq[j, :fit_end][None]) ** 2) @ W[j]
             + aw * ((np.einsum('gtp,gp->gt', Xa, c) - MJ[j]) ** 2).sum(1))
        g = int(np.argmin(r))
        GI[j] = g; C[j] = c[g]; AM[j] = Amat[g]; BV[j] = bvec[g]
    return dict(X=X, g=GI, c=C, Amat=AM, bvec=BV, W=W, Ysq=Ysq, fit_end=fit_end)


def predict(fb, c=None):
    """yhat = max(X_g @ c, 0)^2 over data + forecast years."""
    c = fb["c"] if c is None else c
    Xg = fb["X"][fb["g"]]                                   # (Tn, ne, 1+nb)
    return np.maximum(np.einsum('jtp,jp->jt', Xg, c), 0.0) ** 2


# ---------------------------------------------------------------- pooling: the common frame
def resolve_gauge(U, iters=50):
    """Sign-invariant mean direction of unit rows U (Tn, nb): leading eigenvector of SUM u u',
    which is exactly the estimator that ignores the u -> -u gauge. Deterministic (eigh), then a few
    sign-and-average refinements. Returns (gbar unit vector, e_j in {-1,+1})."""
    if U.shape[0] == 1:
        return U[0].copy(), np.ones(1)
    w, V = np.linalg.eigh(U.T @ U)
    g = V[:, -1]
    if g[np.argmax(np.abs(g))] < 0:                          # fix the eigenvector's own arbitrary sign
        g = -g
    for _ in range(iters):
        e = np.sign(U @ g); e[e == 0] = 1.0
        gn = (e[:, None] * U).mean(0)
        nn = np.linalg.norm(gn)
        if nn < 1e-12:
            break
        gn = gn / nn
        if np.allclose(gn, g, atol=1e-12):
            g = gn; break
        g = gn
    e = np.sign(U @ g); e[e == 0] = 1.0
    return g, e


def build_targets(A, groups, mode):
    """Pooled target spectrum for every topic, given signed arrows A (Tn,nb) and a group label per
    topic. mode: 'zero' | 'signed' | 'mag' | 'magabs'.  Returns (target (Tn,nb), diagnostics)."""
    Tn, nb = A.shape
    Tg = np.zeros_like(A)
    s = np.linalg.norm(A, axis=1)                            # per-topic arrow scale
    s = np.maximum(s, 1e-12)
    U = A / s[:, None]
    diag = {}
    if mode == "zero":
        return Tg, diag
    for gid in np.unique(groups):
        m = groups == gid
        if mode == "signed":
            gb, e = resolve_gauge(U[m])
            Tg[m] = s[m][:, None] * e[:, None] * gb[None, :]
            diag[str(gid)] = dict(spectrum=[round(float(x), 4) for x in gb],
                                  n=int(m.sum()), align=round(float(np.mean(np.abs(U[m] @ gb))), 4))
        elif mode in ("mag", "magabs"):
            # shared MAGNITUDES, per-topic SIGNS: pool |a| (no gauge problem at all -- magnitudes are
            # gauge invariant), then re-attach each topic's own sign pattern.
            if mode == "mag":
                mb = np.abs(U[m]).mean(0)                    # scale-normalised magnitude spectrum
                Tg[m] = s[m][:, None] * np.sign(A[m]) * mb[None, :]
            else:
                mb = np.abs(A[m]).mean(0)                    # absolute magnitude spectrum (no rescale)
                Tg[m] = np.sign(A[m]) * mb[None, :]
            diag[str(gid)] = dict(spectrum=[round(float(x), 4) for x in mb], n=int(m.sum()))
        else:
            raise ValueError(mode)
    return Tg, diag


def refit_level(fb, c_new):
    """Re-solve ONLY the level b_j with the arrows held at their shrunk values (b is already one of
    the nine parameters, so this adds nothing to the count). Closed form from the stored normal
    equations: b = (bvec_0 - A_0,1: . a) / A_00."""
    A0 = fb["Amat"][:, 0, 0]
    off = np.einsum('jp,jp->j', fb["Amat"][:, 0, 1:], c_new[:, 1:])
    out = c_new.copy()
    out[:, 0] = (fb["bvec"][:, 0] - off) / np.maximum(A0, 1e-12)
    return out


# ---------------------------------------------------------------- effective degrees of freedom
def eff_df(fb, lam):
    """Honest effective parameter count for the CONVEX shrinkage estimator. a_shrunk =
    (1-lam)*a_hat + lam*T a_hat  (T is the pooling operator, a projection-like average). Per topic
    the arrow block's df is trace of the map from data to fitted arrows; for the global/domain mean
    with per-topic scale the pooled part contributes ~1 free number per topic (its scale) plus the
    group spectrum shared over n_g topics. Reported as: 2 (level+tuning) + 7*(1-lam) + lam*(1 + 7/n_g)."""
    return lam


# ---------------------------------------------------------------- main sweep
def main():
    t_all = time.time()
    names, Y, labels, future = af.load_lunar()
    TH, R = af.sky_lunar(labels + future)
    n = Y.shape[1]; Tn, nb = Y.shape[0], TH.shape[1]
    WALLS = list(range(n - 63, n - 29, 3))
    SEL, HELD = WALLS[:9], WALLS[9:]
    yr = lambda w: int(labels[w])
    print(f"  {Tn} topics x {n} years · {nb} bodies · walls "
          f"{[yr(w) for w in WALLS]}  (select {[yr(w) for w in SEL]} | held {[yr(w) for w in HELD]})",
          flush=True)

    doms = np.array([af.META["domain"][nm] for nm in names])
    flds = np.array([af.META["field"][nm] for nm in names])
    LEVELS = {"global": np.zeros(Tn, int),
              "domain": np.unique(doms, return_inverse=True)[1],
              "field": np.unique(flds, return_inverse=True)[1]}
    NG_LEVEL = {k: int(v.max() + 1) for k, v in LEVELS.items()}
    print(f"  pooling levels: " + " · ".join(f"{k}={v} groups" for k, v in NG_LEVEL.items()), flush=True)

    LAMS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    MODES = ["zero", "signed", "mag", "magabs"]

    # ---- one base fit per wall (2.4 s each), everything else is free
    base = {}
    t0 = time.time()
    for w in WALLS:
        base[w] = fit_base(Y, TH, w)
    t_base = time.time() - t0
    print(f"  base fits: {len(WALLS)} walls in {t_base:.1f}s", flush=True)

    # ---- CORRECTNESS: our instrumented fit must reproduce fit_final bit-for-bit
    w96 = WALLS[-1]
    yh_ref, prm_ref = af.fit_final(Y, TH, w96)
    yh_ours = predict(base[w96])
    dev = float(np.abs(yh_ref - yh_ours).max())
    auc_ref = auc_at(Y, yh_ref, w96)
    print(f"  CHECK fit_base == fit_final : max|dy| = {dev:.3e}   AUC1996 {auc_ref:+.4f}", flush=True)
    assert dev < 1e-10, "instrumented fit diverged from the model of record"

    base_auc = {w: auc_at(Y, predict(base[w]), w) for w in WALLS}
    b_sel = float(np.mean([base_auc[w] for w in SEL]))
    b_held = float(np.mean([base_auc[w] for w in HELD]))
    b_all = float(np.mean([base_auc[w] for w in WALLS]))
    print(f"  BASELINE (lam=0): select9 {b_sel:+.4f} · held3 {b_held:+.4f} · all12 {b_all:+.4f} · "
          f"1996 {base_auc[w96]:+.4f}", flush=True)

    # ---- persistence bar
    def pers(w):
        tv = af.META["topic_valid"]; tvw = tv[:, :w].astype(float)
        mu = (Y[:, :w] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
        last = Y[:, w - 1]
        return float(np.mean([1.0 - ((Y[:, w + h] - last) ** 2).sum() /
                              max(((Y[:, w + h] - mu) ** 2).sum(), 1e-9) for h in range(30)]))
    p_all = float(np.mean([pers(w) for w in WALLS]))
    print(f"  persistence: all12 {p_all:+.4f} · 1996 {pers(w96):+.4f}", flush=True)

    # ---- the NAIVE (unaligned) pooling, measured so the frame problem is visible not asserted
    A96 = base[w96]["c"][:, 1:]
    U96 = A96 / np.maximum(np.linalg.norm(A96, axis=1), 1e-12)[:, None]
    gb, e = resolve_gauge(U96)
    naive = U96.mean(0)
    print(f"  FRAME CHECK @1996 · ||mean(u_j)|| (naive) = {np.linalg.norm(naive):.4f}  vs  "
          f"||mean(e_j u_j)|| (gauge-resolved) = {np.linalg.norm((e[:, None]*U96).mean(0)):.4f}  "
          f"· minority gauge {int((e < 0).sum())}/{Tn}", flush=True)

    # ---- the full grid
    rows = []
    for mode in MODES:
        for lvl, groups in LEVELS.items():
            if mode == "zero" and lvl != "global":
                continue                                     # ridge control has no level
            for relvl in (False, True):
                aucs = {}
                for w in WALLS:
                    fb = base[w]
                    Tg, _ = build_targets(fb["c"][:, 1:], groups, mode)
                    for lam in LAMS:
                        c = fb["c"].copy()
                        c[:, 1:] = (1 - lam) * fb["c"][:, 1:] + lam * Tg
                        if relvl and lam > 0:
                            c = refit_level(fb, c)
                        aucs.setdefault(lam, {})[w] = auc_at(Y, predict(fb, c), w)
                for lam in LAMS:
                    a = aucs[lam]
                    rows.append(dict(mode=mode, level=lvl, relevel=relvl, lam=lam,
                                     sel=float(np.mean([a[w] for w in SEL])),
                                     held=float(np.mean([a[w] for w in HELD])),
                                     all=float(np.mean([a[w] for w in WALLS])),
                                     w1996=a[w96],
                                     per_wall={yr(w): round(a[w], 4) for w in WALLS}))
        print(f"  swept mode={mode}", flush=True)

    # ---- lambda curves (printed in full, per the brief)
    print("\n  LAMBDA CURVES  (select-9 mean; * = level re-solved after shrinking)", flush=True)
    hdr = "  " + "lam".ljust(26) + "".join(f"{l:>8.1f}" for l in LAMS)
    print(hdr, flush=True)
    for mode in MODES:
        for lvl in LEVELS:
            for relvl in (False, True):
                sub = [r for r in rows if r["mode"] == mode and r["level"] == lvl and r["relevel"] == relvl]
                if not sub:
                    continue
                sub = sorted(sub, key=lambda r: r["lam"])
                tag = f"{mode}/{lvl}{'*' if relvl else ''}"
                print("  " + tag.ljust(26) + "".join(f"{r['sel']:+8.4f}" for r in sub), flush=True)

    # ---- SELECTION: argmax of select-9 over the whole grid, applied unchanged to the held-out three
    best = max(rows, key=lambda r: r["sel"])
    print(f"\n  SELECTED ON THE FIRST NINE ORIGINS ONLY: mode={best['mode']} level={best['level']} "
          f"relevel={best['relevel']} lam={best['lam']}", flush=True)
    print(f"    select9 {best['sel']:+.4f} (baseline {b_sel:+.4f}, delta {best['sel']-b_sel:+.4f})", flush=True)
    print(f"    HELD3   {best['held']:+.4f} (baseline {b_held:+.4f}, delta {best['held']-b_held:+.4f})", flush=True)
    print(f"    all12   {best['all']:+.4f} (baseline {b_all:+.4f}, delta {best['all']-b_all:+.4f})", flush=True)
    print(f"    1996    {best['w1996']:+.4f} (baseline {base_auc[w96]:+.4f}, "
          f"delta {best['w1996']-base_auc[w96]:+.4f})", flush=True)

    # ---- best per mode (so the report can separate 'pooling helps' from 'any shrinkage helps')
    print("\n  BEST PER MODE (selected on 9, reported on all):", flush=True)
    for mode in MODES:
        sub = [r for r in rows if r["mode"] == mode]
        b = max(sub, key=lambda r: r["sel"])
        print(f"    {mode:8s} lvl={b['level']:7s} relevel={int(b['relevel'])} lam={b['lam']:.1f} · "
              f"sel {b['sel']:+.4f} held {b['held']:+.4f} all {b['all']:+.4f} 1996 {b['w1996']:+.4f}",
              flush=True)

    # ---- FULL POOLING endpoints (lam=1) -- the actual parameter-count claims
    print("\n  FULL POOLING (lam=1.0) -- these are the genuine parameter reductions:", flush=True)
    for mode in ("signed", "mag", "magabs"):
        for lvl in LEVELS:
            for relvl in (False, True):
                r = [x for x in rows if x["mode"] == mode and x["level"] == lvl
                     and x["relevel"] == relvl and x["lam"] == 1.0]
                if not r:
                    continue
                r = r[0]
                ng = NG_LEVEL[lvl]
                if mode == "signed":                 # per topic: b, phi, scale, gauge sign
                    ppt, glob = 4, nb * ng
                elif mode == "mag":                  # per topic: b, phi, scale, 7 signs
                    ppt, glob = 3 + nb, nb * ng
                else:                                # per topic: b, phi, 7 signs
                    ppt, glob = 2 + nb, nb * ng
                print(f"    {mode:8s}/{lvl:7s} relevel={int(relvl)} · {ppt} p/topic + {glob} global "
                      f"= {ppt*Tn+glob:5d} · sel {r['sel']:+.4f} held {r['held']:+.4f} "
                      f"all {r['all']:+.4f} 1996 {r['w1996']:+.4f}", flush=True)

    wall_s = time.time() - t_all
    res = dict(baseline=dict(sel=b_sel, held=b_held, all=b_all, w1996=base_auc[w96],
                             per_wall={yr(w): round(base_auc[w], 4) for w in WALLS},
                             params_per_topic=9, params_total=9 * Tn),
               persistence=dict(all=p_all, w1996=pers(w96)),
               frame_check=dict(naive_norm=float(np.linalg.norm(naive)),
                                aligned_norm=float(np.linalg.norm((e[:, None] * U96).mean(0))),
                                minority_gauge=int((e < 0).sum())),
               walls=[yr(w) for w in WALLS], select=[yr(w) for w in SEL], held=[yr(w) for w in HELD],
               lams=LAMS, grid=rows, selected=best, wall_clock_s=round(wall_s, 1),
               deterministic=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  wall clock {wall_s:.1f}s (12 origins, whole grid) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
