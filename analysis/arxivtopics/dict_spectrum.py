#!/usr/bin/env python3
"""THE SHARED-SPECTRUM DICTIONARY -- a GLOBAL model for all 251 topics.

    yhat_j(t) = max( b_j + s_j * <g_{k(j)}, cos(theta(t) - phi_j)> , 0 )^2

A topic no longer owns seven signed arrows. It owns a LEVEL b_j, a TUNING phi_j, a SIGNED GAIN s_j
and a POINTER k_j into a dictionary of K global spectra g_1..g_K (unit vectors in R^7, shared by
every topic). Four parameters a topic instead of nine; the seven-number shape is global.

WHY A DICTIONARY AND NOT ONE SPECTRUM. eight_struct.py already showed that ONE shared shape
(a_ji = s_j*g_i) loses (+0.7719 at the 1996 wall vs +0.7990). The campaign's own diagnosis is that
that variant collapsed the seven bodies into a single regressor column, so a topic could only flip
the WHOLE spectrum, not individual bodies -- and 33.2% of the free model's arrows sit in the
minority direction, so the sign pattern is doing real work. A dictionary keeps that: K atoms ARE K
sign-and-magnitude patterns, and a topic picks the one it needs. K=1 reproduces the known loser, so
the K curve measures exactly how much of the free model's 1,757 signed arrows is irreducible.

THE FIT IS EXHAUSTIVE AND DETERMINISTIC -- no optimiser, no seed. Given the dictionary, (phi, k) is
a discrete pair over 180 tunings x K atoms and (b, s) is a 2x2 weighted least-squares solve at each,
with the SAME N^0.75 evidence weights, the SAME per-topic non-zero crop and the SAME thirty-year
horizon anchor folded in as extra rows as the model of record. So we score every one of the 180*K
candidates in closed form and keep the best. The dictionary itself is learned by alternating least
squares seeded from a deterministic k-lines clustering (sign-invariant, since g and -g are the same
atom -- s_j is free-signed) of the free fit's gauge-aligned arrows. Every step is an eigenvector or
a linear solve; refit it and you get the same bits.

SELECTION: K, the number of ALS rounds and the atom-weighting are chosen on the FIRST NINE origins
(1963..1987) only, then applied unchanged to 1990/1993/1996.

  python3 analysis/arxivtopics/dict_spectrum.py
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arxiv_fit as af                                   # noqa: E402
from pool_shrink import fit_base, predict, auc_at, resolve_gauge   # noqa: E402

OUT = os.path.join(HERE, "dict_spectrum_result.json")


# ---------------------------------------------------------------- per-wall weights (as fit_final)
def wall_weights(Y, TH, fit_end):
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
    hz = min(fit_end + af.HORIZON, TH.shape[0])
    aw = af.LAM_HORIZON / (MJ ** 2) / max(hz - fit_end, 1)      # per-topic scale-free anchor weight
    return W, Ysq, MJ, hz, aw


# ---------------------------------------------------------------- deterministic k-lines init
def klines_init(U, K):
    """Sign-invariant furthest-point seeding on the projective sphere (g and -g are one atom).
    Deterministic: first atom = leading eigenvector of U'U, then repeatedly take the row least
    explained by the atoms so far."""
    w, V = np.linalg.eigh(U.T @ U)
    g0 = V[:, -1]
    if g0[np.argmax(np.abs(g0))] < 0:
        g0 = -g0
    G = [g0]
    while len(G) < K:
        cov = np.max(np.abs(U @ np.array(G).T), axis=1)          # best |cos| to any atom
        j = int(np.argmin(cov))
        g = U[j] / max(np.linalg.norm(U[j]), 1e-12)
        if g[np.argmax(np.abs(g))] < 0:
            g = -g
        G.append(g)
    return np.array(G)


def klines(U, K, iters=25):
    """k-lines: assign by max |u.g|, update each atom to the leading eigenvector of its members."""
    G = klines_init(U, K)
    lab = np.full(U.shape[0], -1)
    for _ in range(iters):
        new = np.argmax(np.abs(U @ G.T), axis=1)
        if (new == lab).all():
            break
        lab = new
        for k in range(K):
            m = lab == k
            if not m.any():
                continue
            g, _ = resolve_gauge(U[m])
            nn = np.linalg.norm(g)
            if nn > 1e-12:
                G[k] = g / nn
    return G, lab


# ---------------------------------------------------------------- the exhaustive E-step
def e_step(Y, TH, G, W, Ysq, MJ, hz, aw, fit_end):
    """Score all 180 x K candidates for every topic in closed form, keep the best.
    Returns (yhat, k, gi, b, s, resid)."""
    Tn, ne, nb = Y.shape[0], TH.shape[0], TH.shape[1]
    K = G.shape[0]; NGD = len(af.GRID)
    # C[g,t,i] = cos(theta_i(t) - phi_g)   ->   Z[c,t] with c = g*K + k
    C = np.cos(TH[None, :, :] - af.GRID[:, None, None])          # (NG, ne, nb)
    Z = np.einsum('gti,ki->gkt', C, G).reshape(NGD * K, ne)      # (NC, ne)
    Zt, Za = Z[:, :fit_end], Z[:, fit_end:hz]
    na = hz - fit_end

    # weighted moments (train)
    S01 = W @ Zt.T                                               # (Tn, NC)  sum w z
    S11 = W @ (Zt ** 2).T                                        #           sum w z^2
    T0 = (W * Ysq[:, :fit_end]).sum(1)                           # (Tn,)     sum w y
    T1 = (W * Ysq[:, :fit_end]) @ Zt.T                           # (Tn, NC)  sum w z y
    Syy = (W * Ysq[:, :fit_end] ** 2).sum(1)                     # (Tn,)
    # anchor moments (topic-independent shape, per-topic weight aw and target MJ)
    Za1 = Za.sum(1)                                              # (NC,)
    Za2 = (Za ** 2).sum(1)                                       # (NC,)

    A00 = 1.0 + aw[:, None] * na                                 # sum w = 1 (W row-normalised)
    A01 = S01 + aw[:, None] * Za1[None, :]
    A11 = S11 + aw[:, None] * Za2[None, :]
    B0 = T0[:, None] + aw[:, None] * MJ[:, None] * na
    B1 = T1 + aw[:, None] * MJ[:, None] * Za1[None, :]

    det = A00 * A11 - A01 ** 2
    det = np.where(np.abs(det) < 1e-14, 1e-14, det)
    b = (A11 * B0 - A01 * B1) / det
    s = (A00 * B1 - A01 * B0) / det
    # residual = Syy + aw*na*MJ^2 - (b*B0 + s*B1)      (normal-equation identity)
    r = (Syy[:, None] + aw[:, None] * na * (MJ ** 2)[:, None]) - (b * B0 + s * B1)
    c = np.argmin(r, axis=1)
    ar = np.arange(Tn)
    bb, ss, rr = b[ar, c], s[ar, c], r[ar, c]
    gi, kk = c // K, c % K
    Zsel = Z[c]                                                  # (Tn, ne)
    yh = np.maximum(bb[:, None] + ss[:, None] * Zsel, 0.0) ** 2
    return yh, kk, gi, bb, ss, rr


def m_step(Y, TH, kk, gi, bb, ss, W, Ysq, MJ, hz, aw, fit_end, K, tw):
    """Refit each atom by weighted least squares over the topics that point at it, holding their
    (b, phi, s) fixed. 7-parameter linear solve per atom, anchor rows included."""
    nb = TH.shape[1]
    G = np.zeros((K, nb))
    Call = np.cos(TH[None, :, :] - af.GRID[gi][:, None, None])   # (Tn, ne, nb) at each topic's tuning
    for k in range(K):
        m = np.where(kk == k)[0]
        if len(m) == 0:
            G[k] = 0.0
            continue
        A = np.zeros((nb, nb)); rhs = np.zeros(nb)
        for j in m:
            wj = tw[j]
            Cj = Call[j]
            Xt = ss[j] * Cj[:fit_end]
            rt = Ysq[j, :fit_end] - bb[j]
            A += wj * (Xt * W[j][:, None]).T @ Xt
            rhs += wj * Xt.T @ (W[j] * rt)
            Xa = ss[j] * Cj[fit_end:hz]
            ra = MJ[j] - bb[j]
            A += wj * aw[j] * (Xa.T @ Xa)
            rhs += wj * aw[j] * Xa.sum(0) * ra
        g = np.linalg.solve(A + 1e-10 * np.eye(nb), rhs)
        nn = np.linalg.norm(g)
        G[k] = g / nn if nn > 1e-12 else 0.0
    # an emptied atom is re-seeded from the worst-fit topic's own direction (deterministic)
    return G


def fit_dict(Y, TH, fit_end, K, rounds, U_free, tw_mode):
    W, Ysq, MJ, hz, aw = wall_weights(Y, TH, fit_end)
    G, _ = klines(U_free, K)
    tw = np.ones(Y.shape[0]) if tw_mode == "flat" else np.maximum(Y[:, :fit_end].mean(1), 1e-6)
    tw = tw / tw.mean()
    out = e_step(Y, TH, G, W, Ysq, MJ, hz, aw, fit_end)
    for _ in range(rounds):
        yh, kk, gi, bb, ss, rr = out
        G2 = m_step(Y, TH, kk, gi, bb, ss, W, Ysq, MJ, hz, aw, fit_end, K, tw)
        dead = np.linalg.norm(G2, axis=1) < 1e-9
        G2[dead] = G[dead]
        nxt = e_step(Y, TH, G2, W, Ysq, MJ, hz, aw, fit_end)
        if nxt[5].sum() > rr.sum():                              # ALS must not go uphill
            break
        G, out = G2, nxt
    return out, G


def main():
    t_all = time.time()
    names, Y, labels, future = af.load_lunar()
    TH, R = af.sky_lunar(labels + future)
    n = Y.shape[1]; Tn, nb = Y.shape[0], TH.shape[1]
    WALLS = list(range(n - 63, n - 29, 3))
    SEL, HELD = WALLS[:9], WALLS[9:]
    yr = lambda w: int(labels[w])
    w96 = WALLS[-1]

    t0 = time.time()
    base = {w: fit_base(Y, TH, w) for w in WALLS}
    base_auc = {w: auc_at(Y, predict(base[w]), w) for w in WALLS}
    b_sel = float(np.mean([base_auc[w] for w in SEL]))
    b_held = float(np.mean([base_auc[w] for w in HELD]))
    b_all = float(np.mean([base_auc[w] for w in WALLS]))
    print(f"  BASELINE 9p/topic ({9*Tn} params): select9 {b_sel:+.4f} · held3 {b_held:+.4f} · "
          f"all12 {b_all:+.4f} · 1996 {base_auc[w96]:+.4f}   [{time.time()-t0:.0f}s]", flush=True)

    UF = {}
    for w in WALLS:
        A = base[w]["c"][:, 1:]
        UF[w] = A / np.maximum(np.linalg.norm(A, axis=1), 1e-12)[:, None]

    KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
    ROUNDS = [0, 1, 2, 4]
    TW = ["flat", "share"]
    rows = []
    t0 = time.time()
    for K in KS:
        for rd in ROUNDS:
            for tw in TW:
                a = {}
                for w in WALLS:
                    (yh, kk, gi, bb, ss, rr), G = fit_dict(Y, TH, w, K, rd, UF[w], tw)
                    a[w] = auc_at(Y, yh, w)
                rows.append(dict(K=K, rounds=rd, tw=tw,
                                 sel=float(np.mean([a[w] for w in SEL])),
                                 held=float(np.mean([a[w] for w in HELD])),
                                 all=float(np.mean([a[w] for w in WALLS])), w1996=a[w96],
                                 params_per_topic=4, params_total=4 * Tn + 7 * K,
                                 per_wall={yr(w): round(a[w], 4) for w in WALLS}))
        best_k = max([r for r in rows if r["K"] == K], key=lambda r: r["sel"])
        print(f"  K={K:3d} · best sel {best_k['sel']:+.4f} (rounds={best_k['rounds']} "
              f"tw={best_k['tw']}) · held {best_k['held']:+.4f} · all {best_k['all']:+.4f} · "
              f"1996 {best_k['w1996']:+.4f} · {4*Tn+7*K} params", flush=True)
    t_grid = time.time() - t0

    print("\n  K CURVE (select-9, rounds/tw at their own select-9 best):", flush=True)
    for K in KS:
        b = max([r for r in rows if r["K"] == K], key=lambda r: r["sel"])
        print(f"    K={K:3d}  params {b['params_total']:5d} ({b['params_total']/(9*Tn):.2f}x baseline)"
              f"  sel {b['sel']:+.4f}  held {b['held']:+.4f}  all {b['all']:+.4f}  1996 {b['w1996']:+.4f}",
              flush=True)

    best = max(rows, key=lambda r: r["sel"])
    print(f"\n  SELECTED ON THE FIRST NINE ORIGINS ONLY: K={best['K']} rounds={best['rounds']} "
          f"tw={best['tw']} · {best['params_total']} params "
          f"({best['params_total']/(9*Tn):.2f}x the baseline's {9*Tn})", flush=True)
    for tag, k, bl in (("select9", "sel", b_sel), ("HELD3", "held", b_held),
                       ("all12", "all", b_all), ("1996", "w1996", base_auc[w96])):
        print(f"    {tag:8s} {best[k]:+.4f}  (baseline {bl:+.4f}, delta {best[k]-bl:+.4f})", flush=True)

    # ---- determinism: the fit has no seed, but prove it by refitting
    (yh1, k1, g1, b1, s1, r1), G1 = fit_dict(Y, TH, w96, best["K"], best["rounds"], UF[w96], best["tw"])
    (yh2, *_), G2 = fit_dict(Y, TH, w96, best["K"], best["rounds"], UF[w96], best["tw"])
    det = float(np.abs(yh1 - yh2).max())
    print(f"\n  determinism: refit max|dy| = {det:.3e} (no seed anywhere)", flush=True)

    print(f"  atom usage @1996: " +
          " ".join(f"{c}" for c in np.bincount(k1, minlength=best['K'])), flush=True)
    print("  atoms @1996 (" + " ".join(b[:2] for b in af.BODIES) + "):", flush=True)
    for k in range(min(best["K"], 8)):
        print("    g%-2d " % k + " ".join(f"{v:+.3f}" for v in G1[k]), flush=True)

    wall_s = time.time() - t_all
    res = dict(model="shared-spectrum dictionary: yhat_j = max(b_j + s_j*<g_k(j), cos(theta-phi_j)>,0)^2",
               baseline=dict(sel=b_sel, held=b_held, all=b_all, w1996=base_auc[w96],
                             params_per_topic=9, params_total=9 * Tn),
               walls=[yr(w) for w in WALLS], select=[yr(w) for w in SEL], held=[yr(w) for w in HELD],
               grid=rows, selected=best, deterministic=True, refit_max_dy=det,
               grid_seconds=round(t_grid, 1), wall_clock_s=round(wall_s, 1))
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  grid {t_grid:.0f}s · wall clock {wall_s:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
