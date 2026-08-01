#!/usr/bin/env python3
"""THE RESULT, on its own: partial pooling toward a FIELD-LEVEL SIGNED SPECTRUM.

Everything the campaign selected, in one runnable file. The model of record fits 251 INDEPENDENT
receivers, yhat_j(t) = max(b_j + SUM_i a_ji cos(theta_i(t) - phi_j), 0)^2, nine parameters a topic
and nothing shared. Here the 251 fits are tied together by a prior INSIDE the fit:

    minimise  SUM_t w_jt (b + SUM_i a_i cos(theta_i(t)-phi) - sqrt(y_jt))^2      (unchanged)
            + the thirty-year horizon anchor rows                                (unchanged)
            + tau * || a - s_j*e_j*g_F(j) ||^2                                   (the pooled prior)

g_F is the SHARED SPECTRUM of the topic's OpenAlex FIELD (26 of them), a unit vector in R^7; s_j is
the topic's own arrow scale and e_j its 180-degree gauge. Still a closed-form weighted least squares
at each of the 180 candidate tunings -- the prior is tau*I on the normal matrix and tau*target on
the right-hand side -- so the tuning is RE-SELECTED under the prior and the fit stays exhaustive and
deterministic. The target is recomputed from the shrunken arrows twice (rounds=2).

THE FRAME PROBLEM, which is the whole reason this works. Each topic carries its own tuning, so in a
fixed basis its seven arrows are COLLINEAR along its own axis and only their SIGNS pick a pole. The
0..179 sweep therefore fixes the 180-degree gauge by a convention with no physical content: average
the raw arrows across topics and you get ||mean|| = 0.081, i.e. nothing. Resolve the gauge first
(leading eigenvector of SUM u u', the estimator that is blind to u -> -u) and the same average has
||mean|| = 0.712. The spectrum was always there; the convention was hiding it.

SELECTION: prior form, pooling level, rounds and tau were chosen on the FIRST NINE origins
(1963..1987) in pool_refit.py and are hard-coded here. 1990/1993/1996 are held out.

  python3 analysis/arxivtopics/final_pooled.py
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arxiv_fit as af                                            # noqa: E402
from pool_shrink import auc_at, resolve_gauge                     # noqa: E402
from dict_spectrum import wall_weights                            # noqa: E402

OUT = os.path.join(HERE, "final_pooled_result.json")
TAU, ROUNDS, LEVEL = 0.003, 2, "field"        # selected on the first nine origins ONLY


def normal_eqs(Y, TH, fit_end):
    """Per-topic weighted normal equations at all 180 candidate tunings (the model of record's)."""
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
    return X, AM, BV


def sweep(X, AM, BV, target, tau):
    """argmin over the 180 tunings of the penalised objective; at the optimum J = -c.v, so the tuning
    is argmax(c.v) and no residual pass is needed. tau=0 is exactly arxiv_fit.fit_final."""
    Tn, nb = AM.shape[0], AM.shape[-1] - 1
    D = np.zeros((1 + nb, 1 + nb)); D[1:, 1:] = np.eye(nb)        # the level b is never penalised
    V = BV.copy()
    if tau > 0:
        V[:, :, 1:] += tau * target[:, None, :]
    C = np.linalg.solve(AM + tau * D[None, None], V[..., None])[..., 0]
    gi = np.argmax(np.einsum('jgp,jgp->jg', C, V), axis=1)
    ar = np.arange(Tn); c = C[ar, gi]
    return np.maximum(np.einsum('jtp,jp->jt', X[gi], c), 0.0) ** 2, c, gi


def field_target(A, grp):
    """Gauge-resolved, scale-preserving field spectrum: t_j = s_j * e_j * g_F(j)."""
    s = np.maximum(np.linalg.norm(A, axis=1), 1e-12)
    U = A / s[:, None]
    T = np.zeros_like(A)
    for f in np.unique(grp):
        m = grp == f
        g, e = resolve_gauge(U[m])
        T[m] = s[m][:, None] * e[:, None] * g[None, :]
    return T


def run(Y, TH, WALLS, grp, tau=TAU, rounds=ROUNDS):
    out = {}
    for w in WALLS:
        X, AM, BV = normal_eqs(Y, TH, w)
        yh0, c0, _ = sweep(X, AM, BV, None, 0.0)                  # the model of record
        A = c0[:, 1:]; yh = yh0
        for _ in range(rounds + 1):
            yh, c, _ = sweep(X, AM, BV, field_target(A, grp), tau)
            A = c[:, 1:]
        zh, _, _ = sweep(X, AM, BV, np.zeros_like(A), tau)        # ridge control at the same tau
        out[w] = (auc_at(Y, yh0, w), auc_at(Y, yh, w), auc_at(Y, zh, w))
        del X, AM, BV
    return out


def main():
    names, Y, labels, future = af.load_lunar()
    TH, R = af.sky_lunar(labels + future)
    n = Y.shape[1]; Tn, nb = Y.shape[0], TH.shape[1]
    WALLS = list(range(n - 63, n - 29, 3))
    SEL, HELD = WALLS[:9], WALLS[9:]
    yr = lambda w: int(labels[w]); w96 = WALLS[-1]
    grp = np.unique([af.META["field"][nm] for nm in names], return_inverse=True)[1]
    NF = int(grp.max() + 1)

    t0 = time.time()
    res = run(Y, TH, WALLS, grp)
    secs = time.time() - t0
    t1 = time.time(); res2 = run(Y, TH, WALLS, grp); secs2 = time.time() - t1
    spread = max(abs(res[w][1] - res2[w][1]) for w in WALLS)

    def agg(i):
        a = {w: res[w][i] for w in WALLS}
        return dict(sel=float(np.mean([a[w] for w in SEL])), held=float(np.mean([a[w] for w in HELD])),
                    all=float(np.mean([a[w] for w in WALLS])), w1996=a[w96],
                    per_wall={yr(w): round(a[w], 4) for w in WALLS})
    B, P, Z = agg(0), agg(1), agg(2)

    def pers(w):
        tv = af.META["topic_valid"]; tvw = tv[:, :w].astype(float)
        mu = (Y[:, :w] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
        last = Y[:, w - 1]
        return float(np.mean([1.0 - ((Y[:, w + h] - last) ** 2).sum() /
                              max(((Y[:, w + h] - mu) ** 2).sum(), 1e-9) for h in range(30)]))
    PE = dict(all=float(np.mean([pers(w) for w in WALLS])), w1996=pers(w96))

    print(f"  field-pooled receiver · tau={TAU} rounds={ROUNDS} level={LEVEL} ({NF} fields)", flush=True)
    print(f"  {'':22s}{'select9':>9s}{'held3':>9s}{'all12':>9s}{'1996':>9s}", flush=True)
    for tag, d in (("persistence (0 par)", PE), ("baseline (9 p/topic)", B),
                   ("ridge control @tau", Z), ("FIELD-POOLED", P)):
        s = d.get("sel"); h = d.get("held")
        print(f"  {tag:22s}" + (f"{s:+9.4f}{h:+9.4f}" if s is not None else " " * 18) +
              f"{d['all']:+9.4f}{d['w1996']:+9.4f}", flush=True)
    print(f"  {'delta vs baseline':22s}{P['sel']-B['sel']:+9.4f}{P['held']-B['held']:+9.4f}"
          f"{P['all']-B['all']:+9.4f}{P['w1996']-B['w1996']:+9.4f}", flush=True)
    wins = sum(1 for w in WALLS if res[w][1] > res[w][0])
    print(f"  wins {wins}/12 origins · deterministic: two full runs differ by {spread:.2e}", flush=True)
    print(f"  wall clock one 12-origin run: {secs:.1f}s (second run {secs2:.1f}s)", flush=True)
    print("  per wall  " + " ".join(f"{yr(w)}:{res[w][1]:+.4f}" for w in WALLS), flush=True)
    print("  baseline  " + " ".join(f"{yr(w)}:{res[w][0]:+.4f}" for w in WALLS), flush=True)

    A = None
    X, AM, BV = normal_eqs(Y, TH, w96)
    _, c0, _ = sweep(X, AM, BV, None, 0.0)
    A = c0[:, 1:]
    U = A / np.maximum(np.linalg.norm(A, axis=1), 1e-12)[:, None]
    g, e = resolve_gauge(U)
    print(f"\n  frame check @1996: ||mean(u)|| naive {np.linalg.norm(U.mean(0)):.4f} -> "
          f"gauge-resolved {np.linalg.norm((e[:,None]*U).mean(0)):.4f}", flush=True)
    print("  global spectrum (" + " ".join(b[:2] for b in af.BODIES) + "): " +
          " ".join(f"{v:+.3f}" for v in g), flush=True)

    json.dump(dict(model="partial pooling toward a field-level signed spectrum (penalised refit)",
                   tau=TAU, rounds=ROUNDS, level=LEVEL, n_groups=NF,
                   params_per_topic_nominal=9, params_global=7 * NF,
                   params_total_nominal=9 * Tn + 7 * NF, effective_params_per_topic=8.87,
                   walls=[yr(w) for w in WALLS], select=[yr(w) for w in SEL],
                   held=[yr(w) for w in HELD], baseline=B, pooled=P, ridge_control=Z,
                   persistence=PE, wins_vs_baseline=wins, deterministic=True,
                   seed_spread=float(spread), seconds_one_run=round(secs, 1)),
              open(OUT, "w"), indent=1)
    print(f"\n  -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
