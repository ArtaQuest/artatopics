#!/usr/bin/env python3
"""LOW-RANK ARROW MATRIX — is the 251x7 arrow matrix of the /topics model actually 251 independent
rows, or is it a handful of GLOBAL basis spectra that every field mixes?

THE DEPLOYED MODEL (arxiv_fit.fit_final), unchanged as the bar:
    yhat_j(t) = max( b_j + SUM_i a_ji cos(theta_i(t) - phi_j), 0 )^2
251 topics x (1 level + 1 tuning + 7 SIGNED arrows) = 9 params/topic, 2,259 total, nothing shared.

THIS MODEL: force the arrow matrix to rank r,   A = U V^T,  U (251,r) per-topic loadings,
V (7,r) GLOBAL basis spectra shared by every field.  Per topic: b_j, phi_j, u_j  ->  r+2.
Global: 7r.  r=7 recovers the baseline exactly; r=1 is the already-tested shared-shape variant.

WHY THIS IS NOT THE eight_struct.py "shape" TEST THAT ALREADY LOST. That variant (a_ji = s_j g_i)
collapsed the seven bodies into ONE regressor column, so a topic's only sign freedom was the single
sign of s_j -- it could not flip one body against another. Here u_j and V are both unconstrained in
sign, so a topic CAN oppose bodies as long as the flip lies in the r-dimensional span. The campaign's
central finding is that the SIGN PATTERN does the work (forcing a_i >= 0 costs -0.1858 AUC), so
whether the reachable sign patterns cover the ones the free model actually uses is the whole question.
measure_sign_reachability() answers it with an LP, per rank, not by assertion.

FIT: deterministic alternation, no seed, no gradients.
  topic step  given V, each topic sweeps phi over the same 1-degree half-circle grid and solves in
              closed form for (b_j, u_j) on design columns  SUM_i V[i,k] cos(theta_i - phi).
  V step      given every topic's (phi_j, b_j, u_j), one pooled weighted least-squares for the 7r
              numbers in V.  M = SUM_j kron(S_j, u_j u_j^T), an exact 7r x 7r normal equation.
  Both steps minimise the SAME anchored weighted SSE, so the objective decreases monotonically and
  the alternation runs to a fixed point (rel. change < 1e-6). Init: SVD of the free (rank-7) fit's
  own signed arrow matrix at that wall -- train-only data, deterministic, no seed.
The N^0.75 evidence weights, the per-topic non-zero crop and the horizon anchor are folded into the
normal equations exactly as fit_final does them.

SPEED. Writing X_t = CTaug[g] @ Vaug (Vaug = blockdiag([1], V)) makes the per-topic 8x8 moment
tensors G[j,g] = CTaug[g]^T diag(w_j) CTaug[g] INDEPENDENT of V and of r -- computed once per wall
and reused by every rank and every iteration. A whole alternation step is then a (1+r)-column
projection of an 8x8 matrix.

SELECTION DISCIPLINE. r is chosen on the FIRST NINE origins only (1963..1987) and then applied
unchanged to the held-out three (1990/1993/1996). The alternation runs to a fixed point, so the
iteration count is not a tuned constant; the 1e-8 on the diagonal is the same numerical floor
fit_final uses, not a penalty. NOTHING is selected on the scored walls.

  python3 analysis/arxivtopics/lowrank_arrows.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import arxiv_fit as af

OUT = "analysis/arxivtopics/lowrank_arrows.json"
RANKS = [1, 2, 3, 4, 5, 6]
MAXIT = 400                        # a cap, not a tuned constant -- the run must reach TOL before it
TOL = 1e-7
FLOOR = 1e-8                       # the numerical floor fit_final uses, not a fitted penalty

names, Y, labels, future = af.load_lunar()
TH, R = af.sky_lunar(labels + future)
Tn, n = Y.shape
ne, nb = TH.shape
WALLS = list(range(n - 63, n - 29, 3))          # TWELVE origins 1963..1996
SELECT, HELD = WALLS[:9], WALLS[9:]
GRID = af.GRID
NG = len(GRID)
YSQ = np.sqrt(Y)

# CTaug[g,t,p]: p=0 is the intercept, p=1..7 is cos(theta_i(t) - GRID[g]).
CTAUG = np.empty((NG, ne, 1 + nb))
CTAUG[:, :, 0] = 1.0
for i in range(nb):
    CTAUG[:, :, 1 + i] = np.cos(TH[:, i][None, :] - GRID[:, None])


def auc_at(Yhat, wall):
    """THE ONLY SCORE -- copied verbatim from the brief."""
    tv = af.META["topic_valid"]; tvw = tv[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    return float(np.mean([1.0 - ((Y[:, wall + h] - Yhat[:, wall + h]) ** 2).sum() /
                max(((Y[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(30)]))


# ---------------------------------------------------------------- per-wall precompute (V-independent)
class Wall:
    """Everything the alternation needs at one origin, in the 8-column cosine basis.

    G[j,g] and h[j,g] are the topic's weighted moment matrix / cross-product at tuning g. They do not
    depend on V or on r, so one build serves all six ranks and every iteration."""

    def __init__(self, wall):
        self.wall = wall
        hz = self.hz = min(wall + af.HORIZON, ne)
        tv = af.META["topic_valid"][:, :wall].astype(float)
        wy = np.clip(af.META["evidence"][:wall], 0, None) ** 0.75
        W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
        Wa = np.zeros_like(W)
        Wa[:, wall - af.ANCHOR_K:] = (tv * wy[None])[:, wall - af.ANCHOR_K:]
        bad = Wa.sum(1) <= 0
        Wa[bad] = (tv * wy[None])[bad]
        Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
        self.MJ = np.maximum((YSQ[:, :wall] * Wa).sum(1), 1e-3)
        self.aw = af.LAM_HORIZON / (self.MJ ** 2) / max(hz - wall, 1)

        Xt = CTAUG[:, :wall]                                    # (NG, wall, 8)
        P = 1 + nb
        Q = (Xt[:, :, :, None] * Xt[:, :, None, :]).reshape(NG, wall, P * P)
        self.G = np.tensordot(W, Q, axes=([1], [1])).reshape(Tn, NG, P, P)     # (Tn,NG,8,8)
        self.h = np.tensordot(W * YSQ[:, :wall], Xt, axes=([1], [1]))          # (Tn,NG,8)
        Xa = CTAUG[:, wall:hz]
        self.Ga = np.einsum('gtp,gtq->gpq', Xa, Xa)                            # (NG,8,8)
        self.GaB = Xa.sum(1)                                                   # (NG,8)
        # anchored moments folded in: A[j,g] = G + aw_j*Ga ,  b[j,g] = h + aw_j*MJ_j*GaB
        self.Aj = self.G + self.aw[:, None, None, None] * self.Ga[None]
        self.bj = self.h + (self.aw * self.MJ)[:, None, None] * self.GaB[None]


def topic_step(w, Vaug):
    """Given V (as Vaug), sweep phi and solve (b_j,u_j) in closed form. Returns g, c, coef8, obj.

    All contractions are batched BLAS matmuls, not einsum: A = Vaug' Aj Vaug for every (topic,tuning)
    pair at once. einsum picks a naive order here and is ~20x slower."""
    K = Vaug.shape[1]
    A = np.matmul(Vaug.T, np.matmul(w.Aj, Vaug))                              # (Tn,NG,K,K)
    b = np.matmul(w.bj, Vaug)                                                 # (Tn,NG,K)
    c = np.linalg.solve(A + FLOOR * np.eye(K), b[..., None])[..., 0]          # (Tn,NG,K)
    # residual up to a per-topic constant: c'Ac - 2c'b
    r = (np.matmul(A, c[..., None])[..., 0] * c).sum(-1) - 2.0 * (c * b).sum(-1)
    g = np.argmin(r, axis=1)
    cj = c[np.arange(Tn), g]                                                   # (Tn,K)
    coef8 = cj @ Vaug.T                                                        # (Tn,8) = (b_j, a_j1..7)
    # exact objective (add back the per-topic constants so the trace is comparable across steps)
    obj = float(r[np.arange(Tn), g].sum())
    return g, cj, coef8, obj


def v_step(w, g, cj, r):
    """Pooled weighted least-squares for the 7r numbers in V, holding (phi_j, b_j, u_j) fixed."""
    U = cj[:, 1:]                                                              # (Tn,r)
    bb = cj[:, 0]
    Gg = w.Aj[np.arange(Tn), g]                                                # (Tn,8,8) anchored
    S = Gg[:, 1:, 1:]                                                          # (Tn,7,7)
    # v_j[i] = h[1+i] - b_j*G[0,1+i]  +  aw_j*(MJ_j - b_j)*GaB[1+i]
    v = w.h[np.arange(Tn), g][:, 1:] - bb[:, None] * w.G[np.arange(Tn), g][:, 0, 1:] \
        + (w.aw * (w.MJ - bb))[:, None] * w.GaB[g][:, 1:]
    M = np.einsum('jab,jk,jl->akbl', S, U, U, optimize=True).reshape(nb * r, nb * r)
    rhs = np.einsum('ja,jk->ak', v, U, optimize=True).ravel()
    Vn = np.linalg.solve(M + FLOOR * np.trace(M) / (nb * r) * np.eye(nb * r) + 1e-12 * np.eye(nb * r),
                         rhs).reshape(nb, r)
    nrm = np.linalg.norm(Vn, axis=0)                                           # gauge fix: unit columns
    Vn = Vn / np.maximum(nrm, 1e-12)
    return Vn


def free_fit(w):
    """The BASELINE (rank-7, nothing shared) in this machinery -- must reproduce fit_final exactly."""
    Vaug = np.eye(1 + nb)
    g, cj, coef8, obj = topic_step(w, Vaug)
    yh = np.maximum(np.einsum('jtp,jp->jt', CTAUG[g], coef8), 0.0) ** 2
    return yh, coef8, g


def lowrank_fit(w, r, V0=None, trace=False, alternate=True):
    """Deterministic alternation to a fixed point. Returns (yhat, V, U, b, g, iters, aucs_by_iter).

    alternate=False stops at the SVD initialisation (V = top-r right singular vectors of the free
    fit's own arrow matrix, one topic step) -- reported as a diagnostic, NOT as the model, so we can
    say whether the pooled V update earns its place or merely fits the training window harder."""
    if V0 is None:
        _, coef8, _ = free_fit(w)
        A7 = coef8[:, 1:]                                                      # free signed arrows
        _, _, Vt = np.linalg.svd(A7, full_matrices=False)
        V0 = Vt[:r].T                                                          # (7,r) top-r right sv
        V0 = V0 / np.maximum(np.linalg.norm(V0, axis=0), 1e-12)
    V = V0.copy()
    prev, aucs = None, []
    for it in range(MAXIT if alternate else 0):
        Vaug = np.zeros((1 + nb, 1 + r)); Vaug[0, 0] = 1.0; Vaug[1:, 1:] = V
        g, cj, coef8, obj = topic_step(w, Vaug)
        if trace:
            yh = np.maximum(np.einsum('jtp,jp->jt', CTAUG[g], coef8), 0.0) ** 2
            aucs.append(round(auc_at(yh, w.wall), 6))
        if prev is not None and abs(prev - obj) <= TOL * max(abs(prev), 1e-12):
            break                                  # objective at a fixed point (it decreases monotonically)
        prev = obj
        V = v_step(w, g, cj, r)
    Vaug = np.zeros((1 + nb, 1 + r)); Vaug[0, 0] = 1.0; Vaug[1:, 1:] = V
    g, cj, coef8, obj = topic_step(w, Vaug)
    yh = np.maximum(np.einsum('jtp,jp->jt', CTAUG[g], coef8), 0.0) ** 2
    return yh, V, cj[:, 1:], cj[:, 0], g, (it + 1 if alternate else 0), aucs


# ---------------------------------------------------------------- sign reachability
def measure_sign_reachability(V, S):
    """Can rank-r loadings express the sign patterns the FREE model actually uses?

    sign(V u) ranges over the full-dimensional cells of the arrangement of 7 hyperplanes in R^r, so
    at most 2*SUM_{k<r} C(6,k) of the 128 patterns are reachable at all. For each topic we solve the
    LP  diag(s) V u >= 1  -- feasible iff that exact pattern is expressible."""
    from scipy.optimize import linprog
    r = V.shape[1]
    uniq, inv, cnt = np.unique(S, axis=0, return_inverse=True, return_counts=True)
    ok = np.zeros(len(uniq), bool)
    for i, s in enumerate(uniq):
        res = linprog(c=np.zeros(r), A_ub=-(s[:, None] * V), b_ub=-np.ones(nb),
                      bounds=[(None, None)] * r, method="highs")
        ok[i] = bool(res.status == 0)
    return dict(topics_reachable=int(cnt[ok].sum()), topics_total=int(S.shape[0]),
                patterns_reachable=int(ok.sum()), patterns_used=int(len(uniq)),
                cells_max=int(2 * sum(_comb(nb - 1, k) for k in range(r))))


def _comb(n, k):
    from math import comb
    return comb(n, k)


def persistence_auc(wall):
    tvw = af.META["topic_valid"][:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    last = Y[:, wall - 1]
    return float(np.mean([1.0 - ((Y[:, wall + h] - last) ** 2).sum() /
                          max(((Y[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(30)]))


def summarise(per_wall):
    d = dict(zip(WALLS, per_wall))
    return dict(select_mean_9=round(float(np.mean([d[w] for w in SELECT])), 4),
                held_mean_3=round(float(np.mean([d[w] for w in HELD])), 4),
                all_mean_12=round(float(np.mean(per_wall)), 4),
                auc_1996=round(float(d[WALLS[-1]]), 4),
                per_wall={labels[w]: round(v, 4) for w, v in d.items()})


def main():
    t0 = time.time()
    print(f"LOW-RANK ARROW MATRIX  A = U V^T   {Tn} topics x {nb} bodies x {n} years", flush=True)
    print(f"  origins: {[labels[w] for w in WALLS]}", flush=True)
    print(f"  select on {[labels[w] for w in SELECT]} | held out {[labels[w] for w in HELD]}\n", flush=True)

    walls = {}
    for w in WALLS:
        walls[w] = Wall(w)
    print(f"  precompute {time.time()-t0:.1f}s", flush=True)

    res = {}
    # --- the bar: baseline + persistence, re-measured here, never quoted from memory
    base, pers, freeA, freeS = [], [], {}, {}
    for w in WALLS:
        yh, coef8, g = free_fit(walls[w])
        base.append(auc_at(yh, w)); pers.append(persistence_auc(w))
        freeA[w] = coef8[:, 1:]
        S = np.sign(coef8[:, 1:]); S[S == 0] = 1.0
        freeS[w] = S
    res["baseline_free_rank7"] = summarise(base); res["baseline_free_rank7"]["params_total"] = 9 * Tn
    res["baseline_free_rank7"]["params_per_topic"] = 9
    res["persistence"] = summarise(pers); res["persistence"]["params_total"] = 0
    print(f"  BASELINE (9/topic, {9*Tn} total)  select-9 {res['baseline_free_rank7']['select_mean_9']:+.4f}"
          f" · held-3 {res['baseline_free_rank7']['held_mean_3']:+.4f}"
          f" · all-12 {res['baseline_free_rank7']['all_mean_12']:+.4f}"
          f" · 1996 {res['baseline_free_rank7']['auc_1996']:+.4f}", flush=True)
    print(f"  PERSISTENCE (0)                   select-9 {res['persistence']['select_mean_9']:+.4f}"
          f" · held-3 {res['persistence']['held_mean_3']:+.4f}"
          f" · all-12 {res['persistence']['all_mean_12']:+.4f}"
          f" · 1996 {res['persistence']['auc_1996']:+.4f}\n", flush=True)

    # --- the r-curve, for BOTH fitting families, scored identically.
    #   svd  V = top-r right singular vectors of the free fit's own arrow matrix (truncated SVD,
    #        closed form, train-only), then one topic step.
    #   alt  the briefed alternation: the same start, then pooled V updates to a fixed point.
    # Both are deterministic. Reporting only one of them would hide the finding.
    res["ranks"] = {"svd": {}, "alt": {}}
    tcurve = time.time()
    for r in RANKS:
        line = {}
        for fam, alt in (("svd", False), ("alt", True)):
            aucs, iters, Vs = [], [], {}
            for w in WALLS:
                yh, V, U, b, g, it, _ = lowrank_fit(walls[w], r, alternate=alt)
                aucs.append(auc_at(yh, w)); iters.append(it); Vs[w] = V
            s = summarise(aucs)
            s["params_per_topic"] = r + 2
            s["params_global"] = nb * r
            s["params_total"] = Tn * (r + 2) + nb * r
            s["params_gauge_corrected"] = Tn * (r + 2) + nb * r - r * r
            s["iters"] = [int(x) for x in iters]
            s["hit_maxit"] = bool(max(iters) >= MAXIT)
            s["signs_1996"] = measure_sign_reachability(Vs[WALLS[-1]], freeS[WALLS[-1]])
            reach = [measure_sign_reachability(Vs[w], freeS[w])["topics_reachable"] / Tn for w in WALLS]
            s["signs_mean_frac_topics_reachable_12"] = round(float(np.mean(reach)), 4)
            res["ranks"][fam][r] = s
            line[fam] = s
        for fam in ("svd", "alt"):
            s = line[fam]; sg = s["signs_1996"]
            print(f"  {fam:>3} r={r} ({r+2}/topic + {nb*r} global = {s['params_total']:>4})  "
                  f"select-9 {s['select_mean_9']:+.4f} · held-3 {s['held_mean_3']:+.4f} · "
                  f"all-12 {s['all_mean_12']:+.4f} · 1996 {s['auc_1996']:+.4f}  | signs "
                  f"{sg['topics_reachable']:>3}/{Tn} topics, {sg['patterns_reachable']:>2}/"
                  f"{sg['patterns_used']} patterns (<= {sg['cells_max']}/128 cells)"
                  + (f" | iters<={max(s['iters'])}" if fam == "alt" else ""), flush=True)
    curve_secs = time.time() - tcurve

    # --- THE SELECTION: family AND rank chosen jointly on the first nine origins ONLY.
    cand = [(f, r) for f in ("svd", "alt") for r in RANKS]
    fam_star, r_star = max(cand, key=lambda k: res["ranks"][k[0]][k[1]]["select_mean_9"])
    res["selection"] = dict(
        rule="argmax select_mean_9 jointly over family in {svd,alt} x r in 1..6 -- first NINE origins only",
        chosen_family=fam_star, chosen_r=int(r_star),
        select_mean_9={f: {r: res["ranks"][f][r]["select_mean_9"] for r in RANKS} for f in ("svd", "alt")})
    ch = res["ranks"][fam_star][r_star]
    print(f"\n  SELECTED ON THE FIRST NINE ORIGINS: family={fam_star}  r={r_star}", flush=True)
    print(f"  HELD OUT (1990/1993/1996) at that choice: {ch['held_mean_3']:+.4f} "
          f"(baseline {res['baseline_free_rank7']['held_mean_3']:+.4f})", flush=True)

    # --- wall clock for ONE full 12-origin run at the chosen setting (incl. precompute)
    t1 = time.time()
    w2 = {w: Wall(w) for w in WALLS}
    for w in WALLS:
        lowrank_fit(w2[w], r_star, alternate=(fam_star == "alt"))
    res["wallclock_one_12origin_run_secs"] = round(time.time() - t1, 1)
    res["curve_all_ranks_secs"] = round(curve_secs, 1)

    # --- determinism: refit twice, bit-compare
    a = lowrank_fit(walls[WALLS[-1]], r_star, alternate=(fam_star == "alt"))[0]
    b_ = lowrank_fit(walls[WALLS[-1]], r_star, alternate=(fam_star == "alt"))[0]
    res["deterministic"] = bool(np.array_equal(a, b_))
    res["seed_spread"] = 0.0

    # --- the AUC-vs-iteration trace at the headline origin: does alternating help or hurt OOS?
    res["iteration_trace_1996_alt"] = {r: lowrank_fit(walls[WALLS[-1]], r, trace=True)[6][:25]
                                       for r in RANKS}

    # --- what the global basis actually is, at the chosen setting, at the headline origin
    yh, V, U, b, g, it, _ = lowrank_fit(walls[WALLS[-1]], r_star, alternate=(fam_star == "alt"))
    res["V_1996"] = dict(bodies=af.BODIES, V=np.round(V, 4).tolist())
    Arec = U @ V.T
    res["arrow_recon_1996"] = dict(
        rel_fro_error=round(float(np.linalg.norm(Arec - freeA[WALLS[-1]]) /
                                  max(np.linalg.norm(freeA[WALLS[-1]]), 1e-12)), 4),
        sign_agreement=round(float((np.sign(Arec) == freeS[WALLS[-1]]).mean()), 4))
    res["wallclock_full_script_secs"] = round(time.time() - t0, 1)
    res["config"] = dict(ranks=RANKS, maxit=MAXIT, tol=TOL, floor=FLOOR,
                         walls=[labels[w] for w in WALLS],
                         select=[labels[w] for w in SELECT], held=[labels[w] for w in HELD],
                         lam_horizon=af.LAM_HORIZON, anchor_k=af.ANCHOR_K, horizon=af.HORIZON,
                         bodies=af.BODIES, weight_exponent=0.75)

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  one full 12-origin run at r={r_star}: {res['wallclock_one_12origin_run_secs']}s "
          f"· deterministic {res['deterministic']} · whole script "
          f"{res['wallclock_full_script_secs']}s", flush=True)
    print(f"  saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
