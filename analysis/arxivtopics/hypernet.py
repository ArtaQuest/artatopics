#!/usr/bin/env python3
"""THE MAP — a GLOBAL model that PREDICTS each topic's parameters out of its own pre-wall history and
its taxonomy instead of FITTING them.  ZERO free parameters per topic: the whole model is the map, so
it can serve a topic it has never seen, which 251 independent curve fits can never do.

    features_j  ->  (b_j, phi_j, sign pattern s_ji, magnitudes m_ji)
    yhat_j(t)   =  max( b_j + SUM_i s_ji*m_ji*cos(theta_i(t) - phi_j), 0 )^2

Features are read ONLY before the wall: level at four horizons, two trends, two variability ratios,
age since emergence, log own-citations, residual autocorrelation, plus the topic's DOMAIN and FIELD
one-hots and quantile bins of the four headline numerics.  Nothing at or after the wall is touched,
ever.  Optional extra blocks (proj/quad) are the topic's own weighted covariance with each body's
cos/sin — still pre-wall, still its own history — offered to selection and NOT chosen.

FOUR ARCHITECTURES, one honest sweep.
  distill    ridge from features to the baseline's own fitted parameters.  Minimises error in
             PARAMETER space, which is not the space anything is scored in.  Worst of the four.
  joint      the map fitted STRAIGHT TO THE DATA.  With phi fixed the model is linear in (b,u,v), and
             (b,u,v) is linear in the features, so the entire global map is ONE weighted least
             squares: sqrt(y_j(t)) ~ s_j * SUM_f x_jf * [B_f0 + SUM_i (B^u_fi cos th_i + B^v_fi sin
             th_i)], 15*F unknowns, closed form.  The horizon anchor folds in as extra rows exactly as
             it does in fit_final.  s_j is the topic's own anchor level, so the map is scale-free.
             NOTE: its seven arrows are not collinear, so it spends 15 numbers a topic, not 9.
  joint_proj joint, then projected onto the shared-tuning manifold (phi = the principal axis of the
             seven (u,v) arrows).  This RESTORES the nine-parameter form and it is a disaster: the
             orthogonal component the projection discards is carrying the fit.
  twostage   the form-preserving route that works.  (1) a ridge on the DOUBLE ANGLE (phi lives mod
             180 deg, so cos2phi/sin2phi is its only honest basis) predicts the tuning; (2) with
             phi_j fixed, one joint weighted LS fits b and the SEVEN SIGNED ARROWS as linear
             functions of the same features.  One phi + seven signed arrows per topic, every number
             read off a global map.  The SIGNS survive: they are the sign of a fitted coefficient,
             which is the binary second phase the campaign showed does the work.

Everything is closed form — ridge and normal equations.  No seed, no iterations, no early stopping.

SELECTION DISCIPLINE.  Architecture, feature blocks and ridge alpha are chosen on the FIRST NINE
origins (1963..1987) and applied unchanged; 1990/1993/1996 are reported as held out.  The sweep table
is cached to hypernet_sweep.json keyed by the exact grid, so a rerun cannot silently reuse a table
built from a different one.  The runner-up architecture is reported too, explicitly flagged as NOT
selected, because its selection margin is far smaller than the origin-to-origin spread and reading
its held-out numbers and then crowning it is precisely the mistake this campaign has shipped twice.

  python3 analysis/arxivtopics/hypernet.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import arxiv_fit as af

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "analysis/arxivtopics/hypernet_result.json")

NAMES, Y, LABELS, FUTURE = af.load_lunar()
TH, R = af.sky_lunar(LABELS + FUTURE)
Tn, n = Y.shape
ne = TH.shape[0]
NB = TH.shape[1]
WALLS = list(range(n - 63, n - 29, 3))          # TWELVE origins 1963..1996
SEL, HELD = WALLS[:9], WALLS[9:]                # selection / held-out origins
TV = af.META["topic_valid"]
EV = np.clip(af.META["evidence"], 0, None)
DOMAIN = np.array([af.META["domain"][nm] for nm in NAMES])
FIELD = np.array([af.META["field"][nm] for nm in NAMES])
DOMS = sorted(set(DOMAIN)); FLDS = sorted(set(FIELD))
COS = np.cos(TH); SIN = np.sin(TH)


# ---------------------------------------------------------------- the score (verbatim from the brief)
def auc_at(Yt, Yhat, wall, rows=None):
    tvw = TV[:, :wall].astype(float)
    mu = (Yt[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    if rows is None:
        rows = slice(None)
    return float(np.mean([1.0 - ((Yt[rows, wall + h] - Yhat[rows, wall + h]) ** 2).sum() /
                          max(((Yt[rows, wall + h] - mu[rows]) ** 2).sum(), 1e-9) for h in range(30)]))


def persistence_auc(wall, rows=None):
    tvw = TV[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    last = np.repeat(Y[:, wall - 1][:, None], 30, 1)
    Yh = np.zeros_like(Y); Yh[:, wall:wall + 30] = last
    return auc_at(Y, Yh, wall, rows)


# ---------------------------------------------------------------- per-topic targets (the thing to replace)
def weights_at(wall):
    """The EXACT weighting arxiv_fit.fit_final uses: per-topic valid mask x N^0.75 evidence."""
    tv = TV[:, :wall].astype(float)
    wy = EV[:wall] ** 0.75
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    Wa = np.zeros_like(W)
    Wa[:, wall - af.ANCHOR_K:] = (tv * wy[None])[:, wall - af.ANCHOR_K:]
    bad = Wa.sum(1) <= 0
    Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    s = np.maximum((np.sqrt(Y[:, :wall]) * Wa).sum(1), 1e-3)     # the anchor level == the topic's scale
    return W, s


_TCACHE = {}
def targets_at(wall):
    """Per-topic baseline fit -> (b, phi in [0,pi), signed arrows a).  This is arxiv_fit.fit_final's
    own output, de-canonicalised: it returns |a| and a per-body tuning that is phi or phi+180."""
    if wall in _TCACHE:
        return _TCACHE[wall]
    yh, prm = af.fit_final(Y, TH, wall)
    P, A, B = prm["p"], prm["a"], prm["b"]
    sign = np.where(P < np.pi - 1e-9, 1.0, -1.0)                 # GRID is [0,180) so P>=180 means a<0
    phi = P[:, 0] % np.pi
    a = A * sign
    _TCACHE[wall] = (yh, B, phi, a)
    return _TCACHE[wall]


def curve(b, phi, a):
    """yhat over ALL years from (b, phi, signed arrows) — the nine-parameter model, verbatim."""
    C = b[:, None] + (a[:, None, :] * np.cos(TH[None, :, :] - phi[:, None, None])).sum(2)
    return np.maximum(C, 0.0) ** 2


# ---------------------------------------------------------------- features (pre-wall history + taxonomy ONLY)
def _wmean_last(ys, wy, valid_idx, k):
    idx = valid_idx[-k:] if k else valid_idx
    if len(idx) == 0:
        return 1e-3
    w = wy[idx]
    return float(max((ys[idx] * w).sum() / max(w.sum(), 1e-12), 1e-6))


_FCACHE = {}
def features_at(wall):
    """All blocks.  Nothing at or after `wall` is ever read."""
    if wall in _FCACHE:
        return _FCACHE[wall]
    W, s = weights_at(wall)
    wy = EV[:wall] ** 0.75
    YS = np.sqrt(Y[:, :wall])
    num = np.zeros((Tn, 11))
    for j in range(Tn):
        vi = np.where(TV[j, :wall])[0]
        ys = YS[j]
        if len(vi) < 3:
            vi = np.arange(max(wall - 3, 0), wall)
        l5 = _wmean_last(ys, wy, vi, 5); lall = _wmean_last(ys, wy, vi, 0)
        l15 = _wmean_last(ys, wy, vi, 15); l40 = _wmean_last(ys, wy, vi, 40)
        prev5 = _wmean_last(ys, wy, vi[:-5] if len(vi) > 6 else vi, 5)
        z = ys[vi]
        seg = z[-30:] if len(z) >= 6 else z
        t = np.arange(len(seg), dtype=float)
        lg = np.log(np.maximum(seg, 1e-6))
        sl = np.polyfit(t, lg, 1)[0] if len(seg) > 2 else 0.0
        res = lg - np.polyval(np.polyfit(t, lg, 1), t) if len(seg) > 2 else lg * 0
        ac1 = float(np.corrcoef(res[:-1], res[1:])[0, 1]) if len(res) > 4 and res.std() > 1e-9 else 0.0
        z20 = z[-20:]
        cit = float((Y[j, :wall] * TV[j, :wall] * EV[:wall]).sum() / 100.0)
        num[j] = [np.log(s[j]), np.log(lall / s[j]), np.log(l15 / s[j]), np.log(l40 / s[j]),
                  np.log(l5 / max(prev5, 1e-6)), np.clip(sl * 10, -5, 5),
                  z20.std() / max(z20.mean(), 1e-6) if len(z20) else 0.0,
                  z.std() / max(z.mean(), 1e-6), np.log(1 + len(vi)),
                  np.log(1 + cit), 0.0 if not np.isfinite(ac1) else ac1]
    num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
    dom = np.stack([(DOMAIN == d).astype(float) for d in DOMS], 1)
    fld = np.stack([(FIELD == f).astype(float) for f in FLDS], 1)
    # SKY PROJECTIONS — weighted covariance of the topic's own pre-wall sqrt-share with each body's
    # cos/sin, scaled by the topic's own level.  Purely pre-wall, purely the topic's own history.
    Wc = W[:, :wall]
    ybar = (YS * Wc).sum(1, keepdims=True)
    yc = YS - ybar
    # sum_t W*yc == 0 by construction, so the weighted covariance is just <W, yc*cos>
    pc = np.einsum('jt,ti->ji', Wc * yc, COS[:wall]) / s[:, None]
    ps = np.einsum('jt,ti->ji', Wc * yc, SIN[:wall]) / s[:, None]
    proj = np.concatenate([pc, ps], 1)
    # DOUBLE-ANGLE (quadratic) features: phi is defined mod 180 deg, so its natural basis is 2*phi.
    Rn = np.maximum((pc ** 2 + ps ** 2).sum(1, keepdims=True), 1e-12)
    quad = np.concatenate([(pc ** 2 - ps ** 2) / Rn, 2 * pc * ps / Rn], 1)
    # QUANTILE BINS of the four headline numerics — capacity for a NON-linear level/age/trend response
    # without a tree.  Edges are cross-topic quantiles of a FEATURE at this wall: no labels, no
    # post-wall data.  (Mildly transductive in the unseen-topic split; noted in the result file.)
    bins = []
    for c in (0, 5, 8, 9):
        q = np.quantile(num[:, c], [0.25, 0.5, 0.75])
        d = np.digitize(num[:, c], q)
        bins.append(np.stack([(d == k).astype(float) for k in range(4)], 1))
    B = dict(num=num, dom=dom, fld=fld, proj=np.nan_to_num(proj), quad=np.nan_to_num(quad),
             bins=np.concatenate(bins, 1))
    _FCACHE[wall] = (B, s)
    return _FCACHE[wall]


BLOCKSETS = {
    "T":        ["num", "dom", "fld"],                     # the literal brief: level/trend/var/age/ev + taxonomy
    "T-nofld":  ["num", "dom"],
    "T+P":      ["num", "dom", "fld", "proj"],
    "T+P+Q":    ["num", "dom", "fld", "proj", "quad"],
    "Tn+P+Q":   ["num", "dom", "proj", "quad"],
    "P+Q":      ["proj", "quad"],
    "num+P+Q":  ["num", "proj", "quad"],
    "T+bins":   ["num", "dom", "fld", "bins"],
    "T+bins+P": ["num", "dom", "fld", "bins", "proj"],
    # TAXONOMY-ONLY maps.  These are the interpretable rungs of the ladder: with only the field
    # one-hot the map degenerates to ONE receiver per FIELD (26 of them, not 251); with only the
    # domain one-hot, one per DOMAIN (4).  A topic the model has never seen still gets a forecast
    # from its label alone.
    "fld-only": ["fld"],
    "dom-only": ["dom"],
    "num-only": ["num"],
}


def design(wall, blocks):
    B, s = features_at(wall)
    return np.concatenate([B[k] for k in blocks], 1), s


# ---------------------------------------------------------------- the map: ridge, closed form, no seed
def zstats(X, rows):
    """Standardisation read off the TRAIN rows only.  A column that is CONSTANT on the train rows
    (e.g. a field one-hot whose only topics are all held out) gets scale 1, not 1/epsilon — dividing
    by a floor would hand an unseen topic a feature value of 1e9 and detonate the forecast."""
    Xt = X[rows]
    sd = Xt.std(0)
    return Xt.mean(0), np.where(sd < 1e-8, 1.0, sd)


def ridge(X, T, alpha, rows, tw=None):
    """Standardise on the TRAIN rows only, solve (X'WX + aI)c = X'WT.  Returns a predict fn."""
    Xt = X[rows]
    mu, sd = zstats(X, rows)
    Z = (Xt - mu) / sd
    w = np.ones(len(rows)) if tw is None else tw[rows]
    Zw = Z * w[:, None]
    A = Zw.T @ Z + alpha * np.eye(Z.shape[1])
    ybar = (T[rows] * w[:, None]).sum(0) / w.sum()
    c = np.linalg.solve(A, Zw.T @ (T[rows] - ybar))
    return lambda Xn: ((Xn - mu) / sd) @ c + ybar, c.size + ybar.size


def project(uv):
    """Collinear projection: the shared tuning is the principal axis of the seven (u,v) arrows."""
    u, v = uv[:, :NB], uv[:, NB:]
    phi = 0.5 * np.arctan2(2 * (u * v).sum(1), (u ** 2 - v ** 2).sum(1)) % np.pi
    a = u * np.cos(phi)[:, None] + v * np.sin(phi)[:, None]
    return phi, a


def map_predict(wall, blocks, alpha, train_rows, apply_rows):
    """Fit the map on train_rows' baseline parameters; predict apply_rows' parameters from features."""
    _, Bt, phit, at = targets_at(wall)
    X, s = design(wall, blocks)
    u = at * np.cos(phit)[:, None]; v = at * np.sin(phit)[:, None]
    T = np.concatenate([(Bt / s)[:, None], u / s[:, None], v / s[:, None]], 1)   # 15 scale-free targets
    f, npar = ridge(X, T, alpha, train_rows)
    P = f(X[apply_rows])
    b = P[:, 0] * s[apply_rows]
    phi, a = project(P[:, 1:] * s[apply_rows][:, None])
    return b, phi, a, npar, X.shape[1]


# ---------------------------------------------------------------- MODE 2: the JOINT map (fit to the DATA)
# Distilling the baseline's fitted parameters minimises error in PARAMETER space, which is not the
# space anyone is scored in.  Because the model is linear once phi is fixed, AND the parameters are
# linear in the features, the WHOLE global map is one weighted least-squares problem:
#     sqrt(y_j(t))  ~  s_j * SUM_f x_jf * [ B_f0 + SUM_i ( B_fi^u cos th_i(t) + B_fi^v sin th_i(t) ) ]
# 15*F unknowns, closed form, no seed.  The horizon anchor folds in as extra rows exactly as it does
# in fit_final.  Afterwards each topic's implied (u,v) arrows are projected onto the shared-tuning
# manifold, which is what recovers the nine-parameter form (one phi + seven signs + seven lengths).
G = np.concatenate([np.ones((ne, 1)), COS, SIN], 1)               # (ne, 15) the sky design


_JCACHE = {}
def joint_normal(wall, blocks, rows_key, rows):
    key = (wall, "".join(blocks), rows_key)
    if key in _JCACHE:
        return _JCACHE[key]
    W, s = weights_at(wall)
    X, _ = design(wall, blocks)
    mu, sd = zstats(X, rows)
    Z = np.concatenate([np.ones((Tn, 1)), (X - mu) / sd], 1)      # intercept + standardised features
    Xs = Z * s[:, None]                                           # params scale with the topic's level
    Wc = W[:, :wall]
    Gt = G[:wall]
    M = np.einsum('jt,tk,tl->jkl', Wc, Gt, Gt)
    Q = np.einsum('jt,tk->jk', Wc * np.sqrt(Y[:, :wall]), Gt)
    hz = min(wall + af.HORIZON, ne)
    aw = af.LAM_HORIZON / (s ** 2) / max(hz - wall, 1)            # the scale-free horizon anchor
    Ga = G[wall:hz]
    M = M + aw[:, None, None] * (Ga.T @ Ga)[None]
    Q = Q + (aw * s)[:, None] * Ga.sum(0)[None]
    P = np.einsum('jf,jg->jfg', Xs, Xs)
    F = Z.shape[1]
    A = np.einsum('jfg,jkl->fkgl', P[rows], M[rows]).reshape(F * 15, F * 15)
    bv = np.einsum('jf,jk->fk', Xs[rows], Q[rows]).reshape(-1)
    _JCACHE[key] = (A, bv, Z, s, F)
    return _JCACHE[key]


def joint_predict(wall, blocks, alpha, rows_key, train_rows, apply_rows, proj=True):
    A, bv, Z, s, F = joint_normal(wall, blocks, rows_key, train_rows)
    pen = np.ones(F * 15) * alpha
    pen[:15] = 0.0                    # never shrink the INTERCEPT block: alpha->inf must fall back to
    B = np.linalg.solve(A + np.diag(pen) + 1e-10 * np.eye(F * 15), bv).reshape(F, 15)   # the global model
    p = (Z[apply_rows] @ B) * s[apply_rows][:, None]
    b = p[:, 0]
    if not proj:
        return b, p[:, 1:1 + NB], p[:, 1 + NB:], None, F * 15, F - 1
    phi, a = project(p[:, 1:])
    return b, phi, a, None, F * 15, F - 1


# ---------------------------------------------------------------- MODE 3: two-stage, SHARED TUNING kept
# The unconstrained joint map above throws the nine-parameter form away (its seven arrows are not
# collinear, and projecting them afterwards destroys the fit).  Two stages keep the form intact:
#   1. a ridge on the DOUBLE ANGLE (phi lives mod 180 deg) predicts each topic's tuning from features;
#   2. with phi_j fixed the model is again linear, so one joint weighted LS over all topics fits
#      b and the SEVEN SIGNED arrows as linear functions of the same features.
# Result: one phi + seven signed arrows per topic, every one of them read off a global map.
def phase_map(wall, blocks, alpha, train_rows, apply_rows):
    _, _, phit, _ = targets_at(wall)
    X, s = design(wall, blocks)
    T2 = np.stack([np.cos(2 * phit), np.sin(2 * phit)], 1)
    f, npar = ridge(X, T2, alpha, train_rows)
    P = f(X[apply_rows])
    return 0.5 * np.arctan2(P[:, 1], P[:, 0]) % np.pi, npar


def mq_given_phi(wall, phi_all):
    """Anchored weighted normal-equation blocks for the shared-tuning design [1, cos(th_i - phi_j)]."""
    W, s = weights_at(wall)
    Gj = np.concatenate([np.ones((Tn, ne, 1)),
                         np.cos(TH[None, :, :] - phi_all[:, None, None])], 2)      # (Tn, ne, 8)
    Wc = W[:, :wall]; Gt = Gj[:, :wall]
    M = np.einsum('jt,jtk,jtl->jkl', Wc, Gt, Gt, optimize=True)
    Q = np.einsum('jt,jtk->jk', Wc * np.sqrt(Y[:, :wall]), Gt, optimize=True)
    hz = min(wall + af.HORIZON, ne)
    aw = af.LAM_HORIZON / (s ** 2) / max(hz - wall, 1)
    Ga = Gj[:, wall:hz]
    M = M + aw[:, None, None] * np.einsum('jtk,jtl->jkl', Ga, Ga, optimize=True)
    Q = Q + (aw * s)[:, None] * Ga.sum(1)
    return M, Q, s


def wls_given_phi(wall, phi_all):
    """PER-TOPIC anchored WLS with the tuning handed in — eight free numbers a topic, no map."""
    M, Q, s = mq_given_phi(wall, phi_all)
    c = np.linalg.solve(M + 1e-8 * np.eye(M.shape[1])[None], Q[..., None])[..., 0]
    return c[:, 0], c[:, 1:]


def joint_shared(wall, blocks, alpha, train_rows, apply_rows, phi_all):
    X, _ = design(wall, blocks)
    M, Q, s = mq_given_phi(wall, phi_all)
    K = M.shape[1]
    mu, sd = zstats(X, train_rows)
    Z = np.concatenate([np.ones((Tn, 1)), (X - mu) / sd], 1)
    Xs = Z * s[:, None]
    F = Z.shape[1]
    P = np.einsum('jf,jg->jfg', Xs, Xs, optimize=True)
    A = np.einsum('jfg,jkl->fkgl', P[train_rows], M[train_rows], optimize=True).reshape(F * K, F * K)
    bv = np.einsum('jf,jk->fk', Xs[train_rows], Q[train_rows], optimize=True).reshape(-1)
    pen = np.ones(F * K) * alpha; pen[:K] = 0.0
    B = np.linalg.solve(A + np.diag(pen) + 1e-10 * np.eye(F * K), bv).reshape(F, K)
    p = (Z[apply_rows] @ B) * s[apply_rows][:, None]
    return p[:, 0], p[:, 1:], F * K


ALL = np.arange(Tn)


def eval_map(wall, blocks, alpha, train_rows, apply_rows, mode="distill", proj=True):
    if mode == "twostage":
        Yh = np.zeros((Tn, ne))
        phi_all, np1 = phase_map(wall, blocks, alpha, train_rows, np.arange(Tn))
        b, a, np2 = joint_shared(wall, blocks, alpha, train_rows, apply_rows, phi_all)
        Yh[apply_rows] = curve(b, phi_all[apply_rows], a)
        nf = design(wall, blocks)[0].shape[1]
        return auc_at(Y, Yh, wall, apply_rows), np1 + np2, nf, (b, phi_all[apply_rows], a)
    Yh = np.zeros((Tn, ne))
    if mode == "distill":
        b, phi, a, npar, nf = map_predict(wall, blocks, alpha, train_rows, apply_rows)
        Yh[apply_rows] = curve(b, phi, a)
        return auc_at(Y, Yh, wall, apply_rows), npar, nf, (b, phi, a)
    rk = "all" if len(train_rows) == Tn else "tr"
    if proj:
        b, phi, a, _, npar, nf = joint_predict(wall, blocks, alpha, rk, train_rows, apply_rows)
        Yh[apply_rows] = curve(b, phi, a)
        return auc_at(Y, Yh, wall, apply_rows), npar, nf, (b, phi, a)
    b, u, v, _, npar, nf = joint_predict(wall, blocks, alpha, rk, train_rows, apply_rows, proj=False)
    C = b[:, None] + u @ COS.T + v @ SIN.T
    Yh[apply_rows] = np.maximum(C, 0) ** 2
    return auc_at(Y, Yh, wall, apply_rows), npar, nf, (b, u, v)


MODES = ["distill", "joint", "joint_proj", "twostage"]
ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]


def run(wall, mode, blocks, alpha, train_rows, apply_rows):
    if mode == "joint":
        return eval_map(wall, blocks, alpha, train_rows, apply_rows, mode="joint", proj=False)
    if mode == "joint_proj":
        return eval_map(wall, blocks, alpha, train_rows, apply_rows, mode="joint", proj=True)
    return eval_map(wall, blocks, alpha, train_rows, apply_rows, mode=mode)


def agg(d, extra=None):
    o = {"per_wall": [round(d[w], 4) for w in WALLS],
         "select9": round(float(np.mean([d[w] for w in SEL])), 4),
         "held3": round(float(np.mean([d[w] for w in HELD])), 4),
         "all12": round(float(np.mean([d[w] for w in WALLS])), 4), "y1996": round(d[WALLS[-1]], 4)}
    if extra:
        o.update(extra)
    return o


def main():
    t0 = time.time()
    res = {"walls": [int(LABELS[w]) for w in WALLS], "select_origins": [int(LABELS[w]) for w in SEL],
           "held_origins": [int(LABELS[w]) for w in HELD]}

    base = {w: auc_at(Y, targets_at(w)[0], w) for w in WALLS}
    pers = {w: persistence_auc(w) for w in WALLS}
    res["baseline_per_topic"] = agg(base, {"params_per_topic": 9, "params_total": 9 * Tn})
    res["persistence"] = agg(pers, {"params_total": 0})
    print(f"  baseline  sel9 {res['baseline_per_topic']['select9']:+.4f} held3 "
          f"{res['baseline_per_topic']['held3']:+.4f} all12 {res['baseline_per_topic']['all12']:+.4f} "
          f"1996 {res['baseline_per_topic']['y1996']:+.4f}", flush=True)
    print(f"  persist   sel9 {res['persistence']['select9']:+.4f} held3 {res['persistence']['held3']:+.4f}"
          f" all12 {res['persistence']['all12']:+.4f} 1996 {res['persistence']['y1996']:+.4f}", flush=True)

    # ---- SELECTION on the FIRST NINE ORIGINS ONLY --------------------------------------------------
    # Cached to disk keyed by the exact grid: the sweep is deterministic, so a rerun must not re-pay
    # for it — and must not silently reuse a table built from a DIFFERENT grid either.
    sig = json.dumps({"m": MODES, "b": {k: v for k, v in BLOCKSETS.items()}, "a": ALPHAS,
                      "sel": [int(LABELS[w]) for w in SEL]}, sort_keys=True)
    cache = os.path.join(REPO, "analysis/arxivtopics/hypernet_sweep.json")
    tbl = None
    if os.path.exists(cache):
        c = json.load(open(cache))
        if c.get("sig") == sig:
            tbl = c["tbl"]; print("    ... selection sweep reused from cache", flush=True)
    if tbl is None:
        tbl = {}
        for mode in MODES:
            for bk, blocks in BLOCKSETS.items():
                for aa in ALPHAS:
                    try:
                        sc = [run(w, mode, blocks, aa, ALL, ALL)[0] for w in SEL]
                    except np.linalg.LinAlgError:
                        continue
                    tbl[f"{mode}|{bk}|{aa}"] = round(float(np.mean(sc)), 4)
            print(f"    ... {mode} swept ({round(time.time()-t0)}s)", flush=True)
        json.dump({"sig": sig, "tbl": tbl}, open(cache, "w"), indent=1)
    best = max(tbl, key=tbl.get)
    mode, bk, al = best.split("|"); al = float(al); blocks = BLOCKSETS[bk]
    res["selection_top12"] = dict(sorted(tbl.items(), key=lambda kv: -kv[1])[:12])
    res["chosen"] = {"mode": mode, "blocks": bk, "block_list": blocks, "alpha": al,
                     "select9_at_selection": tbl[best]}
    print(f"  SELECTED (9 origins only): {mode} · {bk} · alpha={al}  -> sel9 {tbl[best]:+.4f}", flush=True)

    # ---- (a) IN-SAMPLE TOPICS ----------------------------------------------------------------------
    ins, npar, nf = {}, 0, 0
    for w in WALLS:
        ins[w], npar, nf, _ = run(w, mode, blocks, al, ALL, ALL)
    res["a_in_sample_topics"] = agg(ins, {"n_features": int(nf), "params_total": int(npar),
                                          "params_per_topic": 0.0})
    print(f"  (a) MAP   sel9 {res['a_in_sample_topics']['select9']:+.4f} held3 "
          f"{res['a_in_sample_topics']['held3']:+.4f} all12 {res['a_in_sample_topics']['all12']:+.4f} "
          f"1996 {res['a_in_sample_topics']['y1996']:+.4f}  [{nf} feats · {npar} global params]", flush=True)

    # runner-up modes, same discipline, for the record
    res["other_modes"] = {}
    for m in MODES:
        cand = {k: v for k, v in tbl.items() if k.startswith(m + "|")}
        if not cand:
            continue
        bb = max(cand, key=cand.get); _, b2, a2 = bb.split("|")
        d = {}
        for w in WALLS:
            d[w] = run(w, m, BLOCKSETS[b2], float(a2), ALL, ALL)[0]
        res["other_modes"][m] = agg(d, {"blocks": b2, "alpha": float(a2),
                                        "params_total": int(run(WALLS[0], m, BLOCKSETS[b2], float(a2),
                                                                ALL, ALL)[1])})
        print(f"      mode {m:11s} ({b2},{a2}) all12 {res['other_modes'][m]['all12']:+.4f} "
              f"1996 {res['other_modes'][m]['y1996']:+.4f}", flush=True)

    # ---- THE RUNNER-UP, FLAGGED AS SUCH -------------------------------------------------------------
    # The best config that did NOT win selection.  It is reported because the selection margin on the
    # nine origins is far smaller than the spread between origins — i.e. the sweep does not actually
    # resolve these two — and because reading its held-out numbers and then calling it the winner is
    # the exact mistake this campaign has already shipped twice.  It is an OBSERVATION, not a claim.
    cand = {k: v for k, v in tbl.items() if not k.startswith(mode + "|")}
    runner2 = max(cand, key=cand.get)          # best config from a DIFFERENT architecture
    rm, rb, ra = runner2.split("|"); ra = float(ra)
    rd = {}
    for w in WALLS:
        rd[w], rnp, _, _ = run(w, rm, BLOCKSETS[rb], ra, ALL, ALL)
    ru = {"mode": rm, "blocks": rb, "alpha": ra, "params_total": int(rnp),
          "select9_at_selection": tbl[runner2],
          "lost_selection_by": round(tbl[best] - tbl[runner2], 4)}
    res["runner_up_not_selected"] = agg(rd, ru)
    print(f"  RUNNER-UP (NOT selected, lost by {ru['lost_selection_by']:+.4f} on the 9 origins): "
          f"{rm} · {rb} · alpha={ra} -> sel9 {tbl[runner2]:+.4f} held3 "
          f"{res['runner_up_not_selected']['held3']:+.4f} all12 "
          f"{res['runner_up_not_selected']['all12']:+.4f} 1996 "
          f"{res['runner_up_not_selected']['y1996']:+.4f}  [{rnp} params]", flush=True)

    # ---- FLOOR: a no-feature map (intercept block only) ---------------------------------------------
    zero = {}
    for w in WALLS:
        zero[w] = run(w, mode, BLOCKSETS["T-nofld"], 1e12, ALL, ALL)[0]
    res["no_feature_map"] = agg(zero, {"note": "alpha=1e12 — every feature shrunk away, the intercept "
                                               "block only: ONE global scale-free parameter vector"})
    print(f"  FLOOR no-feature map  all12 {res['no_feature_map']['all12']:+.4f}", flush=True)

    # ---- (b) UNSEEN TOPICS -------------------------------------------------------------------------
    rs = np.random.RandomState(0)
    perm = rs.permutation(Tn)
    nho = int(round(0.2 * Tn))
    ho, tr = np.sort(perm[:nho]), np.sort(perm[nho:])
    rmode, rbk, ral = ru["mode"], ru["blocks"], ru["alpha"]
    u_map, u_ctl, u_per, u_zero, u_run = {}, {}, {}, {}, {}
    for w in WALLS:
        u_map[w] = run(w, mode, blocks, al, tr, ho)[0]
        u_ctl[w] = auc_at(Y, targets_at(w)[0], w, ho)
        u_per[w] = persistence_auc(w, ho)
        u_zero[w] = run(w, mode, BLOCKSETS["T-nofld"], 1e12, tr, ho)[0]
        u_run[w] = run(w, rmode, BLOCKSETS[rbk], ral, tr, ho)[0]
    res["b_unseen_topics"] = {"n_held_out": int(nho), "seed": "RandomState(0)",
                              "map": agg(u_map), "control_own_independent_fit": agg(u_ctl),
                              "control_persistence": agg(u_per), "control_no_feature_map": agg(u_zero),
                              "runner_up_map": agg(u_run)}
    print(f"  (b) UNSEEN {nho} topics: map {agg(u_map)['all12']:+.4f} | own-fit "
          f"{agg(u_ctl)['all12']:+.4f} | persistence {agg(u_per)['all12']:+.4f} | no-feature "
          f"{agg(u_zero)['all12']:+.4f}", flush=True)

    # ---- WHICH HALF IS PREDICTABLE — CONSTRAINED REFITS, not cross-fit swaps ------------------------
    # Swapping one parameter group between two independent fits is meaningless here: b and the arrows
    # live on the same amplitude scale and the model rectifies and SQUARES their sum, so a mismatched
    # pair explodes.  The honest question is "fix this group globally and let the topic adapt the
    # rest".  Two hybrids answer it, and they bracket the whole parameter set.
    tsbk, tsal = res["other_modes"]["twostage"]["blocks"], res["other_modes"]["twostage"]["alpha"]
    h_ownphi, h_mapphi, h_ownphi_ownarr, spread = {}, {}, {}, []
    for w in WALLS:
        _, Bt, phit, at = targets_at(w)
        # H1 — the topic's OWN tuning (1 number a topic), b + seven signed arrows from the global map
        b1, a1, np1 = joint_shared(w, blocks, al, ALL, ALL, phit)
        h_ownphi[w] = auc_at(Y, curve(b1, phit, a1), w)
        # H2 — the MAP's tuning, b + seven signed arrows refitted per topic (8 numbers a topic)
        phim, _ = phase_map(w, BLOCKSETS[tsbk], tsal, ALL, ALL)
        b2, a2 = wls_given_phi(w, phim)
        h_mapphi[w] = auc_at(Y, curve(b2, phim, a2), w)
        # H3 — control: the topic's OWN tuning with per-topic arrows == the baseline, to check the
        # anchored WLS reproduces fit_final when nothing is globalised
        b3, a3 = wls_given_phi(w, phit)
        h_ownphi_ownarr[w] = auc_at(Y, curve(b3, phit, a3), w)
        spread.append(float(np.rad2deg(np.abs(np.angle(np.exp(2j * phim)))).std()))
    res["hybrid_own_phi_map_arrows"] = agg(h_ownphi, {"params_per_topic": 1, "params_global": int(np1)})
    res["hybrid_map_phi_own_arrows"] = agg(h_mapphi, {"params_per_topic": 8})
    res["control_own_phi_own_arrows"] = agg(h_ownphi_ownarr, {"params_per_topic": 9,
                                                              "note": "must reproduce the baseline"})
    res["predicted_phi_spread_deg"] = [round(x, 2) for x in spread]
    print(f"  HYB own-phi + map-arrows (1/topic) all12 {res['hybrid_own_phi_map_arrows']['all12']:+.4f}"
          f" · map-phi + own-arrows (8/topic) all12 {res['hybrid_map_phi_own_arrows']['all12']:+.4f}"
          f" · control {res['control_own_phi_own_arrows']['all12']:+.4f}", flush=True)
    print(f"  the phase map's predicted tuning has cross-topic spread {np.mean(spread):.2f} deg "
          f"(features={tsbk}) — under ~5 deg it has collapsed to ONE global angle", flush=True)

    # ---- one full 12-origin run, timed COLD ---------------------------------------------------------
    # Every cache cleared first, or the number is a lie: by this point the features, the normal
    # equations and the baseline fits are all warm.  The joint map never calls fit_final at all, so a
    # cold run of it is genuinely independent of the per-topic baseline.
    _FCACHE.clear(); _JCACHE.clear(); _TCACHE.clear()
    t1 = time.time()
    for w in WALLS:
        run(w, mode, blocks, al, ALL, ALL)
    res["wall_clock_one_12origin_run_s"] = round(time.time() - t1, 2)
    res["wall_clock_note"] = ("cold: all caches cleared. The chosen (joint) map never calls "
                              "arxiv_fit.fit_final — it is fitted straight to the data.")
    res["wall_clock_total_s"] = round(time.time() - t0, 1)
    res["deterministic"] = True
    res["seed_spread"] = 0.0
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"  one 12-origin run: {res['wall_clock_one_12origin_run_s']}s · total {res['wall_clock_total_s']}s"
          f"\n  saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
