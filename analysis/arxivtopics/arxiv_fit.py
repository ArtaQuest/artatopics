#!/usr/bin/env python3
"""MODEL OF RECORD (operator 2026-07-28, FINAL): the DETERMINISTIC rectified receiver — one shared
tuning per topic, seven signed arrows, solved in closed form. Primary criterion: THIRTY-YEAR AUC.

DATA: the platform's public citations rail (OpenAlex snapshot, CC0; notebook nb/9303, DOI
zenodo.21537062), 251 OpenAlex subfields × years 1700..2025 at YEARLY resolution. Target
y_j(t) = 100·citations_received[j,t] / Σ_k citations_received[k,t] — topic j's share of the year's
citations. Each topic is fit only over its own consistently-non-zero suffix, so an emerging field is
fit from emergence and none is dropped. Sky sampled at each year's mid-point (sidereal Lahiri). SEVEN
bodies — mars, jupiter, saturn, uranus, neptune, pluto, lunar node. The sun and inner planets are
excluded (at yearly sampling the sun returns to ~the same sidereal longitude each year, so it carries
no signal, and mercury/venus alias to noise); chiron overfits at the thirty-year horizon.

MODEL (251 topics j, 7 bodies i) — NINE parameters per topic, and nothing shared:
    ŷ_j(t) = max( b_j + Σᵢ a_jᵢ·cos(θᵢ(t) − φ_j) , 0 )²
  a rectified square-law (diode) detector: dark while the summed projection is negative, bright on a
  transit. It fits because a field's share sits near zero for long stretches and then spikes — exactly
  the shape a rectifier draws. ONE tuning φ_j serves all seven bodies; the arrows a_jᵢ are SIGNED, and
  a negative arrow is exactly a 180° flip of that body against the shared tuning. So the model is one
  continuous phase plus seven binary ones, which is the measured optimum: forcing aᵢ ≥ 0 costs −0.1858
  over twelve origins and loses at ALL twelve (nonneg_nine.json — measured on THIS model; the earlier
  −0.1778 was measured on the 8-param jupiter-dropped variant and should not be quoted for this one),
  and giving every body its own free continuous phase costs −0.0145 at the thirty-year wall
  (staged_phases.py). 9 params/topic × 251 = 2,259 total. Everything is a function of the sky alone.

TRAINING: no optimiser, no seed, no early stopping — refit it and you get the same bits. With φ fixed
the model is linear on the amplitude scale, so a 1° sweep of φ with a weighted least-squares solve for
(b, a₁..a₇) at each step fits it exactly. Years are weighted by N^¾ evidence (a share from N citations
carries noise ∝ 1/√N, which alone gives √N; an inner rehearsal wall — fit ≤1965, judged 1966-1995,
test years never touched — selects the heavier N^0.70-0.80 plateau, of which N^¾ is the midpoint).
The HORIZON ANCHOR is folded into the same solve as extra rows: the model's own forecast over the next
thirty years is pulled toward the level the field held in its last five TRAINING years, weighted
λ/m² so it is scale-free. It reads only planet angles (known centuries ahead) and training data, and
it exists because the failure mode was never the fit but the extrapolation — the slow bodies sweep
only a fraction of a cycle inside the window and drift like an unpinned trend once the data runs out.

WHY THIS AND NOT THE GRADIENT MODEL — and where it is worse, said first. The superseded v10 (15
params/topic — seven free tunings, Adam, softplus arrows, L1 on √share, twelve-sign KL term) is the
BETTER model at 3 of 12 origins (1963, 1966, 1996) and at all THREE walls the page displays: 4-yr
+0.9757 vs +0.9737, 12-yr +0.9489 vs +0.9318, 30-yr +0.8193 vs +0.7990. We switched anyway. Over
twelve origins (1963..1996, each forecasting its own next thirty years) the deterministic model
averages +0.8751 against +0.8533, winning 9 of 12 (Δ +0.0218, SE 0.0081 — but those windows overlap
heavily, so that SE understates the true uncertainty and the margin is a direction of travel, not a
measurement). It is also 40% smaller and exactly reproducible, which the gradient fit is not. Those
three displayed walls are three readings of one recent cut-off; twelve origins ask a wider question.
fit_market() below is kept as the superseded v10 for that comparison; main() does not use it.

BENCHMARKS: honest walls at n−30 (the HEADLINE — fit ≤1995, forecast 1996-2025, across the entire
internet and deep-learning era the model never saw), n−12 and n−4, each scored against carrying the
topic's own train-window mean forward, and against the harder bar of carrying its LAST OBSERVED YEAR
forward. Deploy = all but the last year, extended through 2055 — the next thirty years.

  python3 analysis/arxivtopics/arxiv_fit.py
"""
import importlib.util as u, os
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
p2 = _load("analysis/adstopics/astro_phasor2.py", "p2")
EXCLUDE = {"sun", "moon", "mercury", "venus", "chiron"}   # operator 2026-07-24: mars IN, CHIRON OUT
# CHIRON dropped (30-yr ablation, operator "switch to 30-yr AUC"): at the 30-year wall chiron overfits —
# leave-one-out −chiron lifts the honest 30-yr AUC +0.581→+0.641 and median skill +0.135→+0.399. The
# fast bodies (mars/jupiter) also thin out at 30y but stay per the operator's roster order.
BODIES = [b for b in p2.BODIES if b not in EXCLUDE]       # SEVEN bodies: mars ju sa ur ne pl node
EXT_YEARS = 30                                             # 2026..2055 — the next THIRTY years (30-yr headline)
LAM_HORIZON = 0.03                                         # horizon-anchor strength — chosen by ROLLING-ORIGIN
ANCHOR_K = 5                                               # rehearsal at the 1936 AND 1966 origins, never 1996
HORIZON = 30                                               # the deployment horizon the anchor spans
BETA_KL = 0.02                                             # KL phase term — chosen on the INNER wall
TAU_KL = 0.15                                              # (rehearsal at 1965; 1996+ never touched)
NF = 1                                                     # ONE tide — robust winner on the cropped 117-yr window
# CITATIONS-RECEIVED source (operator 2026-07-23): the platform's public citations rail
# (Extra::citations_quarterly · OpenAlex snapshot 2026-06-26, CC0) — the EVENT measure, each
# reference edge dated by the CITING work's quarter. Derived matrix built by build_matrix().
MATRIX = "analysis/citations/citations_received_yearly.csv"
META = {"id": {}, "field": {}, "domain": {}}              # subfield -> id/field/domain (for the page)


def load_lunar():
    """YEARLY citations-received shares (operator 2026-07-24; name kept for the exporter's import).
    Each subfield's value = its % of the YEAR's total citations received (columns sum to ~100). Window
    1700..2025 (2026 is a partial year). At yearly resolution the year-only-dated citations are exact —
    no imputation. Sun + inner planets (mercury/venus) are excluded — at yearly sampling the sun returns to
    ~the same sidereal longitude each year (no signal) and the inner planets alias to noise; CHIRON is
    excluded too (overfits at the 30-year horizon). Seven bodies remain: mars..pluto + the lunar node."""
    df = pd.read_csv(MATRIX)
    labels = [c for c in df.columns if c not in ("subfield_id", "subfield", "field", "domain")]
    # A handful of OpenAlex subfields share a display name across different parent fields
    # (e.g. "Biochemistry" under both Molecular Biology and Medicine) — disambiguate by field.
    dup = df["subfield"].duplicated(keep=False)
    names = [f"{r['subfield']} ({r['field']})" if dup[i] else r["subfield"]
             for i, (_, r) in enumerate(df.iterrows())]
    for nm, (_, r) in zip(names, df.iterrows()):
        META["id"][nm] = int(r["subfield_id"])
        META["field"][nm] = str(r["field"])
        META["domain"][nm] = str(r["domain"])
    M = df[labels].to_numpy(float)
    tot = M.sum(0)
    META["valid"] = tot > 0
    META["evidence"] = tot                                 # citations per year — the share's evidence
    # PER-TOPIC crop (operator): each topic's own consistently-non-zero suffix (leading zeros masked).
    Z = M > 0; nz = np.ones_like(Z, bool)
    for i in range(M.shape[1] - 2, -1, -1): nz[:, i] = Z[:, i] & nz[:, i + 1]
    META["topic_valid"] = nz
    META["start"] = {nm: labels[int(np.argmax(nz[j]))] for j, nm in enumerate(names)}
    S = 100.0 * M / np.maximum(tot[None, :], 1.0)          # share = citations / year-total
    y0 = int(labels[-1])
    future = [str(y0 + k) for k in range(1, EXT_YEARS + 1)]
    return names, S, labels, future


def sky_lunar(dates):
    E = pd.read_csv("analysis/arxivtopics/_ephemeris_yearly.csv").set_index("Time")
    E.index = E.index.astype(str)
    TH = np.stack([np.deg2rad(E[f"{b}_lon"].loc[dates].to_numpy(float)) for b in BODIES], 1)
    R = np.stack([E[f"{b}_dist"].loc[dates].to_numpy(float) for b in BODIES], 1)
    assert np.isfinite(TH).all() and (R > 0).all()
    return TH, R


# THE SWEEP IS A HALF-CIRCLE, NOT A CIRCLE. Rotating φ by 180° and negating every signed amplitude is
# the SAME model — an exact algebraic identity — so a 0..359 sweep evaluates every candidate twice and
# then picks between two numerically tied residuals on a few ulps of floating-point noise. That would
# make "refit it and you get the same bits" true only for one BLAS. Sweeping 0..179 removes the
# degeneracy, halves the work, and changes nothing: the canonical per-body tunings (φ or φ+180,
# decided by each arrow's sign) come out identical either way.
GRID = np.deg2rad(np.arange(0, 180, 1.0))                  # the 1° tuning sweep — the whole "training"


def fit_final(Y, TH, fit_end):
    """THE MODEL OF RECORD — ŷ_j = max(b_j + Σᵢ a_jᵢcos(θᵢ − φ_j), 0)², ONE shared tuning per topic.

    LINEAR IN DISGUISE. On the amplitude scale C = b + Σᵢ aᵢcos(θᵢ − φ) expands to
    b + Σᵢ[aᵢcosφ·cosθᵢ + aᵢsinφ·sinθᵢ], so once φ is FIXED the remaining eight numbers are an ordinary
    weighted least-squares solve. Sweeping φ over a 1° grid and keeping the best therefore fits the
    model EXACTLY — there is no optimiser to converge, no seed to draw, nothing to stop early. That is
    the point: the atlas the page draws is a claim about 251 research fields, and a claim you cannot
    reproduce bit-for-bit is a weaker claim.

    THE ANCHOR IS A LINEAR CONSTRAINT. "Your own forecast over the next thirty years should stay near
    the level of the last five training years" is not a penalty that needs gradients — it is extra
    rows in the same normal equations, weighted λ/m² so it is scale-free. It costs nothing to include.

    SIGNED ARROWS ARE THE SECOND PHASE. aᵢ < 0 is exactly a 180° flip of body i against φ. That single
    binary freedom per body is what lets one shared tuning serve seven bodies at all: remove it
    (aᵢ ≥ 0) and the model collapses by −0.1778 AUC; promote it to a free continuous phase per body and
    it overfits by −0.0145. Returns (ŷ over data+forecast years, params a/p/b) in the CANONICAL form
    the atlas exports — arrows carry length only, and the sign is folded into each body's tuning."""
    Tn, ne, nb = Y.shape[0], TH.shape[0], TH.shape[1]
    tv = META["topic_valid"][:, :fit_end].astype(float)
    wy = np.clip(META["evidence"][:fit_end], 0, None) ** 0.75          # N^¾ evidence (rehearsal-selected)
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    Wa = np.zeros_like(W)                                              # the anchor's ANCHOR_K-year memory
    Wa[:, fit_end - ANCHOR_K:] = (tv * wy[None])[:, fit_end - ANCHOR_K:]
    bad = Wa.sum(1) <= 0                                               # no valid year there -> whole history
    Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    Ysq = np.sqrt(Y)
    MJ = np.maximum((Ysq[:, :fit_end] * Wa).sum(1), 1e-3)              # the level to hold the forecast near
    hz = min(fit_end + HORIZON, ne)

    NG = len(GRID)
    X = np.empty((NG, ne, 1 + nb)); X[:, :, 0] = 1.0                   # design at every candidate tuning
    for i in range(nb):
        X[:, :, 1 + i] = np.cos(TH[:, i][None, :] - GRID[:, None])
    Xt, Xa = X[:, :fit_end], X[:, fit_end:hz]
    XaS = np.einsum('gtp,gtq->gpq', Xa, Xa); XaB = np.einsum('gtp->gp', Xa)
    I = 1e-8 * np.eye(1 + nb)[None]                                    # numerical floor, not a penalty

    yh = np.zeros((Tn, ne)); A = np.zeros((Tn, nb)); P = np.zeros((Tn, nb)); B = np.zeros(Tn)
    for j in range(Tn):
        aw = LAM_HORIZON / (MJ[j] ** 2) / max(hz - fit_end, 1)         # scale-free anchor weight
        Amat = np.einsum('gtp,t,gtq->gpq', Xt, W[j], Xt) + aw * XaS + I
        bvec = np.einsum('gtp,t->gp', Xt, W[j] * Ysq[j, :fit_end]) + aw * XaB * MJ[j]
        c = np.linalg.solve(Amat, bvec[..., None])[..., 0]             # closed form at every tuning
        r = (((np.einsum('gtp,gp->gt', Xt, c) - Ysq[j, :fit_end][None]) ** 2) @ W[j]
             + aw * ((np.einsum('gtp,gp->gt', Xa, c) - MJ[j]) ** 2).sum(1))
        g = int(np.argmin(r))                                          # the tuning, by exhaustive sweep
        yh[j] = np.maximum(X[g] @ c[g], 0.0) ** 2                      # rectified square-law detector
        a_signed = c[g][1:]
        # aᵢcos(θᵢ−φ) ≡ |aᵢ|cos(θᵢ−(φ+180°)) for aᵢ < 0 — an exact identity, so every downstream share,
        # dominant body and pair term is computed on positive arrow lengths without changing the model.
        A[j] = np.abs(a_signed)
        P[j] = np.where(a_signed < 0, GRID[g] + np.pi, GRID[g]) % (2 * np.pi)
        B[j] = c[g][0]
    return yh, dict(a=A, p=P, b=B)


def fit_market(Y, TH, F, fit_end, device, seed=7, steps=8000, lr=2e-2):
    """SUPERSEDED (2026-07-28) — the v10 GRADIENT model, kept only so final_decide.py can re-measure it.
    Fifteen params/topic against fit_final's nine, seed-dependent, and worse over twelve origins
    (+0.8533 vs +0.8751). main() does not call this.

    INDEPENDENT per-topic RECTIFIED SQUARE-LAW receivers (operator 2026-07-24; final exhaustive
    ablation 2026-07-24 — cosine-sum beats the |·|² envelope by +0.030 12yr AUC and +0.18 skill,
    robust across seeds/rosters). NON-cross-sectional — NO pie, NO tides. Each topic j sums the eight
    planets' projections onto its own tuning axis, C_j(t) = b_j + Σᵢ a_jᵢ·cos(θᵢ(t)−p_jᵢ), then reads
    the RECTIFIED power, ŷ_j(t) = max(C_j,0)² — a diode/square-law detector: dark when the summed
    projection is negative, bright on a transit. Fit over ONLY each topic's consistently-non-zero
    suffix (leading pre-existence years masked out), with N^0.75 year-evidence weights (rehearsal-
    selected — see below). 15 params/topic (7 arrows + 7 tunings + 1 anchor). Returns (ŷ, params a/p/b)."""
    import torch as T
    T.manual_seed(seed)
    Tn = Y.shape[0]; nb = TH.shape[1]
    inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
    Ysq = np.sqrt(Y)
    cT = T.tensor((F * np.cos(TH)).astype(np.float32).T, device=device)
    sT = T.tensor((F * np.sin(TH)).astype(np.float32).T, device=device)
    tv = META["topic_valid"][:, :fit_end].astype(np.float32)          # per-topic valid mask (train)
    # N^¾ year evidence (30-yr improve pass, 2026-07-24): √N is the pure multinomial noise weight, but the
    # honest inner-rehearsal wall (fit ≤1965, score 1966-1995 — no test contact) selects a heavier lean onto
    # the high-evidence modern years, plateau e∈[0.70,0.80]; N^¾ is the mid-plateau simple fraction. 30-yr
    # AUC +0.637→+0.691, median skill +0.40→+0.49.
    wy = np.clip(META["evidence"][:fit_end], 0, None) ** 0.75
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)   # per-topic weights, normalised
    Wt = T.tensor(W.astype(np.float32), device=device)
    # HORIZON ANCHOR (agent competition winner, 2026-07-25): the champion's failure mode was never the
    # in-sample fit, it was EXTRAPOLATION — the slow bodies sweep only a fraction of a cycle inside the
    # window, so they act as an unpinned trend that drifts over a 30-year forecast. Penalise the model's
    # OWN predicted trajectory across the deployment window for straying (relatively) from the level the
    # field actually held in its last ANCHOR_K training years. Reads only planet angles (known centuries
    # ahead) + train-only data, so it is a pure empirical-risk match to how the model is deployed.
    Wa = np.zeros_like(W)
    Wa[:, fit_end - ANCHOR_K:] = (tv * wy[None])[:, fit_end - ANCHOR_K:]
    bad = Wa.sum(1) <= 0                                   # no valid year in the window -> whole history
    Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    m_anchor = T.tensor(((Ysq[:, :fit_end] * Wa).sum(1))[:, None].astype(np.float32), device=device)
    # TWELVE-SIGN KL TERM (operator 2026-07-25): the phase is a continuous angle whose JOB is to place
    # the field in one of twelve seasons, so train it to do that job — minimise each field's own sign
    # entropy (a decisive placement) while maximising the atlas-wide spread (all twelve seasons used).
    # Those two together are exactly the mean KL divergence of each field's sign distribution from the
    # atlas marginal, i.e. the mutual information between field and sign. Touches only the objective:
    # the prediction stays a pure function of the sky.
    CENTRES = T.tensor(np.deg2rad(np.arange(12) * 30.0 + 15.0).astype(np.float32), device=device)
    PL = BODIES.index("pluto")
    hz_hi = min(fit_end + HORIZON, TH.shape[0])            # the 30 years the model is asked to forecast
    vmean = (Ysq[:, :fit_end] * tv).sum(1) / np.maximum(tv.sum(1), 1.0)   # per-topic valid-train mean
    A0 = np.full((Tn, nb), -2.0, np.float32); A0[:, BODIES.index("pluto")] = inv_sp(vmean)
    Araw = T.tensor(A0, device=device, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (Tn, nb, 1)).astype(np.float32) +
                 np.random.RandomState(seed).randn(Tn, nb, 2).astype(np.float32) * 0.01,
                 device=device, requires_grad=True)
    Bp = T.tensor(vmean.astype(np.float32), device=device, requires_grad=True)
    params = [Araw, U, Bp]
    Yt = T.tensor(Ysq.astype(np.float32), device=device)
    opt = T.optim.Adam(params, lr=lr)

    def kl_phase():
        """−[H(q̄) − mean_j H(q_j)] over a soft twelve-sign assignment of the classifying phase."""
        pp = T.atan2(U[:, PL, 0], U[:, PL, 1])
        q = T.softmax(T.cos(pp[:, None] - CENTRES[None, :]) / TAU_KL, dim=1)
        Hj = -(q * T.log(q.clamp_min(1e-12))).sum(1).mean()        # per-field uncertainty   ↓
        qb = q.mean(0)
        Hb = -(qb * T.log(qb.clamp_min(1e-12))).sum()              # spread over the seasons ↑
        return -(Hb - Hj)

    def forward():
        p = T.atan2(U[:, :, 0], U[:, :, 1]); A = T.nn.functional.softplus(Araw)
        cp = A * T.cos(p); sp = A * T.sin(p)
        C = Bp[:, None] + cp @ cT + sp @ sT               # real projection onto the topic's tuning axis
        return T.clamp(C, min=1e-4) ** 2 + 1e-8           # rectified square-law detector — cosine-sum share

    best, stall, state = np.inf, 0, None
    for it in range(steps):
        sig = T.sqrt(forward() + 1e-8)                   # √ŷ over data + forecast years
        per = ((sig[:, :fit_end] - Yt[:, :fit_end]).abs() * Wt).sum(1)      # data term (masked, N^¾ L1)
        d = (sig[:, fit_end:hz_hi] - m_anchor) / T.clamp(m_anchor, min=1e-3)
        loss = (per + LAM_HORIZON * (d ** 2).mean(1)).sum() / Tn            # + scale-free horizon anchor
        loss = loss + BETA_KL * kl_phase()                                  # + twelve-sign KL term
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            l = loss.item()
            if l < best - 1e-7: best, stall, state = l, 0, [x.detach().clone() for x in params]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip(params, state): x.copy_(sv)
        yh = forward().cpu().numpy()
        out = dict(p=T.atan2(U[:, :, 0], U[:, :, 1]).cpu().numpy(),
                   a=T.nn.functional.softplus(Araw).cpu().numpy(), b=Bp.cpu().numpy())
    return np.clip(yh, 0, None), out


def main():
    names, Y, labels, future = load_lunar()
    Tn, n = Y.shape
    dates_ext = labels + future
    TH, R = sky_lunar(dates_ext)
    ne = len(dates_ext); EXT = ne - n
    # NO distance factor: over 300+ years angles-only beats 1/r and 1/r² — the distance breathing of a
    # slow body is a short-horizon feature and it does not survive a thirty-year wall.
    DEP = n - 1
    nb = TH.shape[1]
    print(f"  CITATIONS YEARLY · DETERMINISTIC rectified receiver, ONE shared tuning + {nb} signed arrows "
          f"({1+nb+1} params/topic · N^0.75 · per-topic non-zero crop · 1° sweep + closed form) · "
          f"{Tn} fields x {n} years ({labels[0]}..{labels[-1]}) · ext {EXT} · deploy<{DEP}", flush=True)
    Yd, prm = fit_final(Y, TH, DEP)
    ev = META["evidence"]; TV = META["topic_valid"]
    def r2rows(fe, Yh):
        """Per-topic evidence-weighted R² over each topic's VALID (non-zero-suffix) years only."""
        W = TV[:, :fe].astype(float) * (np.clip(ev[:fe], 0, None) ** 0.75)[None]
        W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
        Yv = Y[:, :fe]
        mu = (Yv * W).sum(1, keepdims=True)
        tot2 = np.maximum((((Yv - mu) ** 2) * W).sum(1), 1e-9)
        return 1.0 - (((Yv - Yh[:, :fe]) ** 2) * W).sum(1) / tot2
    r2 = r2rows(DEP, Yd)
    print(f"  DEPLOY: fit R² {r2.mean():+.4f}/{np.median(r2):+.4f}", flush=True)

    def benchmark(wall):
        Yw, _ = fit_final(Y, TH, wall)
        tvw = TV[:, :wall].astype(float)                              # per-topic valid TRAIN years
        mu = (Y[:, :wall] * tvw).sum(1, keepdims=True) / np.maximum(tvw.sum(1, keepdims=True), 1.0)
        den = np.maximum(((Y[:, wall:n] - mu) ** 2).sum(1), 1e-6)     # held-out is valid for all topics
        skill = 1.0 - ((Y[:, wall:n] - Yw[:, wall:n]) ** 2).sum(1) / den
        curve = [round(float(1.0 - ((Y[:, wall + h] - Yw[:, wall + h]) ** 2).sum() /
                       max(((Y[:, wall + h] - mu[:, 0]) ** 2).sum(), 1e-9)), 4) for h in range(n - wall)]
        return Yw, skill, curve

    def persistence_auc(wall):
        """THE HARDER BAR — carry each topic's LAST OBSERVED year forward, unchanged, for the whole
        horizon. Beating a topic's long-run mean is a low hurdle; this costless rule is the strongest
        trivial forecast there is, and any model that cannot clear it is not a model. Measured here,
        never quoted from memory, so the page always shows the current number."""
        tvw = TV[:, :wall].astype(float)
        mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
        last = Y[:, wall - 1]
        return float(np.mean([1.0 - ((Y[:, wall + h] - last) ** 2).sum() /
                              max(((Y[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(n - wall)]))

    W48 = n - 30                                           # the HEADLINE wall (operator 2026-07-24): THIRTY
    Yw, skill, curve = benchmark(W48)                       # unseen years — fit ≤1995, forecast 1996-2025
    pers30 = persistence_auc(W48)
    print(f"  BENCH 30yr(30): skill {np.median(skill):+.4f} ({(skill > 0).mean()*100:.1f}%>0) · "
          f"AUC {np.mean(curve):+.4f}   [carry-forward bar {pers30:+.4f}]", flush=True)
    _, sk2, cur2 = benchmark(n - 4)                        # 4-year context chip
    print(f"  BENCH 4yr(4):   skill {np.median(sk2):+.4f} ({(sk2 > 0).mean()*100:.1f}%>0) · "
          f"AUC {np.mean(cur2):+.4f}", flush=True)
    _, sk4, cur4 = benchmark(n - 12)                       # 12-year context chip
    print(f"  BENCH 12yr(12): skill {np.median(sk4):+.4f} ({(sk4 > 0).mean()*100:.1f}%>0) · "
          f"AUC {np.mean(cur4):+.4f}", flush=True)

    P = prm["p"]; A = prm["a"]; B = prm["b"]
    pl = np.round(np.rad2deg(P[:, BODIES.index("pluto")]) % 360, 6) % 360   # canonical pluto tuning
    hist = np.bincount((pl // 30).astype(int), minlength=12)
    print("  season occupancy: " + " ".join(f"{p2.SIGNS[i][:3]} {hist[i]}" for i in range(12)), flush=True)
    # WHICH WAY IS "FORWARD" IS ARBITRARY: flipping φ by 180° and every sign with it is the SAME model,
    # so "fraction opposing φ" is not a well-defined quantity. The gauge-invariant statement is how many
    # arrows sit in the smaller of the two camps, and how many topics use both camps at all. (An earlier
    # version measured every body against PLUTO's tuning and counted pluto against itself, which both
    # picked a gauge and diluted the figure by 1/7.)
    dg = np.round(np.degrees(P) % 360).astype(int)
    both = int(sum(len(set(r)) > 1 for r in dg))
    minor = int(sum(min(np.bincount(np.unique(r, return_inverse=True)[1])) for r in dg if len(set(r)) > 1))
    print(f"  the binary second phase: {both}/{Tn} topics ({both/Tn*100:.0f}%) use BOTH directions; "
          f"{minor}/{dg.size} arrows ({minor/dg.size*100:.1f}%) sit in the minority direction", flush=True)
    from collections import Counter
    dom = Counter(BODIES[int(k)] for k in A.argmax(1))
    print("  dominant bodies:", dict(dom.most_common(6)), flush=True)
    trm = r2rows(W48, Yw)
    np.savez_compressed("analysis/arxivtopics/arxiv_phasor_final.npz",
                        a=A.astype(np.float32), p=P.astype(np.float64), b=B.astype(np.float32),
                        # p stays FLOAT64: a tuning is an exact integer degree off the sweep, and the
                        # season is floor(deg/30) — a discrete label. Two topics tune to exactly 30° and
                        # 240°, dead on a season boundary, and the float32 round-trip moved both across
                        # it. A label a reader is shown must never be decided by floating-point dust.
                        yhat_dep=Yd.astype(np.float32), yhat_w=Yw.astype(np.float32),
                        r2=r2.astype(np.float32), r2_ins=trm.astype(np.float32),
                        r2_oos_skill=skill.astype(np.float32),
                        skill_curve=np.array(curve, np.float32),
                        skill2=float(np.median(sk2)), auc2=float(np.mean(cur2)),
                        skill4=float(np.median(sk4)), auc4=float(np.mean(cur4)),
                        persistence=float(pers30), params_per_topic=int(1 + len(BODIES) + 1),
                        wall=W48, dep_wall=DEP, ext=EXT,
                        loss="deterministic-oneshared-tuning-signed-arrows-rectified-wls-anchored-N075w",
                        cadence="yearly",
                        dates=np.array(dates_ext), bodies=np.array(BODIES), names=np.array(names))
    print("  saved -> analysis/arxivtopics/arxiv_phasor_final.npz", flush=True)
    print("ARXIVFIT DONE", flush=True)


if __name__ == "__main__":
    main()
