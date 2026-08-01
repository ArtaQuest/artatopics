#!/usr/bin/env python3
"""COMPETITION HARNESS (operator 2026-07-25: "competition of multiple agents … max out 30-yr AUC").

THE ONE SOURCE OF DATA, TARGET, WALLS AND SCORING for every competing model. Import it; never
re-derive the target or the split. Data = the PUBLISHED yearly rail behind notebook nb/9303
("Citations per year across the scholarly record, 1000-2026", DOI zenodo.21537062, OpenAlex CC0),
materialised to analysis/citations/rail_*_yearly.csv — verified cellwise identical to the deployed
matrix on citations_received.

    from comp_harness import *            # Y, TV, N, WORKS, CITED, TH_ALL, R_ALL, YEARS, ...
    yhat = my_model(wall=WALL_INNER)      # (251, n) predicted SHARE, level scale
    inner = evaluate(yhat, WALL_INNER)    # ← select EVERYTHING here
    yhat = my_model(wall=WALL_OUTER)
    outer = evaluate(yhat, WALL_OUTER)    # ← report ONCE, never select on it

TARGET (fixed, = the deployed definition): Y[j,t] = 100 · citations_received[j,t] / Σ_j citations_received[·,t]
  — topic j's share of the year's citations, over the fixed 251-subfield universe, 1700..2025
  (2026 is partial at the snapshot and dropped). Do NOT change the target; the competition compares
  MODELS, not scorekeeping.

PER-TOPIC MASK TV[j,t]: True only from the year topic j becomes CONSISTENTLY non-zero onward. A
topic's pre-existence years are never trained on and never scored. Emerging fields are fit from
emergence; no topic is dropped.

THE TWO WALLS (both refit from scratch behind the wall — no state may cross):
  WALL_OUTER = n-30  → train ≤1995, score 1996..2025   THE HEADLINE (30-yr AUC). Report once.
  WALL_INNER = n-60  → train ≤1965, score 1966..1995   ALL model/hyper-parameter selection.
Anything chosen by looking at outer-wall scores is CHEATING and will be caught by the verifiers:
architecture, roster, exponents, losses, optimizer settings, seeds, ensembling weights — all must be
selected on WALL_INNER (or by pure theory), then applied unchanged to WALL_OUTER.

SCORING (identical for both walls): baseline = each topic's mean over its VALID TRAIN years.
  skill_j = 1 − Σ_h (y_j − ŷ_j)² / Σ_h (y_j − mean_j)²      over the held-out span
  auc     = mean over horizons h of the POOLED skill at h   ← the headline number
evaluate() returns dict(auc, skill, pct, n_topics).

FEATURES available (all (251,326), aligned to YEARS): CITES (raw counts), WORKS (papers published),
CITED (cited_by_sum). N = per-year total citations = the evidence behind each year's share. The sky:
TH_ALL/R_ALL (radians / AU) for BODIES_ALL at each year's mid-point, sidereal Lahiri, 1700..2055 —
index rows by YEAR_IDX[year]; columns past 2025 are the pure-ephemeris forecast region.

CHAMPION to beat — champion_fit(): the deployed v8 model. Independent per-topic rectified square-law
receiver, ŷ = max(b + Σᵢ aᵢcos(θᵢ−pᵢ), 0)², 7 bodies (mars,jupiter,saturn,uranus,neptune,pluto,node),
L1 on √share, N^0.75 year weights, angles only, 15 params/topic:
    OUTER 30-yr AUC +0.6918 (median skill +0.479, 63.3%>0)   INNER AUC ≈ +0.706
Beat that on the OUTER wall — having chosen everything on the INNER wall.
"""
import os
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_D = os.path.join(REPO, "analysis", "citations")

_c = pd.read_csv(os.path.join(_D, "rail_citations_received_yearly.csv"))
_w = pd.read_csv(os.path.join(_D, "rail_works_yearly.csv"))
_b = pd.read_csv(os.path.join(_D, "rail_cited_by_sum_yearly.csv"))
YEARS = [int(x) for x in _c.columns if x[0].isdigit()]
YEAR_IDX = {y: i for i, y in enumerate(YEARS)}
_yc = [str(y) for y in YEARS]
NAMES = list(_c.subfield)
FIELD = list(_c.field)
DOMAIN = list(_c.domain)
SUBFIELD_ID = list(_c.subfield_id)

CITES = _c[_yc].to_numpy(float)
WORKS = _w[_yc].to_numpy(float)
CITED = _b[_yc].to_numpy(float)
N = CITES.sum(0)                                    # per-year total citations = the share's evidence
Y = 100.0 * CITES / np.maximum(N[None, :], 1.0)     # THE TARGET: % share of the year's citations
Tn, n = Y.shape

_z = CITES > 0                                      # per-topic consistently-non-zero suffix
TV = np.ones_like(_z, bool)
for _i in range(n - 2, -1, -1):
    TV[:, _i] = _z[:, _i] & TV[:, _i + 1]


def train_mask(wall):
    """LEAK-FREE per-topic mask: the consistently-non-zero suffix judged using ONLY years < wall.
    (TV above looks at the whole series, which would let "does this topic survive to 2025?" reach the
    training window. Audited 2026-07-25: 0 of 251 topics differ in practice — no topic has a zero year
    after either wall — but the structural leak is removed so the harness is correct by construction.)"""
    m = np.ones((Tn, wall), bool)
    for i in range(wall - 2, -1, -1):
        m[:, i] = _z[:, i] & m[:, i + 1]
    return m

WALL_OUTER = n - 30                                 # train ≤1995 → score 1996..2025 (HEADLINE)
WALL_INNER = n - 60                                 # train ≤1965 → score 1966..1995 (SELECTION)

BODIES_ALL = ["sun", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "node", "chiron"]
CHAMPION_BODIES = ["mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "node"]
_E = pd.read_csv(os.path.join(REPO, "analysis", "arxivtopics", "_ephemeris_yearly.csv")).set_index("Time")
_E.index = _E.index.astype(str)
_EY = [str(y) for y in range(YEARS[0], 2056)]
TH_ALL = np.stack([np.deg2rad(_E[f"{b}_lon"].loc[_EY].to_numpy(float)) for b in BODIES_ALL], 1)
R_ALL = np.stack([_E[f"{b}_dist"].loc[_EY].to_numpy(float) for b in BODIES_ALL], 1)
SKY_YEARS = [int(y) for y in _EY]


HORIZON = 30                                        # both walls score exactly 30 years ahead


def evaluate(yhat, wall, horizon=HORIZON):
    """Score predictions over the `horizon` years after `wall`. yhat: (251, ≥n) LEVEL-scale share.
    The INNER wall is a true rehearsal of the OUTER one: same 30-year horizon, same baseline, same
    metric — so a choice that wins at inner is a choice that wins a 30-year forecast, and 1996+ is
    never touched during selection."""
    yhat = np.asarray(yhat, float)[:, :n]
    assert yhat.shape == (Tn, n), f"expected {(Tn, n)}, got {yhat.shape}"
    assert np.isfinite(yhat).all(), "non-finite predictions"
    hi = min(wall + horizon, n)
    tvw = TV[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1, keepdims=True) / np.maximum(tvw.sum(1, keepdims=True), 1.0)
    den = np.maximum(((Y[:, wall:hi] - mu) ** 2).sum(1), 1e-6)
    skill = 1.0 - ((Y[:, wall:hi] - yhat[:, wall:hi]) ** 2).sum(1) / den
    curve = [1.0 - ((Y[:, wall + h] - yhat[:, wall + h]) ** 2).sum() /
             max(((Y[:, wall + h] - mu[:, 0]) ** 2).sum(), 1e-9) for h in range(hi - wall)]
    return {"auc": round(float(np.mean(curve)), 4), "skill": round(float(np.median(skill)), 4),
            "pct": round(float((skill > 0).mean() * 100), 1), "n_topics": int(Tn)}


def champion_fit(wall=WALL_OUTER, bodies=None, kpow=2.0, wexp=0.75, seed=7, steps=9000, lr=2e-2,
                 device=None):
    """The deployed v8 model — reference implementation and starting point. Copy and modify freely."""
    import torch as T
    dev = device or ("mps" if T.backends.mps.is_available() else "cpu")
    bods = bodies or CHAMPION_BODIES
    bi = [BODIES_ALL.index(b) for b in bods]; nb = len(bi)
    TH = TH_ALL[:n, bi]
    inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
    Ysq = np.sqrt(Y)
    T.manual_seed(seed)
    tb = lambda a: T.tensor(np.asarray(a, np.float32), device=dev)
    cT, sT = tb(np.cos(TH).T), tb(np.sin(TH).T)
    tv = TV[:, :wall].astype(np.float32)
    wy = np.clip(N[:wall], 0, None) ** wexp
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9); Wt = tb(W)
    vmean = (Ysq[:, :wall] * tv).sum(1) / np.maximum(tv.sum(1), 1.0)
    A0 = np.full((Tn, nb), -2.0, np.float32)
    if "pluto" in bods: A0[:, bods.index("pluto")] = inv_sp(np.clip(vmean, 1e-3, None))
    Araw = T.tensor(A0, device=dev, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (Tn, nb, 1)).astype(np.float32) +
                 np.random.RandomState(seed).randn(Tn, nb, 2).astype(np.float32) * 0.01,
                 device=dev, requires_grad=True)
    Bp = T.tensor(vmean.astype(np.float32), device=dev, requires_grad=True)
    Yt = tb(Ysq)
    opt = T.optim.Adam([Araw, U, Bp], lr=lr)

    def fwd():
        p = T.atan2(U[:, :, 0], U[:, :, 1]); A = T.nn.functional.softplus(Araw)
        C = Bp[:, None] + (A * T.cos(p)) @ cT + (A * T.sin(p)) @ sT
        return T.clamp(C, min=1e-4) ** kpow + 1e-8

    best, stall, state = np.inf, 0, None
    for it in range(steps):
        e = T.sqrt(fwd() + 1e-8)[:, :wall] - Yt[:, :wall]
        loss = (e.abs() * Wt).sum() / Tn
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            lv = loss.item()
            if lv < best - 1e-7: best, stall, state = lv, 0, [x.detach().clone() for x in (Araw, U, Bp)]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip((Araw, U, Bp), state): x.copy_(sv)
        return np.clip(fwd().cpu().numpy(), 0, None)


if __name__ == "__main__":
    print(f"data {Tn}×{n} years {YEARS[0]}..{YEARS[-1]} · walls inner {WALL_INNER}({YEARS[WALL_INNER]}) "
          f"outer {WALL_OUTER}({YEARS[WALL_OUTER]})")
    print("champion INNER:", evaluate(champion_fit(WALL_INNER), WALL_INNER))
    print("champion OUTER:", evaluate(champion_fit(WALL_OUTER), WALL_OUTER))


# ═════════════════════════════════════════════════════════════════════════════════════════════
# ROUND 2 (operator 2026-07-25). Each model estimates, for every topic, a PHASE that is a CONTINUOUS
# ANGLE in degrees; that angle's job is to place the topic in one of the TWELVE signs/seasons
# (sign = ⌊φ/30⌋, as the live atlas does). Two numbers are scored, both in [0, 1]:
#
#   AUC   — the honest 30-year forecast skill (evaluate(), unchanged)
#   PHASE — sign_mutual_information()['nmi']: per-topic phase SHARP (low per-topic entropy) AND the
#           atlas SPREAD across the twelve signs (high marginal entropy). Both at once, one number.
#           Reported beside it: phase_confidence() — the peak-to-mean height of the tuning curve,
#           rotated DEGREE BY DEGREE — which is the per-topic sharpness on its own.
#   GLOBAL = F1(AUC, PHASE) — harmonic mean; same scale, so neither half wins on units alone.
#
# ── HOW THE TUNING CURVE IS MEASURED (parameter-free, model-agnostic) ─────────────────────────
# Rotate the WHOLE SKY by δ, one degree at a time, and watch the topic's own training loss L_j(δ).
# Rotating the sky by +δ is exactly rotating every tuning by −δ, so the curve is a likelihood over
# what the topic's phase could have been. Convert it to a tuning RESPONSE at the topic's own
# residual scale (dimensionless — there is no free constant anywhere in this metric):
#
#   w_j(δ) = exp( −(L_j(δ) − L_j^min) / T_j )             T_j = noise_scale(wall): the topic's intrinsic
#                                                        year-to-year noise, MODEL-INDEPENDENT and shared
#                                                        by every competitor (see noise_scale docstring)
#   peak_j = w_j(0)                                       the height AT THE MODEL'S DECLARED PHASE
#   CONF_j = 1 − mean_δ w_j(δ) / peak_j        ∈ [0,1]    peak-to-mean, scaled
#
#   flat curve (phase says nothing)      → mean = peak → CONF = 0
#   needle-sharp peak at the estimate    → mean ≪ peak → CONF → 1
#   peak sits somewhere ELSE than the declared phase → peak_j < 1 → CONF falls automatically, so a
#     model cannot report one angle while its loss prefers another.
# Equivalent reading: peak/mean = 1/(1−CONF) is "how many times taller the tuning peak is than the
# average response". Because it is a plain mean over rotations it converges as the grid refines —
# unlike an entropy over bins, so the grid is NOT a tuning knob (verified 1°..15°).
#
# ALSO REPORTED (diagnostics, not scored): sign_information() — the Shannon information the same
# curve carries about the TWELVE-WAY decision, since twelve-step discrimination is what the atlas
# needs; sign_diversity(); phase_stability(); sign_agreement_across_seeds().
# ═════════════════════════════════════════════════════════════════════════════════════════════

PHASE_GRID = np.arange(-180.0, 180.0, 1.0)      # DEGREE BY DEGREE (operator 2026-07-25)
NSIGN = 12
SIGN_NAMES = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
              "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]


_NOISE = {}


def noise_scale(wall, wexp=0.75):
    """THE TEMPERATURE OF THE PHASE POSTERIOR — the intrinsic year-to-year NOISE of each topic, in the
    same units as the training loss. IDENTICAL FOR EVERY COMPETITOR, and computed from the data alone.

    WHY THIS MATTERS (audited 2026-07-25): the earlier draft used each model's OWN achieved loss as the
    temperature. That hands free sharpness to whoever overfits — dropping the loss floor while leaving
    the well depth untouched inflated the score from 0.4406 to 0.7048 on the real model. In a round that
    invites more parameters that hole would decide the competition. The width of a phase posterior must
    be set by the NOISE IN THE DATA, not by how hard a model was pushed onto the training set.

    Estimate: the weighted mean absolute YEAR-OVER-YEAR step in √share over the topic's valid training
    years — a model-free scale for "how much this series moves anyway", in the loss's own units."""
    key = (wall, wexp)
    if key in _NOISE: return _NOISE[key]
    tv = train_mask(wall)
    both = tv[:, 1:] & tv[:, :-1]
    step = np.abs(np.sqrt(Y[:, 1:wall]) - np.sqrt(Y[:, :wall - 1]))
    wy = np.clip(N[1:wall], 0, None) ** wexp
    W = both * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    t = (step * W).sum(1)
    t = np.where(t > 0, t, np.median(t[t > 0]) if (t > 0).any() else 1e-3)
    _NOISE[key] = t
    return t


def train_loss_rows(yhat, wall, wexp=0.75):
    """Per-topic training loss — the same objective the models are fitted with."""
    yhat = np.asarray(yhat, float)[:, :wall]
    tv = TV[:, :wall].astype(float)
    wy = np.clip(N[:wall], 0, None) ** wexp
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    return (np.abs(np.sqrt(np.clip(yhat, 0, None)) - np.sqrt(Y[:, :wall])) * W).sum(1)


def phase_curve(predict_delta, wall, grid=PHASE_GRID):
    """L[j, i] = topic j's training loss with the whole sky rotated by grid[i] degrees."""
    L = np.zeros((Tn, len(grid)))
    for i, d in enumerate(grid):
        L[:, i] = train_loss_rows(predict_delta(float(d)), wall)
    return L


def _weights(L, wall):
    """Boltzmann weights over rotations at the COMMON, model-independent noise temperature."""
    L = np.asarray(L, float)
    T = noise_scale(wall)[:, None]
    return np.exp(-(L - L.min(1, keepdims=True)) / np.maximum(T, 1e-12))


def phase_confidence(L, wall, grid=PHASE_GRID):
    """Peak-to-mean sharpness of the tuning curve, in [0,1] (reported beside the scored NMI).
    Peak is taken AT THE MODEL'S DECLARED PHASE (δ=0), so a mis-centred estimate is self-penalising."""
    w = _weights(L, wall)                                           # ∈ (0,1], = 1 at the best phase
    i0 = int(np.argmin(np.abs(grid)))                                # the δ=0 column
    peak = w[:, i0]
    conf = np.clip(1.0 - w.mean(1) / np.maximum(peak, 1e-12), 0.0, 1.0)
    return {"conf": round(float(np.median(conf)), 4),
            "conf_mean": round(float(np.mean(conf)), 4),
            "peak_over_mean_median": round(float(np.median(1.0 / np.maximum(1.0 - conf, 1e-9))), 1),
            "peak_at_declared_median": round(float(np.median(peak)), 4),
            "per_topic": conf}


def sign_posterior(L, phase_deg, wall, grid=PHASE_GRID):
    """P[j, s] — the mass topic j's tuning curve puts on each of the twelve signs."""
    phase_deg = np.asarray(phase_deg, float)
    w = _weights(L, wall)
    phi = (phase_deg[:, None] - grid[None, :]) % 360.0        # phase implied by each rotation
    idx = np.floor(phi / 30.0).astype(int) % NSIGN
    P = np.zeros((L.shape[0], NSIGN))
    np.add.at(P, (np.arange(L.shape[0])[:, None], idx), w)
    return P / np.maximum(P.sum(1, keepdims=True), 1e-300)


def sign_information(L, phase_deg, wall, grid=PHASE_GRID):
    """THE SCORED PHASE METRIC: Shannon information of the twelve-sign classification, in [0,1]."""
    P = sign_posterior(L, phase_deg, wall, grid)
    H = -(P * np.log2(np.maximum(P, 1e-300))).sum(1)
    info = 1.0 - H / np.log2(NSIGN)
    declared = (np.asarray(phase_deg, float) // 30).astype(int) % NSIGN
    agree = float((P.argmax(1) == declared).mean())
    return {"info": round(float(np.median(info)), 4),
            "info_mean": round(float(np.mean(info)), 4),
            "entropy_bits_median": round(float(np.median(H)), 3),
            "sign_agreement": round(agree, 4),
            "top_sign_mass_median": round(float(np.median(P.max(1))), 4),
            "per_topic": info}


def sign_mutual_information(L, phase_deg, wall, grid=PHASE_GRID):
    """THE SCORED PHASE QUANTITY (operator 2026-07-25): "minimise the per-topic info (confidence)
    while maximising across topics (diversity)" — which is exactly the MUTUAL INFORMATION between
    topic and sign:

        NMI = [ H(P̄)  −  mean_j H(P_j) ] / log₂(12)  =  mean_j KL( P_j ‖ P̄ ) / log₂(12)   ∈ [0,1]

    THE KL IDENTITY MATTERS: mutual information IS the mean KL divergence of each topic's sign
    distribution from the atlas-wide marginal, so this metric is DIRECTLY TRAINABLE — add a
    differentiable −β·mean_j KL(P_j ‖ P̄) term to the loss and you are optimising the score itself
    (operator 2026-07-25: "so train KL divergence"). Do it honestly: β is a constant and must be
    chosen on WALL_INNER, and raising KL by wrecking the forecast is a losing trade under the F1.

                 ▲                ▲
                 │                └─ per-topic uncertainty — MINIMISED (each topic sharply in one sign)
                 └─ the spread of the atlas over the twelve signs — MAXIMISED

    Every degenerate strategy scores 0 automatically, so no external guard is needed:
      · every topic uniform over the signs (knows nothing)        → H(P̄)=log₂12, mean H=log₂12 → 0
      · every topic razor-sharp but ALL IN ONE SIGN (collapse)    → H(P̄)=0,       mean H=0     → 0
      · sharp AND evenly spread over the twelve signs             →                              → 1
    """
    P = sign_posterior(L, phase_deg, wall, grid)
    Pbar = P.mean(0)
    H_bar = float(-(Pbar * np.log2(np.maximum(Pbar, 1e-300))).sum())
    H_j = -(P * np.log2(np.maximum(P, 1e-300))).sum(1)
    nmi = (H_bar - float(H_j.mean())) / np.log2(NSIGN)
    return {"nmi": round(float(np.clip(nmi, 0.0, 1.0)), 4),
            "across_topic_bits": round(H_bar, 3),
            "per_topic_bits": round(float(H_j.mean()), 3),
            "max_bits": round(float(np.log2(NSIGN)), 3)}


def angle_information(L, wall, grid=PHASE_GRID):
    """COMPANION DIAGNOSTIC (not scored): information about the CONTINUOUS angle, independent of any
    binning. I = log₂(M) − H over the rotation grid = KL(p‖uniform) in bits (grid-independent for
    smooth curves); reported as 1 − 2^(−I) = the fraction of the circle the data has ruled out."""
    w = _weights(L, wall)
    p = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
    H = -(p * np.log2(np.maximum(p, 1e-300))).sum(1)
    I = np.maximum(np.log2(len(grid)) - H, 0.0)
    return {"angle_info": round(float(np.median(1.0 - np.power(2.0, -I))), 4),
            "angle_bits_median": round(float(np.median(I)), 3)}


def sign_diversity(phase_deg):
    """ANTI-COLLAPSE GUARD: normalised entropy of how the 251 topics spread over the twelve signs.
    1 = evenly spread, 0 = every topic in one sign (worthless — the atlas is sorted by this)."""
    s = (np.asarray(phase_deg, float) // 30).astype(int) % NSIGN
    c = np.bincount(s, minlength=NSIGN).astype(float); c /= max(c.sum(), 1e-12)
    H = -(c * np.log2(np.maximum(c, 1e-300))).sum()
    return round(float(H / np.log2(NSIGN)), 4)


def phase_stability(phase_sets):
    """Guard against a sharp-but-arbitrary phase: circular agreement of the per-topic phase across
    independent refits (different seeds). phase_sets: list of (251,) arrays in DEGREES."""
    P = np.deg2rad(np.asarray(phase_sets, float))
    z = np.exp(1j * P).mean(0)
    return round(float(np.median(np.abs(z))), 4)


def sign_agreement_across_seeds(phase_sets):
    """Fraction of topics whose SIGN is unchanged across refits — the classification's own stability."""
    S = [((np.asarray(p, float) // 30).astype(int) % NSIGN) for p in phase_sets]
    return round(float(np.mean([np.mean(S[0] == s) for s in S[1:]])) if len(S) > 1 else 1.0, 4)


def global_f1(auc, phase):
    """THE COMPETITION SCORE: harmonic mean of the 30-year AUC and the PHASE score, both in [0,1].
    The phase half is sign_mutual_information()['nmi'] — sharp per topic AND spread across topics."""
    a = float(np.clip(auc, 0.0, 1.0)); i = float(np.clip(phase, 0.0, 1.0))
    return round(0.0 if a + i <= 0 else 2 * a * i / (a + i), 4)


def champion_with_phase(wall=WALL_OUTER, bodies=None, kpow=2.0, wexp=0.75, seed=7, steps=9000,
                        lr=2e-2, lam_horizon=0.03, anchor_k=5, device=None):
    """REFERENCE IMPLEMENTATION of the round-2 contract (the deployed v9 model).

    Returns (yhat, predict_delta, phase_deg):
      yhat          (251, n_ext) level-scale share predictions
      predict_delta callable(δ_degrees) -> same shape, with EVERY planet longitude advanced by δ
                    (no refit — just the fitted parameters read against a rotated sky)
      phase_deg     (251,) the model's per-topic PHASE in degrees (here: the Pluto tuning, which is
                    what the live site uses to place each topic in a sidereal sign)
    Copy this shape; the competition scores phase_confidence(phase_curve(predict_delta, wall)).
    """
    import torch as T
    dev = device or ("mps" if T.backends.mps.is_available() else "cpu")
    bods = bodies or CHAMPION_BODIES
    bi = [BODIES_ALL.index(b) for b in bods]; nb = len(bi)
    TH = TH_ALL[:, bi]                                   # full 1700..2055 sky
    ne = TH.shape[0]
    inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
    Ysq = np.sqrt(Y)
    T.manual_seed(seed)
    tb = lambda a: T.tensor(np.asarray(a, np.float32), device=dev)
    tv = TV[:, :wall].astype(np.float32)
    wy = np.clip(N[:wall], 0, None) ** wexp
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9); Wt = tb(W)
    Wa = np.zeros_like(W); Wa[:, wall - anchor_k:] = (tv * wy[None])[:, wall - anchor_k:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    m_anchor = tb(((Ysq[:, :wall] * Wa).sum(1))[:, None])
    vmean = (Ysq[:, :wall] * tv).sum(1) / np.maximum(tv.sum(1), 1.0)
    A0 = np.full((Tn, nb), -2.0, np.float32)
    if "pluto" in bods: A0[:, bods.index("pluto")] = inv_sp(np.clip(vmean, 1e-3, None))
    Araw = T.tensor(A0, device=dev, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (Tn, nb, 1)).astype(np.float32) +
                 np.random.RandomState(seed).randn(Tn, nb, 2).astype(np.float32) * 0.01,
                 device=dev, requires_grad=True)
    Bp = T.tensor(vmean.astype(np.float32), device=dev, requires_grad=True)
    Yt = tb(Ysq); opt = T.optim.Adam([Araw, U, Bp], lr=lr)
    hz = min(wall + HORIZON, ne)
    cT, sT = tb(np.cos(TH).T), tb(np.sin(TH).T)

    def fwd(cTx, sTx):
        p = T.atan2(U[:, :, 0], U[:, :, 1]); A = T.nn.functional.softplus(Araw)
        C = Bp[:, None] + (A * T.cos(p)) @ cTx + (A * T.sin(p)) @ sTx
        return T.clamp(C, min=1e-4) ** kpow + 1e-8

    best, stall, state = np.inf, 0, None
    for it in range(steps):
        sig = T.sqrt(fwd(cT, sT) + 1e-8)
        per = ((sig[:, :wall] - Yt[:, :wall]).abs() * Wt).sum(1)
        d = (sig[:, wall:hz] - m_anchor) / T.clamp(m_anchor, min=1e-3)
        loss = (per + lam_horizon * (d ** 2).mean(1)).sum() / Tn
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            lv = loss.item()
            if lv < best - 1e-7: best, stall, state = lv, 0, [x.detach().clone() for x in (Araw, U, Bp)]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip((Araw, U, Bp), state): x.copy_(sv)
        yhat = np.clip(fwd(cT, sT).cpu().numpy(), 0, None)
        ph = np.rad2deg(T.atan2(U[:, :, 0], U[:, :, 1]).cpu().numpy()) % 360.0

        def predict_delta(deg):
            THx = TH + np.deg2rad(float(deg))
            with T.no_grad():
                return np.clip(fwd(tb(np.cos(THx).T), tb(np.sin(THx).T)).detach().cpu().numpy(), 0, None)

    phase_deg = ph[:, bods.index("pluto")] if "pluto" in bods else ph[:, 0]
    return yhat, predict_delta, phase_deg
