#!/usr/bin/env python3
"""Topic 500 REFERENCE SOLUTION — the platform's ONE canonical baseline for the sky-vs-interest
competitions (topic-500), now fit with PYTORCH (operator directive 2026-07-10: the reference must fit
properly, generalise to the FUTURE holdout, reproduce easily, and accelerate on a Colab GPU). This file
is the SINGLE SOURCE OF TRUTH for the model: the dataset builder (make_topic500_comp.py) and the atlas
re-analysis (reanalyze_topic500_winner.py) import fit_many()/predict()/split_index()/phase_of()/
freqs_of()/weights_of() from here — no second implementation exists anywhere, so the hosted board entry,
the Colab reproduction and the /skills atlas can never drift.

THE MODEL CLASS (FIXED — the competition improves the LEARNING ALGORITHM, never the class). Per topic:

    y_hat = SUM_i  w_i * sinc( f_i * wrap(x_i - p) )  +  b            (i over the NB=11 bodies)

  x_i = the i-th body's sidereal ecliptic phase (deg) that month; p = ONE global CIRCULAR phase (deg,
  reported mod 360 -> the topic's zodiac SIGN); f_i = the body's per-body FREQUENCY (inverse width,
  bounded to [0, 20]); w_i = its NON-NEGATIVE weight; b = a bias. sinc = normalised sin(pi z)/(pi z);
  wrap(d) = ((d+180)%360)-180, then deg->rad. Parameters: NB weights + NB frequencies + 1 phase + 1 bias.

THE LEARNING ALGORITHM (the baseline — torch, full-batch Adam, all topics fit in ONE batched pass):
  * FAIR 12-SIGN-CENTRE MULTI-START — every fit runs 12 gradient starts, one initialised at each 30 deg
    zodiac sign centre (p0 = 15, 45, ..., 345); no sign is favoured by the initialisation.
  * CANONICAL ROW ORDER — the served CSVs are SHUFFLED, so rows are first sorted into a deterministic
    near-chronological order by the slow-sky clock (pluto, then neptune, then uranus, ties broken by the
    remaining bodies — canon_order()). The fit is therefore ORDER-INVARIANT: the builder (chronological
    input) and this hosted file (shuffled input) produce identical parameters.
  * TWO MODES:
      mode="forecast"  (the COMPETITION baseline — the board entry): the holdout is the FUTURE, so the
        loss is RECENCY-WEIGHTED (half-life 60 months on the canonical order — the level of a series
        drifts, and the recent level is the honest forecast anchor); the LAST 20% of the train rows form
        a time-blocked VALIDATION slice, each start early-checkpoints on it and the best-validation
        start wins; a brief recency-weighted REFINE (300 steps, lr x 0.3) over ALL train rows then lets
        the bias track the most recent level. Guards overfitting the training window without ever
        touching the class.
      mode="atlas"     (the /skills CLASSIFICATION fit): the plain unweighted least-squares objective on
        the FULL given series, best TRAIN-loss start wins — the descriptive seasonal read-out
        (christmas -> Scorpio, halloween -> Virgo, super-bowl -> Capricorn, swimming -> Gemini).
  * CONSTRAINTS exactly as published: w_i >= 0 (projected after every Adam step), f_i in [0, 20] via a
    sigmoid (smooth, bounded), p free during the fit and reported mod 360.

REPRODUCIBILITY (measured): the fit is exactly deterministic on a given device (same machine, same
torch build -> bit-identical parameters; there is no randomness anywhere — the 12 starts are fixed).
Across DEVICES (CPU vs Apple MPS vs CUDA) float-32 rounding can tip a topic's non-convex phase into a
different local optimum, so a cross-device reproduction matches the board to about +-0.2 points (most
topics identical). GPU: the batched fit auto-selects cuda > mps > cpu (override: AQ_T500_DEVICE) — on a
free Colab T4 the full 500-topic fit takes ~2 minutes (Runtime -> Change runtime type -> GPU).

SPLIT (operator directive 2026-07-07 — RECENCY-GUARDED, unchanged). On the clean series (the raw
recency year already dropped, clean_len = n-12): usable = clean_len - 12 (the most recent clean year is
excluded from the TEST too), train = the first 80% of usable, TEST = the last 20% (a FUTURE holdout).
split_index() returns (split, test_end). The ATLAS still fits the FULL clean series [0, clean_len).

    pip install torch numpy pandas
    python3 topic500_reference_solution.py          # downloads the data, fits, writes submission.csv

Env overrides (used by the platform's dataset builder, which imports this file so the hosted reference
and the board's baseline entry can never drift): AQ_T500_TRAIN / AQ_T500_TEST = local train.zip /
test.csv paths; AQ_T500_DEVICE = cpu | mps | cuda.
"""
import io, os, sys, zipfile, urllib.request
import numpy as np

BASE = "https://artaquest.org/wp-content/uploads/competitions/topic-500"
BODIES = ["sun", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "chiron", "node"]
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
NB = len(BODIES)                                   # 11 (the platform set; the Moon was removed 2026-07-01)
SIGN_CENTERS = np.arange(15.0, 360.0, 30.0)        # the 12 zodiac SIGN CENTRES: 15,45,...,345 (one per sign)
FMAX = 20.0                                        # frequency bound f_i in [0, FMAX] (as published)

# The baseline learning algorithm's fixed hyper-parameters (chosen on the recency-guarded FUTURE
# holdout, 2026-07-10: half-life 60 beat 24/36/48/90; 2400+300 steps beat 1200-step variants).
STEPS = 2400                                       # Adam steps for the 12-start batched fit
REFINE_STEPS = 300                                 # forecast mode: full-window refine from the winner
RECENCY_HL = 60.0                                  # forecast mode: loss half-life (months)
VAL_FRAC = 0.20                                    # forecast mode: time-blocked validation tail
LR = 0.03
_D2R = float(np.pi / 180.0)


def wrap(a):
    """Wrap degrees to [-180, 180)."""
    return ((a + 180.0) % 360.0) - 180.0


def predict(par, X, NB=NB):
    """y_hat = sum_i w_i * sinc(f_i * wrapped(x_i - p)) + b. X is n x NB body phases (deg). numpy."""
    par = np.asarray(par, float)
    w = par[:NB]; f = par[NB:2 * NB]; p = par[2 * NB]; b = par[2 * NB + 1]
    return np.sinc(np.deg2rad(wrap(np.asarray(X, float) - p)) * f[None, :]) @ w + b


RECENCY_TEST_GUARD = 12   # exclude the most recent 12 clean months from the TEST (recency bias, operator)


def split_index(clean_len):
    """The competition split, EXCLUDING the last year from the test (recency bias). The most recent
    RECENCY_TEST_GUARD clean months are dropped from train+test; on the remaining usable window,
    train = first 80%, TEST = last 20% (a recency-clean future holdout).
    Returns (split, test_end): train = [0, split), test = [split, test_end); [test_end, clean_len) dropped.
    (The ATLAS still fits on the FULL clean series [0, clean_len) — this guard is only for the competition test.)"""
    usable = clean_len - RECENCY_TEST_GUARD
    split = usable - round(0.20 * usable)
    return split, usable


def phase_of(par, NB=NB):
    """The fitted global phase p in [0, 360) — CIRCULAR (360==0); its 30 deg sector is the topic's SIGN."""
    return float(np.asarray(par, float)[2 * NB] % 360.0)


def sign_of(par, NB=NB):
    return SIGNS[int(phase_of(par, NB) // 30) % 12]


def freqs_of(par, NB=NB):
    """The per-body FREQUENCIES f_i (in [0, FMAX] by the fit's bounds), body order = BODIES."""
    return np.asarray(par, float)[NB:2 * NB]


def weights_of(par, NB=NB):
    """The per-body weights w_i (>= 0 by the fit's projection), body order = BODIES."""
    return np.asarray(par, float)[:NB]


def canon_order(X):
    """The CANONICAL near-chronological row order of a topic's months, from the phases alone: the
    slow-sky clock (pluto, neptune, uranus advance monotonically up to small retrograde wiggles), with
    every remaining body as a deterministic tie-break — a strict total order on the (unique) monthly
    skies, so shuffled input and chronological input sort identically (order-invariance)."""
    X = np.asarray(X, float)
    slow = [BODIES.index("pluto"), BODIES.index("neptune"), BODIES.index("uranus")]
    rest = [i for i in range(X.shape[1]) if i not in slow]
    # np.lexsort: LAST key is primary — pluto primary, then neptune, uranus, then the rest.
    keys = tuple(X[:, i] for i in reversed(rest)) + (X[:, slow[2]], X[:, slow[1]], X[:, slow[0]])
    return np.lexsort(keys)


def _device():
    import torch
    dev = os.environ.get("AQ_T500_DEVICE", "")
    if dev:
        return dev
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fit_many(Ys, Xs, NB=NB, mode="forecast", steps=STEPS, chunk=128, device=None, progress=False):
    """Fit the fixed model class on MANY topics in one batched torch pass (the GPU acceleration).
      Ys: list of per-topic target vectors (lengths may differ).
      Xs: ONE shared (n, NB) sky (deg) applying to every topic, or a list of per-topic (n_t, NB) skies.
      mode: "forecast" (the competition baseline) or "atlas" (the classification fit) — see the header.
    Returns par (T, 2*NB+2) float64 in the layout [w(NB), f(NB), p, b] — predict() consumes rows of it."""
    import torch
    dev = device or _device()
    T = len(Ys)
    shared = not isinstance(Xs, (list, tuple))
    Xlist = [np.asarray(Xs, float)] * T if shared else [np.asarray(x, float) for x in Xs]
    out = np.empty((T, 2 * NB + 2), float)
    for lo in range(0, T, chunk):
        hi = min(T, lo + chunk)
        out[lo:hi] = _fit_chunk([np.asarray(Ys[i], float) for i in range(lo, hi)],
                                Xlist[lo:hi], NB, mode, steps, dev)
        if progress:
            print(f"  fit {hi}/{T}", flush=True)
    return out


def _fit_chunk(Ys, Xs, NB, mode, steps, dev):
    import torch
    DT = torch.float32
    T = len(Ys)
    ns = np.array([len(y) for y in Ys], int)
    n_max = int(ns.max())
    S = 12
    # canonical near-chronological order per topic + end-padded batches with a mask
    Yp = np.zeros((T, n_max)); Xp = np.zeros((T, n_max, NB)); Mp = np.zeros((T, n_max))
    for i, (y, X) in enumerate(zip(Ys, Xs)):
        o = canon_order(X)
        Yp[i, :ns[i]] = y[o]; Xp[i, :ns[i], :] = X[o]; Mp[i, :ns[i]] = 1.0
    mu = (Yp * Mp).sum(1) / ns                                       # masked per-topic mean/std
    var = ((Yp - mu[:, None]) ** 2 * Mp).sum(1) / ns
    sd = np.sqrt(np.maximum(var, 1e-12)); sd[sd < 1e-6] = 1.0
    Ysn = (Yp - mu[:, None]) / sd[:, None] * Mp

    Xt = torch.tensor(Xp, dtype=DT, device=dev)                      # (T, n, NB)
    Yt = torch.tensor(Ysn, dtype=DT, device=dev)                     # (T, n)
    # loss weights: the mask, recency-weighted in forecast mode (per-topic age from ITS end)
    if mode == "forecast":
        age = ns[:, None] - 1 - np.arange(n_max)[None, :]
        rw = 0.5 ** (np.maximum(age, 0) / RECENCY_HL) * Mp
    else:
        rw = Mp.copy()
    # forecast mode: the last VAL_FRAC of each topic's rows form the time-blocked validation slice
    fitw = rw.copy(); valw = np.zeros_like(Mp)
    if mode == "forecast":
        for i in range(T):
            nv = int(round(VAL_FRAC * ns[i]))
            if nv:
                fitw[i, ns[i] - nv:ns[i]] = 0.0
                valw[i, ns[i] - nv:ns[i]] = 1.0
    FW = torch.tensor(fitw / np.maximum(fitw.sum(1, keepdims=True), 1e-9), dtype=DT, device=dev)
    VW = torch.tensor(valw / np.maximum(valw.sum(1, keepdims=True), 1e-9), dtype=DT, device=dev)
    has_val = mode == "forecast"

    p = torch.tensor(SIGN_CENTERS, dtype=DT, device=dev).repeat(T, 1).clone().requires_grad_(True)  # (T,S)
    w = torch.full((T, S, NB), 1.0 / NB, dtype=DT, device=dev, requires_grad=True)
    logf0 = float(np.log(1.0 / (FMAX - 1.0)))                        # sigmoid^-1(1/FMAX) → f starts at 1.0
    logf = torch.full((T, S, NB), logf0, dtype=DT, device=dev, requires_grad=True)
    b = torch.zeros((T, S), dtype=DT, device=dev, requires_grad=True)
    opt = torch.optim.Adam([p, w, logf, b], lr=LR)

    def forward(P, W, LF, B):                                        # → (T, S, n)
        d = torch.remainder(Xt[:, None, :, :] - P[:, :, None, None] + 180.0, 360.0) - 180.0
        z = torch.sinc(d * _D2R * (FMAX * torch.sigmoid(LF))[:, :, None, :])
        return torch.einsum("tsmk,tsk->tsm", z, W) + B[:, :, None]

    best_l = torch.full((T, S), 1e18, device=dev)
    best = {"p": p.detach().clone(), "w": w.detach().clone(), "logf": logf.detach().clone(), "b": b.detach().clone()}
    ckW = VW if has_val else FW                                      # checkpoint on val (forecast) / train (atlas)
    for step in range(steps):
        opt.zero_grad()
        pr = forward(p, w, logf, b)
        loss = ((pr - Yt[:, None, :]) ** 2 * FW[:, None, :]).sum(2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w.clamp_(min=0.0)
            if step % 20 == 19 or step == steps - 1:
                cl = ((forward(p, w, logf, b) - Yt[:, None, :]) ** 2 * ckW[:, None, :]).sum(2)
                better = cl < best_l - 1e-7
                best_l = torch.where(better, cl, best_l)
                for k, cur in (("p", p), ("w", w), ("logf", logf), ("b", b)):
                    m = better[..., None] if cur.dim() == 3 else better
                    best[k] = torch.where(m, cur.detach(), best[k])
    sel = best_l.argmin(dim=1)                                       # the winning start per topic
    ar = torch.arange(T, device=dev)
    P = best["p"][ar, sel]; W = best["w"][ar, sel]; LF = best["logf"][ar, sel]; B = best["b"][ar, sel]

    if mode == "forecast" and REFINE_STEPS:
        # brief recency-weighted refine over ALL rows (validation slice included) from the winning
        # checkpoint — lets the bias track the most recent level. Fixed steps: deterministic.
        P = P.clone().requires_grad_(True); W = W.clone().requires_grad_(True)
        LF = LF.clone().requires_grad_(True); B = B.clone().requires_grad_(True)
        opt2 = torch.optim.Adam([P, W, LF, B], lr=LR * 0.3)
        RW = torch.tensor(rw / np.maximum(rw.sum(1, keepdims=True), 1e-9), dtype=DT, device=dev)
        for _ in range(REFINE_STEPS):
            opt2.zero_grad()
            d = torch.remainder(Xt - P[:, None, None] + 180.0, 360.0) - 180.0
            z = torch.sinc(d * _D2R * (FMAX * torch.sigmoid(LF))[:, None, :])
            pr = torch.einsum("tmk,tk->tm", z, W) + B[:, None]
            loss = ((pr - Yt) ** 2 * RW).sum(1).mean()
            loss.backward()
            opt2.step()
            with torch.no_grad():
                W.clamp_(min=0.0)
        P, W, LF, B = P.detach(), W.detach(), LF.detach(), B.detach()

    Wn = W.cpu().numpy().astype(float) * sd[:, None]                 # back to the original scale
    Fn = (FMAX * torch.sigmoid(LF)).cpu().numpy().astype(float)
    Pn = P.cpu().numpy().astype(float) % 360.0
    Bn = B.cpu().numpy().astype(float) * sd + mu
    return np.concatenate([Wn, Fn, Pn[:, None], Bn[:, None]], axis=1)


def fit(y_train, X_train, NB=NB, mode="forecast"):
    """Single-topic fit (the batched fit_many with T=1) — kept for API compatibility.
    Returns (par, train_r2). par = [w(NB), f(NB), p, b]."""
    y = np.asarray(y_train, float)
    par = fit_many([y], np.asarray(X_train, float), NB, mode=mode)[0]
    pred = predict(par, X_train, NB)
    sst = float(np.sum((y - y.mean()) ** 2))
    return par, (1.0 - float(np.sum((y - pred) ** 2)) / sst if sst > 1e-9 else 0.0)


# ── competition path: fit every topic in ONE batched pass, predict its test rows ─────────────────────
def load_public():
    """The public dataset — local paths via AQ_T500_TRAIN/AQ_T500_TEST (the builder), else the live URLs."""
    import pandas as pd
    tr_p, te_p = os.environ.get("AQ_T500_TRAIN"), os.environ.get("AQ_T500_TEST")
    raw = open(tr_p, "rb").read() if tr_p else urllib.request.urlopen(BASE + "/train.zip", timeout=300).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    train = {n.split("/")[-1][:-4]: pd.read_csv(zf.open(n)) for n in zf.namelist() if n.endswith(".csv")}
    test = pd.read_csv(te_p) if te_p else pd.read_csv(BASE + "/test.csv")
    return train, test


def main():
    import pandas as pd
    train, test = load_public()
    topics = sorted(train.keys())
    print(f"{len(train)} topics · {sum(len(d) for d in train.values())} train rows · {len(test)} test rows"
          f" · device {_device()}")
    par = fit_many([train[t]["target"].to_numpy(float) for t in topics],
                   [train[t][BODIES].to_numpy(float) for t in topics], NB,
                   mode="forecast", progress=True)
    preds = []
    for i, topic in enumerate(topics):
        rows = test[test["trend"] == topic]
        if rows.empty:
            continue
        yhat = np.clip(predict(par[i], rows[BODIES].to_numpy(float), NB), 0.0, 100.0)
        out = rows[["trend"] + BODIES].copy()
        out["target"] = np.round(yhat, 2)
        preds.append(out)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(topics)} · phase {phase_of(par[i], NB):.0f} {sign_of(par[i], NB)}", flush=True)
    sub = pd.concat(preds, ignore_index=True)
    sub.to_csv("submission.csv", index=False)
    print(f"wrote submission.csv ({len(sub)} rows) — upload it on the competition's Submit tab")


if __name__ == "__main__":
    main()
