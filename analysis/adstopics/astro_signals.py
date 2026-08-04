#!/usr/bin/env python3
"""ASTROLOGY-ONLY SIGNALS (operator 2026-07-19): predict the rate-change direction from the SKY
alone — planetary longitudes AND aspects (pairwise angular relationships), retrogrades, nodes.
The question the whole program circles: does anything beyond the Sun's annual cycle carry honest
predictive signal?

Every feature is a deterministic function of the sky at month t (known for the test months — NOT
leakage). Per topic, per feature we fit a coefficient by cross-year CV (fit the sign-projection on
even years, score on odd, average) so a feature is credited only if its cycle TRANSFERS across
years — this is what separates a real annual signal (the Sun) from spurious in-window fits.

    logit(i, t) = sum_f  g_{i,f} * c_{i,f} * feature_f(t)            (cross-year-gated astro furnace)

Feature families (toggle via FEATURES):
  lon    : cos/sin(k * theta_p)          per body, harmonics k=1..K      (the annual cycle lives in the Sun)
  aspect : cos/sin(k * (theta_p-theta_q)) per pair, harmonics k=1..Ka    (conjunction/opposition/trine/square...)
  retro  : per-body retrograde indicator (monthly velocity < 0)
"""
import importlib.util as u, itertools, json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
rc = _load("analysis/adstopics/ratechange.py", "rc")
pr = _load("analysis/adstopics/phase_rotary.py", "pr")
H = 24
BODIES = pr.BODIES; NP = pr.NP


def sky_features(n, families=("lon",), K=6, Ka=4, pairs=None):
    """Return (Phi [n, F], names[F]). All deterministic functions of the sky at each month."""
    TH = pr.planet_longitudes(n)                      # (n, NP) radians
    cols, names = [], []
    if "lon" in families:
        for p in range(NP):
            for k in range(1, K + 1):
                cols.append(np.cos(k * TH[:, p])); names.append(f"cos{k}·{BODIES[p]}")
                cols.append(np.sin(k * TH[:, p])); names.append(f"sin{k}·{BODIES[p]}")
    if "aspect" in families:
        prs = pairs if pairs is not None else list(itertools.combinations(range(NP), 2))
        for (p, q) in prs:
            sep = TH[:, p] - TH[:, q]
            for k in range(1, Ka + 1):
                cols.append(np.cos(k * sep)); names.append(f"cos{k}·{BODIES[p]}-{BODIES[q]}")
                cols.append(np.sin(k * sep)); names.append(f"sin{k}·{BODIES[p]}-{BODIES[q]}")
    if "retro" in families:
        for p in range(NP):
            d = np.diff(np.unwrap(TH[:, p]), prepend=TH[0, p])
            cols.append((d < 0).astype(float)); names.append(f"retro·{BODIES[p]}")
    return np.stack(cols, 1).astype(np.float64), names


def gated_furnace(b, Phi, ridge=1.0):
    """Per-topic cross-year-gated linear astro furnace. Returns test logits (T,24) + kept-feature mass."""
    Tn = b.Y.shape[0]; F = Phi.shape[1]
    lw = np.where(b.tie, 0.0, b.L * 2.0 - 1.0)
    tr = np.arange(1, b.b)                             # fit coeffs on all pre-test months (<b.b), leak-free
    yr = (tr - 1) // 12; fold = yr % 2
    Lc = lw[:, tr]; Ph = Phi[tr]                       # (T,ntr), (ntr,F)
    # standardize features on the fit window
    mu, sd = Ph.mean(0), Ph.std(0) + 1e-9; Phs = (Ph - mu) / sd
    # per-topic coefficient = ridge projection of direction onto features (closed form, shared Gram)
    G = Phs.T @ Phs + ridge * F * np.eye(F)           # (F,F)
    Ginv = np.linalg.inv(G)
    C = (Lc @ Phs) @ Ginv                             # (T,F) per-topic coefficients
    # cross-year gate per (topic,feature): does coef fit on one fold predict on the other?
    gate = np.zeros((Tn, F))
    for fit, sc in ((0, 1), (1, 0)):
        mf = fold == fit; ms = fold == sc
        Cf = (Lc[:, mf] @ Phs[mf]) @ np.linalg.inv(Phs[mf].T @ Phs[mf] + ridge * F * np.eye(F))
        pred = Cf @ Phs[ms].T                         # (T, nsc)
        gate += (Lc[:, ms] * pred).sum(1, keepdims=True) / (np.abs(Lc[:, ms]).sum(1, keepdims=True) + 1e-9) * np.sign(Cf * C).clip(0)
    gate = np.clip(gate / 2.0, 0, None)
    gate = gate / (gate.max(1, keepdims=True) + 1e-9)  # per-topic relative gate
    test = np.arange(b.b, b.n)
    Pht = (Phi[test] - mu) / sd
    Z = (C * gate) @ Pht.T                            # (T,24)
    kept = (gate > 0.1).sum(1).mean()
    return Z, kept


def experiment(fam, K=6, Ka=4, ridge=1.0, pairs=None, tag=None):
    names, Y = rc.bal.load_all(); b = rc.BenchR(Y)
    Phi, fn = sky_features(b.n, fam, K, Ka, pairs)
    Z, kept = gated_furnace(b, Phi, ridge)
    auc = b.score(Z, tag or "+".join(fam))
    print(f"    [{tag or '+'.join(fam)}] features {Phi.shape[1]} · mean kept {kept:.1f} · test {auc:.4f}", flush=True)
    return auc


def main():
    names, Y = rc.bal.load_all(); b = rc.BenchR(Y); rc.ceilings(b)
    print("  -- astrology-only signal search --", flush=True)
    experiment(("lon",), K=6, tag="Sun+planets longitudes")
    experiment(("aspect",), Ka=4, tag="aspects only")
    experiment(("lon", "aspect"), K=6, Ka=4, tag="longitudes + aspects")
    experiment(("lon", "aspect", "retro"), K=6, Ka=4, tag="longitudes + aspects + retro")
    # Sun-only baseline (the annual cycle)
    experiment(("lon",), K=6, pairs=[], tag="all-body longitudes (K6)")


if __name__ == "__main__":
    main()
