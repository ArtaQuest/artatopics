#!/usr/bin/env python3
"""THE MODELS MOST COMPATIBLE WITH ASTROLOGY (operator 2026-08-07), tested like everything else.

Astrology's structural claims, mapped honestly:
  RETURNS/TRANSITS TO A NATAL CHART -- an entity has a birth moment, and effects come when planets
    return to their natal places. A research field HAS a birth: its emergence year. So the natal
    model is the record receiver with the phases FIXED BY DOCTRINE instead of fitted:
        y_j(t) = max( b_j + SUM_i a_ji * cos( theta_i(t) - theta_i(T0_j) ) , 0 )^2
    theta_i(T0_j) = body i's position in the field's birth year. ZERO fitted phases; 8 parameters.
    One closed-form solve per field -- no sweep at all.
  ASPECTS -- effects from angular separations between planet PAIRS (conjunction / sextile / square /
    trine / opposition are harmonics m of theta_i - theta_k, with the aspect's own orientation).
    cos(m*D - psi) = cos(psi)cos(mD) + sin(psi)sin(mD), so the model is LINEAR with the per-pair
    orientation absorbed:
        y_j(t) = max( b_j + SUM_{i<k in SLOW} [ u cos(m D_ik) + v sin(m D_ik) ] , 0 )^2
    Slow pairs only (jupiter..pluto + node: 15 pairs), first harmonic: 31 parameters. Zodiac-zero
    free -- the purest "aspects drive events" doctrine.
  SIGNS/DIGNITIES are step functions over 12x30-degree bins == harmonics k<=6, which lost in earlier
    ablations (higher harmonics collapse). HOUSES need a time of day and a place; a yearly global
    series has neither. Both excluded, stated rather than hidden.

Same weights, same anchor, same walls, same reference as every test in this repository.

  python3 analysis/arxivtopics/astrology_models.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import arxiv_fit as af

names, Y, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
n = Y.shape[1]; ne = TH.shape[0]; nb = TH.shape[1]
WALLS = list(range(n - 63, n - 29, 3))
BIRTH = af.META["topic_valid"].argmax(1)                     # each field's emergence year index
SLOW = [af.BODIES.index(b) for b in ("jupiter", "saturn", "uranus", "neptune", "pluto", "node")]


def wls_anchored(Xj, wall, hz, w_j, ysq_j, m_j):
    """One anchored weighted least-squares solve on a per-field design (T, d)."""
    d = Xj.shape[1]
    Xt, Xa = Xj[:wall], Xj[wall:hz]
    aw = af.LAM_HORIZON / (m_j ** 2) / max(hz - wall, 1)
    A = Xt.T @ (Xt * w_j[:, None]) + aw * (Xa.T @ Xa) + 1e-8 * np.eye(d)
    b = Xt.T @ (w_j * ysq_j) + aw * Xa.sum(0) * m_j
    return np.linalg.solve(A, b)


def prep(wall):
    tv = af.META["topic_valid"][:, :wall].astype(float)
    wy = np.clip(af.META["evidence"][:wall], 0, None) ** 0.75
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    Wa = np.zeros_like(W); Wa[:, wall - af.ANCHOR_K:] = (tv * wy[None])[:, wall - af.ANCHOR_K:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    Ysq = np.sqrt(Y)
    return W, Ysq, np.maximum((Ysq[:, :wall] * Wa).sum(1), 1e-3), min(wall + af.HORIZON, ne)


def fit_natal(wall):
    """Phases from the BIRTH CHART: 8 fitted numbers per field, no sweep."""
    W, Ysq, MJ, hz = prep(wall)
    P = np.zeros((Y.shape[0], ne))
    for j in range(Y.shape[0]):
        natal = TH[BIRTH[j]]                                  # the sky the year the field emerged
        Xj = np.concatenate([np.ones((ne, 1)), np.cos(TH - natal[None, :])], 1)
        c = wls_anchored(Xj, wall, hz, W[j], Ysq[j, :wall], MJ[j])
        P[j] = np.maximum(Xj @ c, 0.0) ** 2
    return P


def fit_aspects(wall):
    """Pure aspects: slow-pair separations, first harmonic, orientation absorbed. 31 params."""
    W, Ysq, MJ, hz = prep(wall)
    feats = [np.ones((ne, 1))]
    for a in range(len(SLOW)):
        for b_ in range(a + 1, len(SLOW)):
            D = TH[:, SLOW[a]] - TH[:, SLOW[b_]]
            feats += [np.cos(D)[:, None], np.sin(D)[:, None]]
    Xall = np.concatenate(feats, 1)                          # (ne, 31)
    P = np.zeros((Y.shape[0], ne))
    for j in range(Y.shape[0]):
        c = wls_anchored(Xall, wall, hz, W[j], Ysq[j, :wall], MJ[j])
        P[j] = np.maximum(Xall @ c, 0.0) ** 2
    return P


def auc_at(Yh, wall):
    tv = af.META["topic_valid"]; tvw = tv[:, :wall].astype(float)
    mu = (Y[:, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    return float(np.mean([1.0 - ((Y[:, wall + h] - Yh[:, wall + h]) ** 2).sum() /
                max(((Y[:, wall + h] - mu) ** 2).sum(), 1e-9) for h in range(min(30, n - wall))]))


def main():
    print("═══ ASTROLOGY-STRUCTURED MODELS vs THE RECORD · twelve origins ═══", flush=True)
    t0 = time.time()
    out = {}
    for lab, fn, par in (("natal returns (8p, no fitted phase)", fit_natal, 8),
                          ("pure aspects (31p, slow pairs)", fit_aspects, 31),
                          ("record (9p, fitted tuning)", lambda w: af.fit_final(Y, TH, w)[0], 9)):
        a = np.array([auc_at(fn(w), w) for w in WALLS])
        out[lab] = a
        print(f"  {lab:38s} mean {a.mean():+.4f} · 1996 {a[-1]:+.4f}   " +
              " ".join(f"{v:+.3f}" for v in a), flush=True)
    print(f"  [{time.time()-t0:.0f}s] · persistence +0.8511 for scale", flush=True)
    json.dump({k: {"auc": [round(float(v), 4) for v in out[k]],
                   "mean": round(float(out[k].mean()), 4)} for k in out},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "astrology_models.json"), "w"), indent=1)
    print("ASTRODONE", flush=True)


if __name__ == "__main__":
    main()


def fit_natal_offset(wall, delta):
    """Natal phases plus ONE shared rotation delta for the whole atlas."""
    W, Ysq, MJ, hz = prep(wall)
    P = np.zeros((Y.shape[0], ne))
    for j in range(Y.shape[0]):
        natal = TH[BIRTH[j]] + delta
        Xj = np.concatenate([np.ones((ne, 1)), np.cos(TH - natal[None, :])], 1)
        c = wls_anchored(Xj, wall, hz, W[j], Ysq[j, :wall], MJ[j])
        P[j] = np.maximum(Xj @ c, 0.0) ** 2
    return P


def offset_experiment():
    """delta on a 15-degree grid, CHOSEN ON THE NINE EARLY ORIGINS, held out on 1990/93/96."""
    SEL = 9
    grid = np.deg2rad(np.arange(0, 180, 15.0))               # +180 == sign flips, already free
    best, curves = None, {}
    for dlt in grid:
        a = np.array([auc_at(fit_natal_offset(w, dlt), w) for w in WALLS])
        curves[round(float(np.degrees(dlt)))] = a
        print(f"    delta={np.degrees(dlt):5.1f}  early(9) {a[:SEL].mean():+.4f} · held(3) {a[SEL:].mean():+.4f}", flush=True)
        if best is None or a[:SEL].mean() > curves[best][:SEL].mean():
            best = round(float(np.degrees(dlt)))
    a = curves[best]
    print(f"\n  CHOSEN ON EARLY ORIGINS: delta = {best} deg", flush=True)
    print(f"  natal+dial  mean {a.mean():+.4f} · held(3) {a[SEL:].mean():+.4f} · 1996 {a[-1]:+.4f}", flush=True)
    print(f"  record      mean +0.8751 · held(3) +0.8237 · 1996 +0.7990", flush=True)
    json.dump({"grid_deg": [int(np.degrees(g)) for g in grid], "chosen": best,
               "auc": {str(k): [round(float(v), 4) for v in curves[k]] for k in curves}},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "astrology_offset.json"), "w"), indent=1)
    print("OFFSETDONE", flush=True)


if __name__ == "__main__" and os.environ.get("OFFSET"):
    offset_experiment()
