#!/usr/bin/env python3
"""PHASOR ABLATION SUITE (operator 2026-07-20): triple-check the finalized model and measure —
rigorously, same budget per variant — what it captures: the SIGN (per-topic phase), the ASPECTS
(pairwise interference terms), RETROGRADE (true kinematics vs uniform motion), each body, and
whether the REAL sky matters at all (shifted-sky and fake-sky controls).

CHECKS (on the shipped npz, no refit):
  aspects algebra   |b+Σa·e^(iΔ)|² == b² + 2bΣaᵢcosΔᵢ + Σaᵢ² + 2Σᵢ<ₖaᵢaₖcos(θᵢ-θₖ)  (numeric)
  phase gradient    d loss/d p == 0 in the literal model (cancellation), != 0 in dc
  retrograde audit  per-body fraction of months with westward motion at the MONTHLY cadence
                    (slow bodies well-sampled; Moon/Mercury/Venus aliased — stated, not hidden)

ABLATIONS (dc phasor, raw data, identical budget: steps/lr/lam fixed; full-fit R² per topic):
  reference         12 bodies, magnitude link, learned p_j
  sign OFF          p_j frozen at 0             -> the value of the per-topic rotation
  sign RANDOM       p_j frozen at random        -> placebo for the learned phase
  aspects OFF       REAL-part link b+Σaᵢcos(θᵢ-p) (no |·| -> no pairwise cross terms)
  retrograde OFF    de-retrograded sky: each body's longitude replaced by its mean motion
                    (linear fit of the unwrapped θ) — loops and speed variation removed
  shifted sky       the true ephemeris rolled +37 months (real structure, wrong calendar)
  fake sky          same mean speeds, random start phases (seed 11)
  body groups       sun-only · fast-5 (sun..mars) · slow-7 (jupiter..lilith) · minus-sun ·
                    minus-node&lilith
  walls (OOS skill) for reference · aspects OFF · retrograde OFF · shifted sky

  python3 analysis/adstopics/phasor_ablation.py
"""
import importlib.util as u, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
ap = _load("analysis/adstopics/astro_phasor.py", "ap")
BODIES = ap.BODIES


def r2_rows(Y, Yh, floor=True):
    res = ((Y - Yh) ** 2).sum(1)
    tot = ((Y - Y.mean(1, keepdims=True)) ** 2).sum(1)
    if floor: tot = np.maximum(tot, Y.shape[1] / 6.0)
    return 1.0 - res / tot


def checks(Y, TH):
    D = np.load("analysis/adstopics/astro_phasor_final.npz", allow_pickle=True)
    Ad, pd_, bd = D["a_dc"].astype(float), D["p_dc"].astype(float), D["b_dc"].astype(float)
    rng = np.random.RandomState(0); ok = True
    # 1. ASPECTS ALGEBRA: the squared magnitude expands into pairwise separation (aspect) terms.
    for j in rng.choice(len(Ad), 3, replace=False):
        a = Ad[j]; dphi = TH - pd_[j]
        C = bd[j] + (a * np.cos(dphi)).sum(1); S = (a * np.sin(dphi)).sum(1)
        lhs = C ** 2 + S ** 2
        cross = np.zeros(TH.shape[0])
        for i in range(12):
            for k in range(i + 1, 12):
                cross += 2 * a[i] * a[k] * np.cos(TH[:, i] - TH[:, k])
        rhs = bd[j] ** 2 + 2 * bd[j] * (a * np.cos(dphi)).sum(1) + (a ** 2).sum() + cross
        ok &= np.abs(lhs - rhs).max() < 1e-6
    print(f"  [{'PASS' if ok else 'FAIL'}] aspects algebra: |S|² == Σa² + b² + 2bΣa·cosΔ + 2ΣΣ aᵢaₖcos(θᵢ-θₖ)  "
          f"(the aspect terms are INTRINSIC to the magnitude)", flush=True)
    # 2. PHASE GRADIENT: p cancels in the literal magnitude, acts in dc.
    import torch as T
    j = 5; a = T.tensor(Ad[j]); THt = T.tensor(TH); yt = T.tensor(Y[j])
    for mode, want0 in (("literal", True), ("dc", False)):
        p = T.tensor(0.7, requires_grad=True)
        dphi = THt - p
        C = (a * T.cos(dphi)).sum(1); S = (a * T.sin(dphi)).sum(1)
        if mode == "dc": C = C + T.tensor(bd[j])
        loss = ((T.sqrt(C ** 2 + S ** 2 + 1e-8) - yt) ** 2).mean(); loss.backward()
        g = abs(float(p.grad))
        good = (g < 1e-6) if want0 else (g > 1e-4)
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] phase gradient ({mode}): |dL/dp| = {g:.2e} "
              f"({'must be ~0 — p cancels' if want0 else 'must be nonzero — p acts against b'})", flush=True)
    # 3. RETROGRADE AUDIT at monthly cadence.
    d = np.angle(np.exp(1j * np.diff(TH, axis=0)))
    frac = (d < 0).mean(0)
    print("  retrograde months seen by the model (monthly cadence): "
          + " ".join(f"{b} {f*100:.0f}%" for b, f in zip(BODIES, frac)), flush=True)
    print("    (jupiter..pluto/node/lilith well-sampled; moon/mercury/venus aliased at monthly cadence — "
          "their 'motion' is a snapshot alias, stated openly)", flush=True)
    return ok


def fit_dc(Y, TH, learn_phase=True, p_fix=None, link="mag", fit_end=None, F=None,
           seed=7, steps=5000, lr=2e-2, lam=0.01, device="cpu"):
    """Same recipe as the shipped model, budget-matched across variants. TH: (n, nb)."""
    import torch as T
    T.manual_seed(seed); Tn, n = Y.shape; nb = TH.shape[1]
    fe = n if fit_end is None else fit_end
    inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
    A0 = np.full((Tn, nb), -2.0, np.float32)
    slow = nb - 3 if nb > 3 else 0                              # a slow-ish carrier column
    A0[:, slow] = inv_sp(Y[:, :fe].mean(1))
    Araw = T.tensor(A0, device=device, requires_grad=True)
    if p_fix is None: p_fix = np.zeros(Tn)
    U = T.tensor(np.stack([np.sin(p_fix), np.cos(p_fix)], 1).astype(np.float32),
                 device=device, requires_grad=learn_phase)
    Bp = T.tensor(Y[:, :fe].mean(1).astype(np.float32), device=device, requires_grad=True)
    params = [Araw, Bp] + ([U] if learn_phase else [])
    THd = T.tensor(TH.T.astype(np.float32), device=device)
    Yt = T.tensor(Y.astype(np.float32), device=device)
    Fd = None if F is None else T.tensor(F.T.astype(np.float32), device=device)   # (nb, n) amp factor
    opt = T.optim.Adam(params, lr=lr)

    def forward():
        A = T.nn.functional.softplus(Araw)
        p = T.atan2(U[:, 0], U[:, 1])
        ang = THd[None] - p[:, None, None]
        AF = A[:, :, None] if Fd is None else A[:, :, None] * Fd[None]
        C = (AF * T.cos(ang)).sum(1); S = (AF * T.sin(ang)).sum(1)
        if link == "mag":
            return T.sqrt((C + Bp[:, None]) ** 2 + S ** 2 + 1e-8), A
        return Bp[:, None] + C, A                                # REAL link: no interference terms

    best, stall, state = np.inf, 0, None
    for it in range(steps):
        yh, Acur = forward()
        loss = ((yh[:, :fe] - Yt[:, :fe]) ** 2).mean() + lam * (Acur ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            l = loss.item()
            if l < best - 1e-7: best, stall, state = l, 0, [x.detach().clone() for x in params]
            else:
                stall += 1
                if stall >= 8: break
    with T.no_grad():
        for x, sv in zip(params, state): x.copy_(sv)
        Yh, _ = forward()
    return Yh.cpu().numpy()


def deretro(TH):
    """Mean-motion sky: per body, linear fit of the unwrapped longitude — retrograde loops and
    speed variation removed, mean speed and mean phase kept."""
    n = TH.shape[0]; t = np.arange(n); out = np.zeros_like(TH)
    for i in range(TH.shape[1]):
        uw = np.unwrap(TH[:, i])
        c = np.polyfit(t, uw, 1)
        out[:, i] = np.mod(np.polyval(c, t), 2 * np.pi)
    return out


def fake_sky(TH, seed=11):
    n = TH.shape[0]; t = np.arange(n); rng = np.random.RandomState(seed); out = np.zeros_like(TH)
    for i in range(TH.shape[1]):
        c1 = np.polyfit(t, np.unwrap(TH[:, i]), 1)[0]           # same mean speed
        out[:, i] = np.mod(rng.uniform(0, 2 * np.pi) + c1 * t, 2 * np.pi)
    return out


def main():
    dev = "cpu"
    try:
        import torch as T
        if T.backends.mps.is_available(): dev = "mps"
    except Exception: pass
    names, Y = ap.load_raw(); Y = Y.astype(np.float64)
    Tn, n = Y.shape; TH = ap.sky12(n); WALL = n - 24
    print(f"  {Tn} topics · dc phasor · budget-matched ablations (steps 5000, lam 0.01) · dev {dev}", flush=True)
    print("\n  == TRIPLE-CHECK (shipped artifact) ==", flush=True)
    checks(Y, TH)

    FAST = [0, 1, 2, 3, 4]; SLOW = [5, 6, 7, 8, 9, 10, 11]
    rng = np.random.RandomState(23)
    exps = [
        ("reference (12 bodies · mag · learned p)", dict()),
        ("sign OFF (p frozen 0)",                   dict(learn_phase=False)),
        ("sign RANDOM (p frozen random)",           dict(learn_phase=False, p_fix=rng.uniform(0, 2 * np.pi, Tn))),
        ("aspects OFF (real link, no |·|)",         dict(link="real")),
        ("retrograde OFF (mean-motion sky)",        dict(TH2=deretro(TH))),
        ("shifted sky (+37 months)",                dict(TH2=np.roll(TH, 37, axis=0))),
        ("fake sky (same speeds, random phases)",   dict(TH2=fake_sky(TH))),
        ("sun only",                                dict(TH2=TH[:, [0]])),
        ("fast-5 (sun..mars)",                      dict(TH2=TH[:, FAST])),
        ("slow-7 (jupiter..lilith)",                dict(TH2=TH[:, SLOW])),
        ("minus sun (11 bodies)",                   dict(TH2=TH[:, 1:])),
        ("minus node+lilith (10)",                  dict(TH2=TH[:, :10])),
    ]
    ref = None
    for tag, kw in exps:
        TH2 = kw.pop("TH2", TH)
        Yh = fit_dc(Y, TH2, device=dev, **kw)
        r2 = r2_rows(Y, Yh)
        d = "" if ref is None else f" · Δref {r2.mean() - ref:+.4f}"
        if ref is None: ref = r2.mean()
        print(f"  [{tag:42s}] mean R² {r2.mean():+.4f} · median {np.median(r2):+.4f}{d}", flush=True)

    print("\n  == WALLS (fit<186 · OOS skill on the last 24 months) ==", flush=True)
    mu = Y[:, :WALL].mean(1, keepdims=True)
    den = np.maximum(((Y[:, WALL:] - mu) ** 2).sum(1), 24 / 6.0)
    for tag, kw in [("reference", dict()), ("aspects OFF", dict(link="real")),
                    ("retrograde OFF", dict(TH2=deretro(TH))), ("shifted sky", dict(TH2=np.roll(TH, 37, axis=0)))]:
        TH2 = kw.pop("TH2", TH)
        Yh = fit_dc(Y, TH2, fit_end=WALL, device=dev, **kw)
        skill = 1.0 - ((Y[:, WALL:] - Yh[:, WALL:]) ** 2).sum(1) / den
        print(f"  [{tag:42s}] OOS skill median {np.median(skill):+.4f} · >0: {(skill > 0).mean()*100:.1f}%", flush=True)


# ── PHYSICAL WEIGHTING (operator 2026-07-20): |b + Σᵢ wᵢ·(Mᵢ/rᵢ(t))·e^(i(θᵢ(t)-p))| ──────────────
# Masses in kg; node/lilith are massless points (mass 0 -> absent from the physics mix; factor 1 in
# the w-free spec run, where any CONSTANT scale is absorbed by wᵢ anyway — only the TIME-VARYING
# 1/rᵢ(t) modulation adds content there).
MASS = {"sun": 1.989e30, "moon": 7.342e22, "mercury": 3.301e23, "venus": 4.867e24, "mars": 6.417e23,
        "jupiter": 1.898e27, "saturn": 5.683e26, "uranus": 8.681e25, "neptune": 1.024e26,
        "pluto": 1.303e22, "node": 0.0, "lilith": 0.0}


def dist12(n):
    import pandas as pd
    E = pd.read_csv("analysis/_ephemeris_dist.csv")
    grid = pd.DatetimeIndex(pd.to_datetime(E["Time"]))
    i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
    R = np.stack([E[f"{b}_dist"].to_numpy(float)[i0:i0 + n] for b in BODIES], 1)
    assert np.isfinite(R).all() and (R > 0).all()
    return R                                                     # (n, 12) AU


def fit_mix(Y, G, fit_end=None, seed=7, steps=4000, lr=2e-2, device="cpu"):
    """Pure-physics mix: y_j = |b_j + s_j·e^(-i p_j)·G(t)| — the body blend is FIXED by M/r^k;
    only scale s_j, phase p_j, reference b_j are learned (3 params per topic)."""
    import torch as T
    T.manual_seed(seed); Tn, n = Y.shape
    fe = n if fit_end is None else fit_end
    G = G / (np.abs(G).mean() + 1e-30)                            # unit scale (absorbed by s_j; avoids fp32 overflow)
    Gr = T.tensor(np.real(G).astype(np.float32), device=device)
    Gi = T.tensor(np.imag(G).astype(np.float32), device=device)
    scale = float(np.abs(G).mean())
    s_raw = T.tensor(np.log(np.expm1(np.clip(Y[:, :fe].std(1) / scale, 1e-3, None))).astype(np.float32),
                     device=device, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (Tn, 1)).astype(np.float32), device=device, requires_grad=True)
    Bp = T.tensor(Y[:, :fe].mean(1).astype(np.float32), device=device, requires_grad=True)
    Yt = T.tensor(Y.astype(np.float32), device=device)
    opt = T.optim.Adam([s_raw, U, Bp], lr=lr)
    best, stall, state = np.inf, 0, None
    for it in range(steps):
        sA = T.nn.functional.softplus(s_raw)
        cp = U[:, 1] / (U.norm(dim=1) + 1e-9); sp = U[:, 0] / (U.norm(dim=1) + 1e-9)
        C = Bp[:, None] + sA[:, None] * (Gr[None] * cp[:, None] + Gi[None] * sp[:, None])
        S = sA[:, None] * (Gi[None] * cp[:, None] - Gr[None] * sp[:, None])
        yh = T.sqrt(C ** 2 + S ** 2 + 1e-8)
        loss = ((yh[:, :fe] - Yt[:, :fe]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            l = loss.item()
            if l < best - 1e-7: best, stall, state = l, 0, [x.detach().clone() for x in (s_raw, U, Bp)]
            else:
                stall += 1
                if stall >= 8: break
    with T.no_grad():
        for x, sv in zip((s_raw, U, Bp), state): x.copy_(sv)
        sA = T.nn.functional.softplus(s_raw)
        cp = U[:, 1] / (U.norm(dim=1) + 1e-9); sp = U[:, 0] / (U.norm(dim=1) + 1e-9)
        C = Bp[:, None] + sA[:, None] * (Gr[None] * cp[:, None] + Gi[None] * sp[:, None])
        S = sA[:, None] * (Gi[None] * cp[:, None] - Gr[None] * sp[:, None])
        return T.sqrt(C ** 2 + S ** 2 + 1e-8).cpu().numpy()


def physics():
    dev = "cpu"
    try:
        import torch as T
        if T.backends.mps.is_available(): dev = "mps"
    except Exception: pass
    names, Y = ap.load_raw(); Y = Y.astype(np.float64)
    Tn, n = Y.shape; TH = ap.sky12(n); R = dist12(n); WALL = n - 24
    M = np.array([MASS[b] for b in BODIES])
    # APPROACH/RECEDE SIGN (operator): flip the factor while the body RECEDES (r increasing) —
    # the phasor points the opposite way on the outbound half of each synodic cycle.
    sgn = -np.sign(np.gradient(R, axis=0)); sgn[sgn == 0] = 1.0
    ii_pt = [BODIES.index("node"), BODIES.index("lilith")]        # massless points, r ~ constant
    print(f"  {Tn} topics · PHYSICAL weighting |b+Σ wᵢ·(±Mᵢ/rᵢᵏ(t))·e^(i(θᵢ-p))| · dev {dev}", flush=True)
    for name, signed in (("unsigned", False), ("SIGNED (flip when receding)", True)):
        print(f"\n  == {name} ==", flush=True)
        F = 1.0 / R
        F = F / F.mean(0, keepdims=True)
        if signed: F = F * sgn
        F[:, ii_pt] = 1.0
        Yh = fit_dc(Y, TH, F=F, device=dev)
        r2 = r2_rows(Y, Yh)
        print(f"  [spec: w free · amp {'±' if signed else ''}1/rᵢ(t)-modulated       ] mean R² {r2.mean():+.4f} · median {np.median(r2):+.4f}", flush=True)
        for k, tag in ((1, "M/r  (spec mass power)"), (2, "M/r² (gravity)"), (3, "M/r³ (tidal — Moon:Sun ~2:1)")):
            W = M[None] / (R * 1.496e8) ** k                      # kg/km^k; absolute scale absorbed by s_j
            if signed: W = W * sgn
            G = (W * np.exp(1j * TH)).sum(1)                      # fixed complex mix (n,)
            Yh = fit_mix(Y, G, device=dev)
            r2 = r2_rows(Y, Yh)
            print(f"  [physics mix {tag:30s}] mean R² {r2.mean():+.4f} · median {np.median(r2):+.4f}", flush=True)
        W = (1.0 / R) * (M[None] > 0)
        if signed: W = W * sgn
        G = (W * np.exp(1j * TH)).sum(1)                          # massless-control: 1/r only
        Yh = fit_mix(Y, G, device=dev)
        r2 = r2_rows(Y, Yh)
        print(f"  [physics mix 1/r only (no mass)                 ] mean R² {r2.mean():+.4f} · median {np.median(r2):+.4f}", flush=True)
    # walls (OOS skill): spec factor + tidal mix, both signed per the operator
    mu = Y[:, :WALL].mean(1, keepdims=True)
    den = np.maximum(((Y[:, WALL:] - mu) ** 2).sum(1), 24 / 6.0)
    print("", flush=True)
    F = (1.0 / R); F = F / F.mean(0, keepdims=True); F = F * sgn; F[:, ii_pt] = 1.0
    Yh = fit_dc(Y, TH, F=F, fit_end=WALL, device=dev)
    sk = 1.0 - ((Y[:, WALL:] - Yh[:, WALL:]) ** 2).sum(1) / den
    print(f"  [WALLS spec ±1/r-modulated                      ] OOS skill median {np.median(sk):+.4f} · >0: {(sk > 0).mean()*100:.1f}%", flush=True)
    W = (M[None] / (R * 1.496e8) ** 3) * sgn
    G = (W * np.exp(1j * TH)).sum(1)
    Yh = fit_mix(Y, G, fit_end=WALL, device=dev)
    sk = 1.0 - ((Y[:, WALL:] - Yh[:, WALL:]) ** 2).sum(1) / den
    print(f"  [WALLS physics mix tidal ±M/r³                  ] OOS skill median {np.median(sk):+.4f} · >0: {(sk > 0).mean()*100:.1f}%", flush=True)


if __name__ == "__main__":
    import sys
    physics() if "--physics" in sys.argv else main()
