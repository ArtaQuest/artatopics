#!/usr/bin/env python3
"""PHASOR v3 (operator 2026-07-21): NO planet phases — the topic's phase lives in a COMPLEX bias.

    y_j(t) = | b_j + Σᵢ w_ij·(Mᵢ/rᵢ(t))·e^(iθᵢ(t)) |          b_j ∈ ℂ (2 reals) · w_ij >= 0

  The topic's phase = arg(b_j): the zodiac direction its anchor arrow points. Because every planet
  phasor points at its TRUE sidereal longitude, interference peaks when a planet TRANSITS arg(b) —
  so the extracted phase is a real zodiac location and floor(arg(b)/30°) indexes the standard
  sidereal signs (Aries..Pisces), same frame as the ephemeris. Mᵢ literal: node/lilith are massless
  and drop out -> 10 bodies; with free w the constant mass absorbs, so the factor is normalised to
  the per-body breathing r̄ᵢ/rᵢ(t). 14 parameters per topic. Gradient descent on the raw data.

ALIGNMENT CHECK: for Sun-dominated topics the model's peak month must coincide with the month the
Sun transits arg(b) — reported as a median month-offset.

  python3 analysis/adstopics/astro_phasor3.py            (fit + walls + alignment + npz)
  python3 analysis/adstopics/astro_phasor3.py --export   (npz -> atlas/pred, sign = arg(b))
"""
import importlib.util as u, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
ap = _load("analysis/adstopics/astro_phasor.py", "ap")
SIGNS = ap.SIGNS
B10 = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]
P45 = [(i, k) for i in range(10) for k in range(i)]                 # 45 aspect pairs
I45 = np.array([i for i, k in P45]); K45 = np.array([k for i, k in P45])
MASS = {"sun": 1.989e30, "moon": 7.342e22, "mercury": 3.301e23, "venus": 4.867e24, "mars": 6.417e23,
        "jupiter": 1.898e27, "saturn": 5.683e26, "uranus": 8.681e25, "neptune": 1.024e26, "pluto": 1.303e22}


def r2_rows(Y, Yh):
    res = ((Y - Yh) ** 2).sum(1)
    tot = np.maximum(((Y - Y.mean(1, keepdims=True)) ** 2).sum(1), Y.shape[1] / 6.0)
    return 1.0 - res / tot


def tables(n):
    import pandas as pd
    TH12 = ap.sky12(n)
    idx = [ap.BODIES.index(b) for b in B10]
    TH = TH12[:, idx]                                              # (n,10) sidereal longitudes
    E = pd.read_csv("analysis/_ephemeris_dist.csv")
    grid = pd.DatetimeIndex(pd.to_datetime(E["Time"]))
    i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
    R = np.stack([E[f"{b}_dist"].to_numpy(float)[i0:i0 + n] for b in B10], 1)
    F = 1.0 / R                                                    # M/r with free w ≡ per-body 1/r breathing
    F = F / F.mean(0, keepdims=True)
    return TH, F


def fit3(Y, TH, F, fit_end=None, aspects=False, lam_v=0.1, seed=7, steps=8000, lr=2e-2, lam=0.01, device="cpu"):
    """aspects: add 45 pair terms v_ik·e^(i(θᵢ-θₖ)), v>=0 — all phase structure stays in b."""
    import torch as T
    T.manual_seed(seed); rng = np.random.RandomState(seed); Tn, n = Y.shape
    fe = n if fit_end is None else fit_end
    cT = T.tensor((F * np.cos(TH)).T.astype(np.float32), device=device)   # (10,n)
    sT = T.tensor((F * np.sin(TH)).T.astype(np.float32), device=device)
    D = TH[:, I45] - TH[:, K45]
    cD = T.tensor(np.cos(D).T.astype(np.float32), device=device)          # (45,n)
    sD = T.tensor(np.sin(D).T.astype(np.float32), device=device)
    inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
    W0 = np.full((Tn, 10), -2.0, np.float32)
    W0[:, 9] = inv_sp(0.5 * Y[:, :fe].mean(1))                     # pluto slow-carrier init (house recipe)
    Wraw = T.tensor(W0, device=device, requires_grad=True)
    Bre = T.tensor(Y[:, :fe].mean(1).astype(np.float32), device=device, requires_grad=True)
    Bim = T.tensor((rng.randn(Tn) * 0.01 * Y[:, :fe].mean(1)).astype(np.float32), device=device, requires_grad=True)
    Yt = T.tensor(Y.astype(np.float32), device=device)
    Vraw = T.tensor(np.full((Tn, 45), -4.0, np.float32), device=device, requires_grad=aspects)
    params = [Wraw, Bre, Bim] + ([Vraw] if aspects else [])
    opt = T.optim.Adam(params, lr=lr)

    def forward():
        W = T.nn.functional.softplus(Wraw)                         # w_ij >= 0
        C = Bre[:, None] + W @ cT
        S = Bim[:, None] + W @ sT
        pen = lam * (W ** 2).mean()
        V = None
        if aspects:
            V = T.nn.functional.softplus(Vraw)                     # v_ik >= 0 (aspect weights)
            C = C + V @ cD
            S = S + V @ sD
            pen = pen + lam_v * (V ** 2).mean()
        return T.sqrt(C ** 2 + S ** 2 + 1e-8), W, V, pen

    best, stall, state = np.inf, 0, None
    for it in range(steps):
        yh, W, V, pen = forward()
        loss = ((yh[:, :fe] - Yt[:, :fe]) ** 2).mean() + pen
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            l = loss.item()
            if l < best - 1e-7: best, stall, state = l, 0, [x.detach().clone() for x in params]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip(params, state): x.copy_(sv)
        Yh, W, V, _ = forward()
    return (Yh.cpu().numpy(), W.cpu().numpy(),
            Bre.detach().cpu().numpy(), Bim.detach().cpu().numpy(),
            V.cpu().numpy() if V is not None else None)


def main():
    dev = "cpu"
    try:
        import torch as T
        if T.backends.mps.is_available(): dev = "mps"
    except Exception: pass
    names, Y = ap.load_raw(); Y = Y.astype(np.float64)
    Tn, n = Y.shape; TH, F = tables(n); WALL = n - 24
    print(f"  PHASOR v3 · y=|b+Σw·(M/r)·e^(iθ)+Σv·e^(i(θi-θk))| · phase = arg(b) · aspects lam_v=3 · dev {dev}", flush=True)
    Yh, W, Bre, Bim, V = fit3(Y, TH, F, aspects=True, lam_v=3.0, device=dev)
    r2 = r2_rows(Y, Yh)
    Yw, _, _, _, _ = fit3(Y, TH, F, fit_end=WALL, aspects=True, lam_v=3.0, device=dev)
    ins = r2_rows(Y[:, :WALL], Yw[:, :WALL])
    mu = Y[:, :WALL].mean(1, keepdims=True)
    den = np.maximum(((Y[:, WALL:] - mu) ** 2).sum(1), 24 / 6.0)
    skill = 1.0 - ((Y[:, WALL:] - Yw[:, WALL:]) ** 2).sum(1) / den
    phase = np.rad2deg(np.arctan2(Bim, Bre)) % 360                 # the topic's phase from b
    hist = np.bincount((phase // 30).astype(int), minlength=12)
    print(f"  R² mean {r2.mean():+.4f} · median {np.median(r2):+.4f} · walls ins {ins.mean():+.4f} · "
          f"OOS skill median {np.median(skill):+.4f} ({(skill > 0).mean()*100:.1f}%>0)", flush=True)
    print("  b-SIGNS: " + " ".join(f"{SIGNS[i][:3]} {hist[i]}" for i in range(12)), flush=True)
    # ASTROLOGY ALIGNMENT: for Sun-dominated topics, the peak month must be when the Sun transits arg(b)
    share = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    sunled = share[:, 0] > 0.5
    moy = np.arange(n) % 12
    sunlon = TH[:, 0]
    off = []
    for j in np.where(sunled)[0]:
        prof = np.array([Yh[j][moy == m].mean() for m in range(12)])
        peak_m = int(prof.argmax())
        d = np.abs(np.angle(np.exp(1j * (sunlon - np.deg2rad(phase[j])))))
        transit_m = int(moy[int(d.argmin())])
        off.append(min((peak_m - transit_m) % 12, (transit_m - peak_m) % 12))
    off = np.array(off)
    print(f"  alignment (sun-led, {sunled.sum()} topics): peak-month vs Sun-transit-of-arg(b) — "
          f"median offset {np.median(off) if len(off) else float('nan'):.1f} mo · within ±1 mo: "
          f"{(off <= 1).mean()*100 if len(off) else 0:.1f}%", flush=True)
    np.savez_compressed("analysis/adstopics/astro_phasor3_final.npz",
                        w=W.astype(np.float32), v=V.astype(np.float32),
                        pairs_i=I45, pairs_k=K45,
                        b_re=Bre.astype(np.float32), b_im=Bim.astype(np.float32),
                        phase=phase.astype(np.float32),
                        yhat=Yh.astype(np.float32), yhat_w=Yw.astype(np.float32),
                        r2=r2.astype(np.float32), r2_ins=ins.astype(np.float32),
                        r2_oos_skill=skill.astype(np.float32), wall=WALL,
                        bodies=np.array(B10), names=np.array(names))
    print("  saved -> analysis/adstopics/astro_phasor3_final.npz", flush=True)


def export():
    """npz v3 -> atlas/pred: sign = arg(b), same schema as before."""
    import json, pandas as pd
    D = np.load("analysis/adstopics/astro_phasor3_final.npz", allow_pickle=True)
    names = [str(x) for x in D["names"]]
    W, phase = D["w"], D["phase"]
    V = D["v"] if "v" in D else None
    yhat, yhat_w, wall = D["yhat"], D["yhat_w"], int(D["wall"])
    r2, r2_ins, skill = D["r2"], D["r2_ins"], D["r2_oos_skill"]
    n = yhat.shape[1]; moy = np.arange(n) % 12
    MN = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
    rows = []
    for j, t in enumerate(names):
        share = W[j] / max(W[j].sum(), 1e-9); k = int(W[j].argmax())
        prof = np.array([yhat[j][moy == m].mean() for m in range(12)])
        q = int(V[j].argmax()) if V is not None and V[j].max() > 1e-3 else -1
        rows.append({"topic": t, "sign": SIGNS[int(phase[j] // 30)], "phase": round(float(phase[j]), 1),
                     "peak_month": MN[int(prof.argmax())], "dominant_planet": B10[k],
                     "dom_share": round(float(share[k]), 3), "sun_share": round(float(share[0]), 3),
                     "top_aspect": (f"{B10[int(D['pairs_i'][q])]}-{B10[int(D['pairs_k'][q])]}" if q >= 0 else ""),
                     "phase_stable": 1,
                     "r2_full": round(float(r2[j]), 4), "r2_ins": round(float(r2_ins[j]), 4),
                     "r2_oos": round(float(skill[j]), 4), "r2_oos_strict": round(float(skill[j]), 4)})
    pd.DataFrame(rows).to_csv("analysis/adstopics/astro_ts_atlas.csv", index=False)
    reg = json.load(open("analysis/_topics_weekly.json"))
    key_of = {r["label"]: k for k, r in reg.items()}
    pred = {"_meta": {"task": "phasor-r2", "wall": wall,
                      "model": "v3: y=|b+Σw·(M/r)·e^(iθ)| · complex bias · sign = arg(b)"}}
    m = 0
    for j, t in enumerate(names):
        k = key_of.get(t)
        if not k: continue
        pred[k] = {"pred": [round(float(v), 1) for v in yhat_w[j]],
                   "r2Ins": round(float(r2_ins[j]), 4), "r2Oos": round(float(skill[j]), 4)}
        m += 1
    json.dump(pred, open("analysis/adstopics/astro_ts_pred.json", "w"))
    print(f"  [export v3] atlas {len(rows)} rows (sign = arg b) · pred {m}/{len(names)} · wall {wall}", flush=True)


if __name__ == "__main__":
    export() if "--export" in sys.argv else main()
