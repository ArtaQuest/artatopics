#!/usr/bin/env python3
"""PHASOR v2 (operator 2026-07-20, rethink): per-body phases + PROPER aspect weights.

    y_j(t) = | b_j + Σᵢ a_ij·e^(i(θᵢ(t)-p_ij)) + Σ_{i>k} w_ikj·e^(i(θᵢ(t)-θₖ(t)+p_ij-p_kj)) |

  p_ij : a SEPARATE phase (sign) per body per topic — the SUN's phase is the topic's SUN SIGN,
         used for the prod classification.
  a_ij : 12 single-body amplitudes >= 0.
  w_ikj: 66 EXPLICIT aspect weights >= 0 — one per pair term e^(i(θᵢ-θₖ+(pᵢ-pₖ))), i>k, decoupled
         from the aᵢaₖ products the plain magnitude induces.
  All parameters by gradient descent on the RAW data (no balancing). Angle-addition folds every
  term into (Tn,12)@(12,n) / (Tn,66)@(66,n) matmuls; time-varying distance factors fold into the
  fixed tables, so no (Tn,66,n) tensor ever exists.

ABLATIONS (budget-matched): structure — full vs singles-only vs pairs-only vs shared-phase (v1 dc);
distance factors on the single-body amplitudes — none · 1/r · 1/r² · sgn(-ṙ)/r · sgn(-ṙ)/r² ·
sgn-only; walls (fit<186) OOS skill for the leaders.

  python3 analysis/adstopics/astro_phasor2.py            (ablation table + final fit + npz)
  python3 analysis/adstopics/astro_phasor2.py --export   (npz -> atlas/pred with SUN-SIGN classification)
"""
import importlib.util as u, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
ap = _load("analysis/adstopics/astro_phasor.py", "ap")
BODIES, SIGNS = ap.BODIES, ap.SIGNS
PAIRS = [(i, k) for i in range(12) for k in range(i)]              # i > k, 66 pairs
II = np.array([i for i, k in PAIRS]); KK = np.array([k for i, k in PAIRS])


def r2_rows(Y, Yh):
    res = ((Y - Yh) ** 2).sum(1)
    tot = np.maximum(((Y - Y.mean(1, keepdims=True)) ** 2).sum(1), Y.shape[1] / 6.0)
    return 1.0 - res / tot


def dist12(n):
    import pandas as pd
    E = pd.read_csv("analysis/_ephemeris_dist.csv")
    grid = pd.DatetimeIndex(pd.to_datetime(E["Time"]))
    i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
    R = np.stack([E[f"{b}_dist"].to_numpy(float)[i0:i0 + n] for b in BODIES], 1)
    assert np.isfinite(R).all() and (R > 0).all()
    return R


def fit2(Y, TH, F=None, singles=True, pairs=True, per_body_phase=True, fit_end=None,
         seed=7, steps=8000, lr=2e-2, lam=0.01, lam_w=None, lam_p=0.0, tw=None, device="cpu"):
    """Returns (Yh, params) — params = dict(a, p (Tn,12), w66, b). F: (n,12) factor on singles."""
    import torch as T
    T.manual_seed(seed); Tn, n = Y.shape
    fe = n if fit_end is None else fit_end
    lw = lam if lam_w is None else lam_w
    inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))
    # fixed tables (factor folded into the singles tables)
    Fs = np.ones((n, 12)) if F is None else F
    cT = (Fs * np.cos(TH)).astype(np.float32); sT = (Fs * np.sin(TH)).astype(np.float32)      # (n,12)
    D = TH[:, II] - TH[:, KK]
    cD = np.cos(D).astype(np.float32); sD = np.sin(D).astype(np.float32)                       # (n,66)
    cTt = T.tensor(cT.T, device=device); sTt = T.tensor(sT.T, device=device)                   # (12,n)
    cDt = T.tensor(cD.T, device=device); sDt = T.tensor(sD.T, device=device)                   # (66,n)
    IIt = T.tensor(II, device=device); KKt = T.tensor(KK, device=device)
    A0 = np.full((Tn, 12), -2.0, np.float32); A0[:, 9] = inv_sp(Y[:, :fe].mean(1))             # pluto carrier
    Araw = T.tensor(A0, device=device, requires_grad=singles)
    W0 = np.full((Tn, 66), -4.0, np.float32)
    Wraw = T.tensor(W0, device=device, requires_grad=pairs)
    U = T.tensor(np.tile([0.0, 1.0], (Tn, 12, 1)).astype(np.float32) +
                 np.random.RandomState(seed).randn(Tn, 12, 2).astype(np.float32) * 0.01,
                 device=device, requires_grad=True)                                            # (Tn,12,2)
    Bp = T.tensor(Y[:, :fe].mean(1).astype(np.float32), device=device, requires_grad=True)
    params = [U, Bp] + ([Araw] if singles else []) + ([Wraw] if pairs else [])
    Yt = T.tensor(Y.astype(np.float32), device=device)
    opt = T.optim.Adam(params, lr=lr)

    def forward():
        if per_body_phase:
            p = T.atan2(U[:, :, 0], U[:, :, 1])                    # (Tn,12) per-body phases
        else:
            p = T.atan2(U[:, 0, 0], U[:, 0, 1])[:, None].expand(-1, 12)   # ONE shared phase (v1)
        C = Bp[:, None]; S = 0.0
        pen = 0.0
        if singles:
            A = T.nn.functional.softplus(Araw)
            cp = A * T.cos(p); sp = A * T.sin(p)
            # cos(θ-p)=cosθcosp+sinθsinp · sin(θ-p)=sinθcosp-cosθsinp   (factor folded in cT/sT)
            C = C + cp @ cTt + sp @ sTt
            S = S + cp @ sTt - sp @ cTt
            pen = pen + lam * (A ** 2).mean()
        if pairs:
            W = T.nn.functional.softplus(Wraw)
            dlt = p.gather(1, IIt[None].expand(Tn, -1)) - p.gather(1, KKt[None].expand(Tn, -1))  # (Tn,66)
            cw = W * T.cos(dlt); sw = W * T.sin(dlt)
            # cos(D+δ)=cosDcosδ-sinDsinδ · sin(D+δ)=sinDcosδ+cosDsinδ
            C = C + cw @ cDt - sw @ sDt
            S = S + cw @ sDt + sw @ cDt
            pen = pen + lw * (W ** 2).mean()
        if lam_p > 0 and per_body_phase:
            # PHASE COHESION: pull the 12 per-body phases together (circular resultant length);
            # lam_p -> inf recovers the shared-phase model, 0 = free per-body phases.
            Rbar = T.sqrt(T.cos(p).mean(1) ** 2 + T.sin(p).mean(1) ** 2 + 1e-9)
            pen = pen + lam_p * (1.0 - Rbar).mean()
        return T.sqrt(C ** 2 + S ** 2 + 1e-8), pen

    twT = None if tw is None else T.tensor((tw[:fe] / tw[:fe].mean()).astype(np.float32), device=device)
    best, stall, state = np.inf, 0, None
    for it in range(steps):
        yh, pen = forward()
        se = (yh[:, :fe] - Yt[:, :fe]) ** 2
        loss = (se.mean() if twT is None else (se * twT[None]).mean()) + pen
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            l = loss.item()
            if l < best - 1e-7: best, stall, state = l, 0, [x.detach().clone() for x in params]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip(params, state): x.copy_(sv)
        Yh, _ = forward()
        p = T.atan2(U[:, :, 0], U[:, :, 1]) if per_body_phase else \
            T.atan2(U[:, 0, 0], U[:, 0, 1])[:, None].expand(-1, 12)
        out = dict(p=p.cpu().numpy(), b=Bp.cpu().numpy(),
                   a=T.nn.functional.softplus(Araw).cpu().numpy() if singles else None,
                   w=T.nn.functional.softplus(Wraw).cpu().numpy() if pairs else None)
    return Yh.cpu().numpy(), out


def main():
    dev = "cpu"
    try:
        import torch as T
        if T.backends.mps.is_available(): dev = "mps"
    except Exception: pass
    names, Y = ap.load_raw(); Y = Y.astype(np.float64)
    Tn, n = Y.shape; TH = ap.sky12(n); R = dist12(n); WALL = n - 24
    sgn = -np.sign(np.gradient(R, axis=0)); sgn[sgn == 0] = 1.0
    print(f"  PHASOR v2 · per-body phases + 66 aspect weights · {Tn} topics · dev {dev}", flush=True)

    print("\n  == STRUCTURE (budget-matched) ==", flush=True)
    keep = {}
    for tag, kw in [("FULL (singles + pairs + per-body phases)", dict()),
                    ("singles only (no aspect terms)", dict(pairs=False)),
                    ("pairs only (aspects alone)", dict(singles=False)),
                    ("shared phase (v1 dc, for reference)", dict(pairs=False, per_body_phase=False))]:
        Yh, prm = fit2(Y, TH, device=dev, **kw)
        r2 = r2_rows(Y, Yh); keep[tag] = (r2.mean(), prm)
        print(f"  [{tag:44s}] mean R² {r2.mean():+.4f} · median {np.median(r2):+.4f}", flush=True)

    print("\n  == DISTANCE FACTORS on the single-body amplitudes (FULL structure) ==", flush=True)
    facs = [("none (reference)", None),
            ("1/r", 1.0 / R), ("1/r²", 1.0 / R ** 2),
            ("sgn(-ṙ)/r", sgn / R), ("sgn(-ṙ)/r²", sgn / R ** 2), ("sgn(-ṙ) only", sgn.copy())]
    best_tag, best_r2, best_F = None, -1e9, None
    for tag, F in facs:
        Fn = None
        if F is not None:
            Fn = F / np.abs(F).mean(0, keepdims=True)
            Fn[:, [BODIES.index("node"), BODIES.index("lilith")]] = 1.0
        Yh, _ = fit2(Y, TH, F=Fn, device=dev)
        r2 = r2_rows(Y, Yh)
        print(f"  [factor {tag:36s}] mean R² {r2.mean():+.4f} · median {np.median(r2):+.4f}", flush=True)
        if r2.mean() > best_r2: best_tag, best_r2, best_F = tag, r2.mean(), Fn

    print(f"\n  == WALLS (fit<{WALL} · OOS skill last 24) — full structure, best factor [{best_tag}] ==", flush=True)
    mu = Y[:, :WALL].mean(1, keepdims=True)
    den = np.maximum(((Y[:, WALL:] - mu) ** 2).sum(1), 24 / 6.0)
    for tag, F in [("full + best factor", best_F), ("full, no factor", None)]:
        Yh, _ = fit2(Y, TH, F=F, fit_end=WALL, device=dev)
        sk = 1.0 - ((Y[:, WALL:] - Yh[:, WALL:]) ** 2).sum(1) / den
        print(f"  [{tag:44s}] OOS skill median {np.median(sk):+.4f} · >0: {(sk > 0).mean()*100:.1f}%", flush=True)

    print("\n  == FINAL fit (all months, full structure, best factor) ==", flush=True)
    Yh, prm = fit2(Y, TH, F=best_F, device=dev)
    r2 = r2_rows(Y, Yh)
    Yw, _ = fit2(Y, TH, F=best_F, fit_end=WALL, device=dev)
    ins = r2_rows(Y[:, :WALL], Yw[:, :WALL])
    skill = 1.0 - ((Y[:, WALL:] - Yw[:, WALL:]) ** 2).sum(1) / den
    sun = np.rad2deg(prm["p"][:, 0]) % 360
    hist = np.bincount((sun // 30).astype(int), minlength=12)
    print(f"  final mean R² {r2.mean():+.4f} · median {np.median(r2):+.4f} · walls ins {ins.mean():+.4f} · "
          f"OOS skill median {np.median(skill):+.4f} ({(skill > 0).mean()*100:.1f}%>0)", flush=True)
    print("  SUN SIGNS: " + " ".join(f"{SIGNS[i][:3]} {hist[i]}" for i in range(12)), flush=True)
    top_pairs = np.argsort(-prm["w"].mean(0))[:5]
    print("  top aspects: " + ", ".join(f"{BODIES[II[q]]}-{BODIES[KK[q]]} {prm['w'].mean(0)[q]:.2f}" for q in top_pairs), flush=True)
    np.savez_compressed("analysis/adstopics/astro_phasor2_final.npz",
                        a=prm["a"].astype(np.float32), p=prm["p"].astype(np.float32),
                        w=prm["w"].astype(np.float32), b=prm["b"].astype(np.float32),
                        yhat=Yh.astype(np.float32), yhat_w=Yw.astype(np.float32),
                        r2=r2.astype(np.float32), r2_ins=ins.astype(np.float32),
                        r2_oos_skill=skill.astype(np.float32), wall=WALL,
                        factor=best_tag, bodies=np.array(BODIES), names=np.array(names),
                        pairs_i=II, pairs_k=KK)
    print("  saved -> analysis/adstopics/astro_phasor2_final.npz", flush=True)


def export():
    """npz v2 -> atlas/pred: the topic's classification = its SUN SIGN (phase of the Sun's phasor)."""
    import json, pandas as pd
    D = np.load("analysis/adstopics/astro_phasor2_final.npz", allow_pickle=True)
    names = [str(x) for x in D["names"]]
    p, a, w, b = D["p"], D["a"], D["w"], D["b"]
    yhat, yhat_w, wall = D["yhat"], D["yhat_w"], int(D["wall"])
    yhat_dep = D["yhat_dep"] if "yhat_dep" in D else yhat_w
    dep_wall = int(D["dep_wall"]) if "dep_wall" in D else wall
    r2, r2_ins, skill = D["r2"], D["r2_ins"], D["r2_oos_skill"]
    n = yhat.shape[1]; moy = np.arange(n) % 12
    if "tune" in D:
        # EXPLAINABLE SIGN (operator 2026-07-21): the sign IS the peak of the tuning curve — the
        # rotation of the whole constellation at which the fit is best. Classification == the plot.
        phase = (D["tune"].argmax(1) * 5.0).astype(np.float32)
    MN = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
    strict = None
    rows = []
    for j, t in enumerate(names):
        # sign = the TUNING-CURVE PEAK when available (classification == the demonstration plot);
        # legacy fallback: the fitted Sun phase.
        sun = float(phase[j]) if "tune" in D else float(np.rad2deg(p[j, 0]) % 360)
        prof = np.array([yhat[j][moy == m].mean() for m in range(12)])
        amp = a[j] / max(a[j].sum(), 1e-9); k = int(a[j].argmax())
        q = int(w[j].argmax()) if w[j].max() > 1e-6 else -1
        if "asp_i" in D:
            top_asp = f"{BODIES[int(D['asp_i'][j])]}-{BODIES[int(D['asp_k'][j])]}"
        else:
            top_asp = f"{BODIES[int(D['pairs_i'][q])]}-{BODIES[int(D['pairs_k'][q])]}" if q >= 0 else ""
        rows.append({"topic": t, "sign": SIGNS[int(sun // 30)], "phase": round(sun, 1),
                     "peak_month": MN[int(prof.argmax())], "dominant_planet": BODIES[k],
                     "dom_share": round(float(amp[k]), 3),
                     "sun_share": round(float(amp[BODIES.index("sun")]), 3),
                     "top_aspect": top_asp,
                     "phase_stable": 1,
                     "r2_full": round(float(r2[j]), 4), "r2_ins": round(float(r2_ins[j]), 4),
                     "r2_oos": round(float(skill[j]), 4), "r2_oos_strict": round(float(skill[j]), 4)})
    pd.DataFrame(rows).to_csv("analysis/adstopics/astro_ts_atlas.csv", index=False)
    _TH_LAST = ap.sky12(n)[-1]                                   # latest-month longitudes (rad)
    # AGGREGATE benchmark analysis for the index page: pooled skill per months-ahead + AUC.
    _, Yraw = ap.load_raw(); Yraw = Yraw.astype(float)
    ndat = Yraw.shape[1]
    muW = Yraw[:, :wall].mean(1, keepdims=True)
    skc = []
    for h in range(ndat - wall):
        e = Yraw[:, wall + h] - yhat_w[:, wall + h]
        bl = Yraw[:, wall + h] - muW[:, 0]
        skc.append(round(float(1.0 - (e ** 2).sum() / max((bl ** 2).sum(), 1e-9)), 4))
    agg = {"skillCurve": skc, "auc": round(float(np.mean(skc)), 4),
           "skillMedian": round(float(np.median(skill)), 4),
           "pctPositive": round(float((skill > 0).mean() * 100), 1), "topics": len(names)}
    reg = json.load(open("analysis/_topics_weekly.json"))
    key_of = {r["label"]: k for k, r in reg.items()}
    pred = {"_meta": {"task": "phasor-r2", "wall": wall, "dep_wall": dep_wall, "aggregate": agg, "ext": int(D["ext"]) if "ext" in D else 0,
                      "model": "v2: |b+Σa·e^(i(θ-p_i))+Σw·e^(i(θi-θk+pi-pk))| · per-body phases · sun-sign classification"}}
    m = 0
    for j, t in enumerate(names):
        k = key_of.get(t)
        if not k: continue
        pred[k] = {"pred": [round(float(v), 1) for v in yhat_dep[j]],
                   "bench": [round(float(v), 1) for v in yhat_w[j]],
                   "r2Ins": round(float(r2_ins[j]), 4), "r2Oos": round(float(skill[j]), 4)}
        if "tune" in D:
            pred[k]["sunCurve"] = [round(float(v), 4) for v in D["tune"][j]]
        # RECEIVER INTERNALS for the page: the twelve arrow lengths (amplitude shares) and the
        # topic's strongest aspects by interference weight 2·a_i·a_k (relative to the strongest).
        sh = a[j] / max(float(a[j].sum()), 1e-9)
        pred[k]["bodyShares"] = [round(float(v), 3) for v in sh]
        pred[k]["bodyPhases"] = [round(float(np.degrees(p[j, i_]) % 360.0), 1) for i_ in range(12)]
        bj_safe = max(float(b[j]), 1e-6)
        pw = a[j][II] * a[j][KK] / bj_safe                      # magnitude model: pair term ~ a_i·a_k/b
        top = np.argsort(-pw)[:3]; mx = max(float(pw[top[0]]), 1e-9)
        # NUMERICAL aspect demonstration (operator): for each top pair — the weight 2·a_i·a_k (why
        # prominent), the pair's CURRENT separation relative to this topic's tuning offset
        # eff = (θ_i−θ_k) − (p_i−p_k) at the latest month, and its current contribution to the
        # topic's power, as % of the topic's mean level.
        TH_l = _TH_LAST                                             # (12,) latest-month longitudes
        ybar = max(float(yhat[j].mean()), 1e-9)
        asps = []
        for q in top:
            i_, k_ = int(II[q]), int(KK[q])
            sep = float(np.degrees(np.angle(np.exp(1j * (TH_l[i_] - TH_l[k_]))))) % 360.0
            tun = float(np.degrees(np.angle(np.exp(1j * (p[j, i_] - p[j, k_]))))) % 360.0
            eff = (sep - tun) % 360.0
            cosv = float(np.cos(np.radians(eff)))
            con = 100.0 * float(pw[q]) * cosv / ybar
            asps.append({"a": BODIES[i_], "b": BODIES[k_], "rel": round(float(pw[q] / mx), 2),
                         "ai": round(float(a[j, i_]), 2), "ak": round(float(a[j, k_]), 2),
                         "w": round(float(pw[q]), 2), "sep": round(sep, 1), "tun": round(tun, 1),
                         "eff": round(eff, 1), "cos": round(cosv, 2), "con": round(con, 1)})
        pred[k]["topAspects"] = asps
        # TRANSIT-TO-NATAL channel (the homodyne term 2·b·a_i·cos(θ_i−p_i)): each planet against
        # its OWN tuning, top 3 by present effect — astrology's "transits to natal points".
        trs = []
        for i_ in range(12):
            th_now = float(np.degrees(TH_l[i_])) % 360.0
            tu = float(np.degrees(p[j, i_])) % 360.0
            ang = (th_now - tu) % 360.0
            con = 100.0 * float(a[j, i_]) * float(np.cos(np.radians(ang))) / ybar     # magnitude model, first order
            trs.append({"b": BODIES[i_], "th": round(th_now, 1), "tu": round(tu, 1),
                        "ang": round(ang, 1), "con": round(con, 1)})
        trs.sort(key=lambda r: -abs(r["con"]))
        pred[k]["transits"] = trs[:3]
        m += 1
    json.dump(pred, open("analysis/adstopics/astro_ts_pred.json", "w"))
    print(f"  [export v2] atlas {len(rows)} rows (SUN-SIGN) · pred {m}/{len(names)} · wall {wall}", flush=True)


def final(lam_p, hl=12, gamma=0.90):
    """FINAL cohesion model + FORECAST RECIPE (operator 2026-07-21, tested +0.6065/76.0% vs +0.5046):
    per-body phases (cohesion lam_p), 1/r(t) breathing, RECENCY-weighted fit (half-life hl months,
    anchored at each fit's end) and the walls forecast damped gamma toward the trained mean.
    Sun phase = the topic's SUN SIGN."""
    dev = "cpu"
    try:
        import torch as T
        if T.backends.mps.is_available(): dev = "mps"
    except Exception: pass
    names, Y = ap.load_raw(); Y = Y.astype(np.float64)
    Tn, n = Y.shape; TH = ap.sky12(n); WALL = n - 24
    R = dist12(n)
    F = 1.0 / R; F = F / np.abs(F).mean(0, keepdims=True)
    F[:, [BODIES.index("node"), BODIES.index("lilith")]] = 1.0     # massless points: no distance breathing
    print(f"  FINAL cohesion model · lam_p {lam_p} · 1/r breathing · recency hl={hl}mo · gamma {gamma} · dev {dev}", flush=True)
    tw_full = 0.5 ** ((n - 1 - np.arange(n)) / hl)
    tw_wall = 0.5 ** ((WALL - 1 - np.arange(n)) / hl)
    Yh, prm = fit2(Y, TH, F=F, pairs=False, lam_p=lam_p, tw=tw_full, device=dev)
    r2 = r2_rows(Y, Yh)
    Yw, _ = fit2(Y, TH, F=F, pairs=False, lam_p=lam_p, fit_end=WALL, tw=tw_wall, device=dev)
    mu_sh = Y[:, :WALL].mean(1, keepdims=True)
    Yw[:, WALL:] = gamma * Yw[:, WALL:] + (1 - gamma) * mu_sh      # damped forecast (the shipped curve)
    ins = r2_rows(Y[:, :WALL], Yw[:, :WALL])
    mu = Y[:, :WALL].mean(1, keepdims=True)
    den = np.maximum(((Y[:, WALL:] - mu) ** 2).sum(1), 24 / 6.0)
    skill = 1.0 - ((Y[:, WALL:] - Yw[:, WALL:]) ** 2).sum(1) / den
    sun = np.rad2deg(prm["p"][:, 0]) % 360
    hist = np.bincount((sun // 30).astype(int), minlength=12)
    print(f"  final mean R² {r2.mean():+.4f} · median {np.median(r2):+.4f} · walls ins {ins.mean():+.4f} · "
          f"OOS skill median {np.median(skill):+.4f} ({(skill > 0).mean()*100:.1f}%>0)", flush=True)
    print("  SUN SIGNS: " + " ".join(f"{SIGNS[i][:3]} {hist[i]}" for i in range(12)), flush=True)
    np.savez_compressed("analysis/adstopics/astro_phasor2_final.npz",
                        a=prm["a"].astype(np.float32), p=prm["p"].astype(np.float32),
                        w=np.zeros((Tn, 66), np.float32), b=prm["b"].astype(np.float32),
                        yhat=Yh.astype(np.float32), yhat_w=Yw.astype(np.float32),
                        r2=r2.astype(np.float32), r2_ins=ins.astype(np.float32),
                        r2_oos_skill=skill.astype(np.float32), wall=WALL,
                        factor=f"cohesion lam_p={lam_p} · 1/r", bodies=np.array(BODIES), names=np.array(names),
                        pairs_i=II, pairs_k=KK)
    print("  saved -> analysis/adstopics/astro_phasor2_final.npz", flush=True)


if __name__ == "__main__":
    if "--export" in sys.argv:
        export()
    elif "--final" in sys.argv:
        final(float(sys.argv[sys.argv.index("--final") + 1]))
    else:
        main()
