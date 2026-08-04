#!/usr/bin/env python3
"""PHASOR model, FINAL (operator 2026-07-20): |b_j + sum_i a_ij * e^(1j(theta_i(t) - p_j))| directly
predicts the RAW data — no balancing, no labels, no thresholds anywhere in this pipeline.

Each body is a complex phasor of amplitude a_ij at its longitude; the score is the MAGNITUDE of the
interference sum. All parameters fitted by gradient descent (Adam, full batch) on ALL 210 months —
no validation set. a_ij >= 0 via softplus (phasor amplitudes; NNLS lineage). No wrap needed — the
complex exponential is periodic by construction.

MATH NOTE: in a pure magnitude the topic phase CANCELS — |e^{-1j p_j} * S| = |S| — so in the literal
model p_j is unidentifiable (zero gradient; kept as parameters for spec fidelity). The squared
magnitude is a pure ASPECTS model: |S|^2 = sum a^2 + 2*sum_{i<k} a_i a_k cos(theta_i - theta_k) —
pairwise planet separations (conjunction constructive, opposition destructive).
Mode "dc" adds a real b_j INSIDE the modulus (y = |b_j + sum_i a_ij e^{1j(theta_i - p_j)}|): a fixed
reference phasor the constellation rotates against — p_j becomes identifiable.

  python3 analysis/adstopics/astro_phasor.py            (fits literal + dc, saves the artifact)
"""
import importlib.util as u, json, os
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
tf = _load("analysis/trends_fit.py", "tf")

BODIES = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "node", "chiron"]
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]


def load_raw():
    """RAW series only — no labels, no thresholds, no balancing. Same filters/order as the audited
    campaign loader, so topic order matches every artifact."""
    grid = pd.DatetimeIndex(tf.GRID)
    i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
    iend = len(grid) - tf.DROP_LAST
    vocab = json.load(open("analysis/adstopics/vocabulary.json"))
    bl = set(json.load(open("analysis/adstopics/blacklist.json")).get("excluded_topics", []))
    names, Ys = [], []
    for t in sorted(vocab):
        if t in bl: continue
        pth = f"analysis/adstopics/series/{tf.slug(t)}.csv"
        if not os.path.exists(pth): continue
        df = pd.read_csv(pth); df["Time"] = pd.to_datetime(df["Time"])
        v = pd.to_numeric(df.drop_duplicates("Time").set_index("Time")["v"]
                          .reindex(grid[i0:iend]), errors="coerce")
        if v.notna().sum() < (iend - i0) * 0.5: continue
        y = v.interpolate(limit_direction="both").to_numpy(float)
        if not np.isfinite(y).all() or y.max() <= 0: continue
        if y.mean() < 10.0: continue                    # POPULARITY FILTER (operator 2026-07-21): >=10% mean interest only
        names.append(t); Ys.append(y)
    return names, np.stack(Ys)


def sky12(n):
    E = pd.read_csv("analysis/_ephemeris_extended.csv")
    grid = pd.DatetimeIndex(pd.to_datetime(E["Time"]))
    i0 = int(np.where(grid == pd.Timestamp("2008-01-01"))[0][0])
    TH = np.stack([np.deg2rad(E[f"{b}_lon"].to_numpy(float)[i0:i0 + n]) for b in BODIES], 1)
    assert not np.isnan(TH).any()
    return TH


def r2_rows(Y, Yh):
    """SS_tot floored at the integer-quantization variance (n/6) — a perfectly flat segment has no
    explainable variance, and an unfloored denominator turns tiny errors into R2 ~ -1e9 artifacts."""
    res = ((Y - Yh) ** 2).sum(1); tot = np.maximum(((Y - Y.mean(1, keepdims=True)) ** 2).sum(1), Y.shape[1] / 6.0)
    return 1.0 - res / tot


def r2_xsec(Y, Yh):
    res = ((Y - Yh) ** 2).sum(0); tot = ((Y - Y.mean(0, keepdims=True)) ** 2).sum(0) + 1e-9
    return 1.0 - res / tot


def fit(Y, TH, mode="literal", fit_end=None, seed=7, steps=12000, lr=2e-2, lam=0.01, device="cpu"):
    """fit_end: fit the loss on months < fit_end only (walls run); predictions are full-length."""
    import torch as T
    T.manual_seed(seed); Tn, n = Y.shape
    fe = n if fit_end is None else fit_end
    inv_sp = lambda v: np.log(np.expm1(np.clip(v, 1e-3, None)))            # softplus inverse
    A0 = np.full((Tn, 12), -2.0, np.float32)
    A0[:, BODIES.index("pluto")] = inv_sp(Y[:, :fe].mean(1))            # slowest body starts as carrier
    Araw = T.tensor(A0, device=device, requires_grad=True)
    U = T.tensor(np.tile([0.0, 1.0], (Tn, 1)).astype(np.float32), device=device, requires_grad=True)
    params = [Araw, U]
    if mode == "dc":
        Bp = T.tensor(Y[:, :fe].mean(1).astype(np.float32), device=device, requires_grad=True)
        params.append(Bp)
    THd = T.tensor(TH.T.astype(np.float32), device=device)                 # (12, n)
    Yt = T.tensor(Y.astype(np.float32), device=device)
    opt = T.optim.Adam(params, lr=lr)

    def forward():
        A = T.nn.functional.softplus(Araw)                                 # a_ij >= 0
        p = T.atan2(U[:, 0], U[:, 1])
        ang = THd[None] - p[:, None, None]                                 # (Tn,12,n)
        C = (A[:, :, None] * T.cos(ang)).sum(1); S = (A[:, :, None] * T.sin(ang)).sum(1)
        if mode == "dc": C = C + Bp[:, None]                               # real reference phasor
        return T.sqrt(C ** 2 + S ** 2 + 1e-8), A, p

    best, stall, state = np.inf, 0, None
    for it in range(steps):
        yh, Acur, _ = forward()
        # L2 amplitude penalty: huge opposing phasors can cancel INSIDE the fit window and diverge
        # beyond it — weight decay keeps the interference bounded so extrapolation stays sane.
        loss = ((yh[:, :fe] - Yt[:, :fe]) ** 2).mean() + lam * (Acur ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            l = loss.item()
            if l < best - 1e-7: best, stall, state = l, 0, [x.detach().clone() for x in params]
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        for x, sv in zip(params, state): x.copy_(sv)
        Yh, A, p = forward()
    bv = Bp.detach().cpu().numpy() if mode == "dc" else np.zeros(Tn, np.float32)
    return Yh.cpu().numpy(), A.cpu().numpy(), p.cpu().numpy(), bv


def main():
    dev = "cpu"
    try:
        import torch as T
        if T.backends.mps.is_available(): dev = "mps"
    except Exception: pass
    names, Y = load_raw(); Y = Y.astype(np.float64)
    Tn, n = Y.shape; TH = sky12(n)
    print(f"  PHASOR y_j(t)=|Σ_i a_ij·e^(1j(θ_i(t)-p_j))| · a>=0 · all params gradient descent · "
          f"fit on ALL {n} months · {Tn} topics · dev {dev}", flush=True)
    WALL = n - 24                                                          # 186: last 24 months held out
    keep = {}
    for mode, fe in (("literal", None), ("dc", None), ("dc", WALL)):
        Yh, A, p, bv = fit(Y, TH, mode=mode, fit_end=fe, device=dev)
        key = mode if fe is None else "walls"
        keep[key] = (A, p, Yh, bv)
        if fe is None:
            pt = r2_rows(Y, Yh); xs = r2_xsec(Y, Yh); keep[key] = (A, p, Yh, bv, pt)
            tag = "literal (per spec; p_j cancels in |·|)" if mode == "literal" else "dc      (+b_j inside |·|, p_j active)"
            mass = A.mean(0); topm = sorted(zip(BODIES, mass), key=lambda x: -x[1])
            print(f"    [{tag}] R² per-topic mean {pt.mean():+.4f} · median {np.median(pt):+.4f} · "
                  f"xsec {xs.mean():+.4f} · R²>0: {(pt > 0).mean()*100:.1f}%", flush=True)
            print(f"        amplitude mass: {', '.join(f'{k} {v:.2f}' for k, v in topm[:6])}", flush=True)
        else:
            ins = r2_rows(Y[:, :WALL], Yh[:, :WALL]); oos = r2_rows(Y[:, WALL:], Yh[:, WALL:])
            mu_tr = Y[:, :WALL].mean(1, keepdims=True)
            skill = 1.0 - ((Y[:, WALL:] - Yh[:, WALL:]) ** 2).sum(1) / np.maximum(
                ((Y[:, WALL:] - mu_tr) ** 2).sum(1), (n - WALL) / 6.0)
            print(f"    [dc WALLS fit<{WALL} · OOS last {n - WALL} months] in-sample R² mean {ins.mean():+.4f} · "
                  f"median {np.median(ins):+.4f} · OOS strict mean {oos.mean():+.4f} · median {np.median(oos):+.4f} "
                  f"(>0: {(oos > 0).mean()*100:.1f}%) · OOS skill median {np.median(skill):+.4f} "
                  f"(>0: {(skill > 0).mean()*100:.1f}%)", flush=True)
    A, p, Yh_l, _, pt = keep["literal"]; Ad, pd_, Yh_d, bd, ptd = keep["dc"]; Aw, pw, Yh_w, bw = keep["walls"]
    ins = r2_rows(Y[:, :WALL], Yh_w[:, :WALL]); oos = r2_rows(Y[:, WALL:], Yh_w[:, WALL:])
    mu_tr = Y[:, :WALL].mean(1, keepdims=True)
    skill = 1.0 - ((Y[:, WALL:] - Yh_w[:, WALL:]) ** 2).sum(1) / np.maximum(
        ((Y[:, WALL:] - mu_tr) ** 2).sum(1), (n - WALL) / 6.0)
    out = "analysis/adstopics/astro_phasor_final.npz"
    np.savez_compressed(out, a=A.astype(np.float32), p=p.astype(np.float32), r2=pt.astype(np.float32),
                        a_dc=Ad.astype(np.float32), p_dc=pd_.astype(np.float32), r2_dc=ptd.astype(np.float32),
                        yhat_dc=Yh_d.astype(np.float32),
                        b_dc=bd.astype(np.float32), b_w=bw.astype(np.float32),
                        a_w=Aw.astype(np.float32), p_w=pw.astype(np.float32), yhat_w=Yh_w.astype(np.float32),
                        r2_ins=ins.astype(np.float32), r2_oos=oos.astype(np.float32),
                        r2_oos_skill=skill.astype(np.float32), wall=WALL,
                        bodies=np.array(BODIES), names=np.array(names))
    print(f"    saved -> {out}", flush=True)


def export():
    """npz -> the prod pipeline inputs: astro_ts_atlas.csv (per-topic card fields + R²) and
    astro_ts_pred.json (the full in+out-of-sample monthly prediction per topic, wall index)."""
    import json, pandas as pd
    D = np.load("analysis/adstopics/astro_phasor_final.npz", allow_pickle=True)
    names = [str(x) for x in D["names"]]; bodies = [str(x) for x in D["bodies"]]
    a_dc, p_dc, r2_dc = D["a_dc"], D["p_dc"], D["r2_dc"]
    yhat_dc, yhat_w, p_w = D["yhat_dc"], D["yhat_w"], D["p_w"]
    r2_ins, r2_oos, wall = D["r2_ins"], D["r2_oos"], int(D["wall"])
    r2_skill = D["r2_oos_skill"] if "r2_oos_skill" in D else r2_oos
    n = D["yhat_dc"].shape[1]
    moy = np.arange(n) % 12                                # grid starts 2008-01 -> 0 = January
    MN = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
    rows = []
    for j, t in enumerate(names):
        deg = float(np.rad2deg(p_dc[j]) % 360)
        prof = np.array([yhat_dc[j][moy == m].mean() for m in range(12)])       # model month-of-year profile
        dphi = np.angle(np.exp(1j * (p_dc[j] - p_w[j])))
        share = a_dc[j] / max(a_dc[j].sum(), 1e-9)
        k = int(a_dc[j].argmax())
        rows.append({"topic": t, "sign": SIGNS[int(deg // 30)], "phase": round(deg, 1),
                     "peak_month": MN[int(prof.argmax())], "dominant_planet": bodies[k],
                     "dom_share": round(float(share[k]), 3),
                     "sun_share": round(float(share[bodies.index("sun")]), 3),
                     "phase_stable": int(abs(np.rad2deg(dphi)) < 30),
                     "r2_full": round(float(r2_dc[j]), 4), "r2_ins": round(float(r2_ins[j]), 4),
                     "r2_oos": round(float(r2_skill[j]), 4), "r2_oos_strict": round(float(r2_oos[j]), 4)})
    pd.DataFrame(rows).to_csv("analysis/adstopics/astro_ts_atlas.csv", index=False)
    reg = json.load(open("analysis/_topics_weekly.json"))
    key_of = {r["label"]: k for k, r in reg.items()}
    pred = {"_meta": {"task": "phasor-r2", "wall": wall,
                      "model": "y_j(t)=|b_j+Σ_i a_ij·e^(i(θ_i(t)-p_j))| · a>=0 · gradient descent · fit<186, last 24 months out-of-sample"}}
    matched = 0
    for j, t in enumerate(names):
        k = key_of.get(t)
        if not k: continue
        pred[k] = {"pred": [round(float(v), 1) for v in yhat_w[j]],
                   "r2Ins": round(float(r2_ins[j]), 4), "r2Oos": round(float(r2_skill[j]), 4),
                   "r2OosStrict": round(float(r2_oos[j]), 4)}
        matched += 1
    json.dump(pred, open("analysis/adstopics/astro_ts_pred.json", "w"))
    print(f"  [export] atlas {len(rows)} rows -> astro_ts_atlas.csv · pred {matched}/{len(names)} keyed -> astro_ts_pred.json (wall {wall})", flush=True)


if __name__ == "__main__":
    import sys
    export() if "--export" in sys.argv else main()
