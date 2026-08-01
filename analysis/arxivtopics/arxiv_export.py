#!/usr/bin/env python3
"""ARXIV EXPORT (operator 2026-07-21): the fitted weekly-citation-share models -> the /topics page.

Reads arxiv_phasor_final.npz (+ weekly_counts.csv for the actual shares) and REPLACES the SPA's
research atlas: src/data/research.json topics/aggregate become the 129 arXiv categories, and
src/data/field-daily/ holds one weekly chunk per category (actual share, deploy prediction to
today, benchmark curve, tuning sweep, receiver internals, aspects, transits). Non-topic top-level
keys of research.json (cycles, articles, ...) are carried over so /cycles and resonance keep working.

  python3 analysis/arxivtopics/arxiv_export.py
"""
import importlib.util as u, json, os, shutil
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
p2 = _load("analysis/adstopics/astro_phasor2.py", "p2")
af = _load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "arxiv_fit.py"), "af")
BODIES, SIGNS = af.BODIES, p2.SIGNS                       # SEVEN bodies (mars+outer; sun+inner+chiron excluded)
MN = ["January", "February", "March", "April", "May", "June",
      "July", "August", "September", "October", "November", "December"]
# Orbital / cycle periods in years (resonance tooltips: the dominant body's rhythm).
PERIOD_Y = {"sun": 1.0, "moon": 0.075, "mercury": 0.24, "venus": 0.62, "mars": 1.88,
            "jupiter": 11.86, "saturn": 29.46, "uranus": 84.0, "neptune": 164.8,
            "pluto": 248.0, "node": 18.6, "chiron": 50.4}
import re as _re
def slug(c):
    return _re.sub(r"-+", "-", _re.sub(r"[^a-z0-9]+", "-", str(c).lower())).strip("-")


def main():
    D = np.load("analysis/arxivtopics/arxiv_phasor_final.npz", allow_pickle=True)
    names = [str(x) for x in D["names"]]
    A, P, B = D["a"].astype(float), D["p"].astype(float), D["b"].astype(float)
    Yd, Yw = D["yhat_dep"].astype(float), D["yhat_w"].astype(float)
    r2, trm, skill = D["r2"].astype(float), D["r2_ins"].astype(float), D["r2_oos_skill"].astype(float)
    curve = [round(float(v), 4) for v in D["skill_curve"]]
    WALL, DEP, EXT = int(D["wall"]), int(D["dep_wall"]), int(D["ext"])
    weeks_ext = [str(w) for w in D["dates"]]
    Tn = len(names); ne = len(weeks_ext); n = ne - EXT
    wk = weeks_ext[:n]

    # The ACTUAL lunation shares — the fit's own loader (one source of truth).
    lnames, Y, labels, _future = af.load_lunar()
    assert lnames == names and labels == wk, "lunation axis mismatch vs npz"

    # Sky (lunar, extended) + the full moon nearest today for the transit card.
    E = pd.read_csv("analysis/arxivtopics/_ephemeris_yearly.csv").set_index("Time"); E.index = E.index.astype(str)
    TH = np.stack([np.deg2rad(E[f"{b}_lon"].loc[weeks_ext].to_numpy(float)) for b in BODIES], 1)
    R = np.stack([E[f"{b}_dist"].loc[weeks_ext].to_numpy(float) for b in BODIES], 1)
    F = np.ones_like(R)                                              # NO distance factor (matches the fit)
    F[:, [BODIES.index("node")]] = 1.0
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    i_now = int(np.argmin([abs(int(w) - pd.Timestamp(today).year) for w in weeks_ext]))   # nearest year to now
    TH_now = TH[i_now]

    # SIGN := the PLUTO SIGN (operator 2026-07-23): every topic is classified by the sign its fitted
    # PLUTO tuning p_pluto falls in — regardless of the weight distribution. Pluto is the atlas's
    # dominant body (the deepest, slowest rotor); its tuning direction is the classification.
    # ROUND BEFORE FLOORING: tunings come off a 1° sweep, so every one is an exact integer degree (or
    # that + 180). Without the round, a topic tuned to exactly 240° lands at 239.9999999 and floors into
    # the previous season — the label flips on floating-point dust rather than on the model.
    Pdeg = np.round(np.degrees(P) % 360.0, 6) % 360.0          # (Tn, nb) tunings, degrees
    sign_of = (Pdeg // 30).astype(int) % 12
    PLUTO = BODIES.index("pluto")
    phase = Pdeg[:, PLUTO].astype(float)                      # the representative degree = pluto's tuning
    dom_sign = (phase // 30).astype(int) % 12
    signW = np.stack([np.where(sign_of == s, A, 0.0).sum(1) for s in range(12)], 1)
    signW_frac = signW / np.maximum(signW.sum(1, keepdims=True), 1e-9)               # kept as context
    dom_sign_share = signW_frac[np.arange(Tn), dom_sign]      # how much weight sits in the pluto sign

    II, KK = np.triu_indices(len(BODIES), 1)
    web = "artaquest-web"
    old = json.load(open(f"{web}/src/data/research.json"))

    topics, chunks = [], {}
    for j, t in enumerate(names):
        k = slug(t)
        amp = A[j] / max(float(A[j].sum()), 1e-9); kb = int(A[j].argmax())
        yv = Y[j]
        # NO WITHIN-YEAR SEASON EXISTS at yearly cadence, so there is nothing to measure here. The old
        # quarterly code averaged the series into twelve calendar buckets; under `moy = zeros(...)` that
        # left eleven of the twelve empty, so `prof` was [mean, nan × 11] — and `prof.argmax()` returns
        # the first NaN, which shipped "peaks in February" for all 251 topics and a variance share of
        # exactly 0.0 for all of them. State the truth instead of computing a degenerate number.
        annual = 0.0                       # share of variance explained by a within-year cycle: none
        peak_month = None                  # undefined at this cadence (not read by the page either)
        vq = np.where(af.META.get("valid", np.ones(len(yv), bool)))[0]
        first, last = float(yv[vq[:8]].mean()), float(yv[vq[-8:]].mean())
        tdir = "rising" if last > first * 1.15 else "falling" if last < first * 0.85 else "steady"
        ybar = max(float(Yd[j, :n].mean()), 1e-9)
        pw = 2.0 * A[j][II] * A[j][KK]                    # pair COUPLING 2aᵢaₖ (the most a pair can add)
        top = np.argsort(-pw)[:3]; mx = max(float(pw[top[0]]), 1e-9)
        # RECTIFIED cosine-sum: y = max(C,0)², C = b + Σaᵢcos(θᵢ−pᵢ). Where C>0 (the lit regime that
        # carries the signal) C² expands EXACTLY into baseline + transits (2b·aᵢcosφᵢ) + pair terms
        # (2aᵢaₖ·cosφᵢ·cosφₖ — two planets BOTH lit past the topic's tuning multiply up). % of level = /ȳ.
        phi = np.angle(np.exp(1j * (TH_now - P[j])))      # each body's current projection angle φᵢ=θᵢ−pᵢ
        asps = []
        for q in top:
            i_, k_ = int(II[q]), int(KK[q])
            sep = float(np.degrees(np.angle(np.exp(1j * (TH_now[i_] - TH_now[k_]))))) % 360.0
            tun = float(np.degrees(np.angle(np.exp(1j * (P[j, i_] - P[j, k_]))))) % 360.0
            eff = (sep - tun) % 360.0                     # classical aspect angle φᵢ−φₖ (context)
            cosv = float(np.cos(phi[i_]) * np.cos(phi[k_]))          # JOINT projection cosφᵢ·cosφₖ (exact)
            asps.append({"a": BODIES[i_], "b": BODIES[k_], "rel": round(float(pw[q] / mx), 2),
                         "ai": round(float(A[j, i_]), 2), "ak": round(float(A[j, k_]), 2),
                         "w": round(float(pw[q]), 2), "sep": round(sep, 1), "tun": round(tun, 1),
                         "eff": round(eff, 1), "pi": round(float(np.cos(phi[i_])), 2),
                         "pk": round(float(np.cos(phi[k_])), 2), "cos": round(cosv, 2),
                         "con": round(100.0 * float(pw[q]) * cosv / ybar, 1)})
        trs = []
        for i_ in range(len(BODIES)):
            th_now = float(np.degrees(TH_now[i_])) % 360.0
            tu = float(np.degrees(P[j, i_])) % 360.0
            ang = (th_now - tu) % 360.0
            trs.append({"b": BODIES[i_], "th": round(th_now, 1), "tu": round(tu, 1), "ang": round(ang, 1),
                        "con": round(100.0 * 2.0 * float(B[j]) * float(A[j, i_]) *
                                     float(np.cos(np.radians(ang))) / ybar, 1)})
        trs.sort(key=lambda r: -abs(r["con"]))
        qa = int((2 * A[j][II] * A[j][KK]).argmax())
        topics.append({
            "key": k, "label": t, "sign": SIGNS[int(dom_sign[j])],
            "ref": round(phase[j], 1), "axis": "topic", "system": "Citations",
            "domain": af.META["domain"].get(t, ""), "fieldGroup": af.META["field"].get(t, ""),
            "r2": round(float(np.clip(r2[j], 0, 1)), 3), "rep": round(float(np.clip(r2[j], 0, 1)) * 100, 1),
            "popularity": round(float(yv[wk.index(af.META["start"][t]):].mean()), 2), "period": PERIOD_Y[BODIES[kb]],
            "shares": [{"slug": b, "pct": round(float(amp[i_]) * 100, 1)}
                       for i_, b in enumerate(BODIES)],
            "trend": annual < 0.2, "trendDir": tdir, "annualFrac": round(annual, 3),
            "season": {"month": peak_month, "peak": round(phase[j], 1)},
            "phaseEst": round(phase[j], 1), "phaseSign": SIGNS[int(dom_sign[j])],
            "domSignShare": round(float(dom_sign_share[j]), 3),
            "peakMonth": peak_month, "dominantPlanet": BODIES[kb],
            "domShare": round(float(amp[kb]), 3),
            "topAspect": f"{BODIES[int(II[qa])]}-{BODIES[int(KK[qa])]}",
            "r2Full": round(float(r2[j]), 4), "r2Ins": round(float(trm[j]), 4), "r2Oos": round(float(skill[j]), 4),
        })
        off = wk.index(af.META["start"][t])                          # this topic's consistently-non-zero start
        chunks[k] = {
            "actual": [round(float(v), 3) for v in yv[off:]],         # cropped to the topic's own history
            "fit": [], "err": [], "forecast": [],
            "start": int(pd.Timestamp(int(wk[off]), 7, 2).timestamp()), "step": 31557600, "from": wk[off], "to": "2025",
            "pred": [round(float(v), 3) for v in Yd[j, off:]], "predStart": 0, "predWall": DEP - off,
            "bench": [round(float(v), 3) for v in Yw[j, off:n]], "benchWall": WALL - off,
            "r2Ins": round(float(trm[j]), 4), "r2Oos": round(float(skill[j]), 4),
            "bodyShares": [round(float(v), 3) for v in amp],
            "bodyPhases": [round(float(np.degrees(P[j, i_]) % 360.0), 1) for i_ in range(len(BODIES))],
            "topAspects": asps, "transits": trs[:3],
        }

    out = dict(old)
    out["n"] = Tn
    out["topics"] = topics
    out["dataBase"] = "openalex-citations-received-yearly-1700-2025"
    out["aggregate"] = {"skillCurve": curve, "auc": round(float(np.mean(curve)), 4),
                        "skill2": round(float(D["skill2"]), 4) if "skill2" in D else None,
                        "auc2": round(float(D["auc2"]), 4) if "auc2" in D else None,
                        "skill4": round(float(D["skill4"]), 4) if "skill4" in D else None,
                        "auc4": round(float(D["auc4"]), 4) if "auc4" in D else None,
                        "skillMedian": round(float(np.median(skill)), 4),
                        "pctPositive": round(float((skill > 0).mean() * 100), 1), "topics": Tn,
                        # the harder bar and the model's size, MEASURED and carried through so the page
                        # never hardcodes a number the model can quietly drift away from
                        "persistence": round(float(D["persistence"]), 4) if "persistence" in D else None,
                        "paramsPerTopic": int(D["params_per_topic"]) if "params_per_topic" in D else None}
    json.dump(out, open(f"{web}/src/data/research.json", "w"))
    dd = f"{web}/src/data/field-daily"
    shutil.rmtree(dd); os.makedirs(dd)
    for k, c in chunks.items():
        json.dump(c, open(f"{dd}/{k}.json", "w"))
    sz = sum(os.path.getsize(f"{dd}/{f}") for f in os.listdir(dd)) // 1024
    print(f"  [arxiv export] research.json topics {Tn} · chunks {len(chunks)} ({sz} KB) · "
          f"transit year {weeks_ext[i_now]} · dep_wall {DEP} · bench_wall {WALL} · ext {EXT}", flush=True)
    print("ARXIVEXPORT DONE", flush=True)


if __name__ == "__main__":
    main()
