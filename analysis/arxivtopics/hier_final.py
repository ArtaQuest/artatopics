#!/usr/bin/env python3
"""THE FRONTIER -- one hierarchical family that contains both results, measured end to end.

Two things came out of the pooling campaign and they are NOT the same thing:

  * dict_spectrum.py  -- a genuinely GLOBAL model. A topic owns a level, a tuning, a signed gain and
    a POINTER into K shared spectra: 4 params/topic + 7K global. At K=12 that is 1,088 numbers
    against the per-topic baseline's 2,259.
  * pool_refit.py     -- partial pooling toward a FIELD-level signed spectrum. Better AUC than the
    baseline, but at essentially unchanged parameter count.

They are the two ends of ONE family. Take the dictionary fit's implied arrows t_j = s_j*g_{k(j)} as
the prior mean and let each topic buy its way back out at price tau:

    minimise  data + anchor + tau*||a_j - t_j||^2

tau=0 is the 251 free fits (9 params/topic). tau=infinity is the dictionary (4 params/topic + 7K).
Everything between is a real hierarchical model, and the ridge trace df says exactly where on that
line it sits. This script sweeps the whole line and prints the Pareto frontier, so the report can
say what an AUC point costs in parameters instead of asserting it.

Every constant (K, ALS rounds, atom weighting, tau) is chosen on the FIRST NINE origins (1963..1987).
1990/1993/1996 are reported held out. Deterministic throughout -- no seed anywhere.

  python3 analysis/arxivtopics/hier_final.py
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arxiv_fit as af                                            # noqa: E402
from pool_shrink import auc_at, build_targets                     # noqa: E402
from dict_spectrum import fit_dict                                # noqa: E402
from pool_refit import cache_wall, solve_pen, edf                 # noqa: E402

OUT = os.path.join(HERE, "hier_final_result.json")
K, RD, TW = 12, 4, "share"          # dict_spectrum.py's select-9 winner, carried over unchanged
TAUS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 1e3, 1e6]


def main():
    t_all = time.time()
    names, Y, labels, future = af.load_lunar()
    TH, R = af.sky_lunar(labels + future)
    n = Y.shape[1]; Tn, nb = Y.shape[0], TH.shape[1]
    WALLS = list(range(n - 63, n - 29, 3))
    SEL, HELD = WALLS[:9], WALLS[9:]
    yr = lambda w: int(labels[w]); w96 = WALLS[-1]
    flds = np.unique([af.META["field"][nm] for nm in names], return_inverse=True)[1]
    NF = int(flds.max() + 1)

    auc = {}; dfs = {}; dict_auc = {}; base_auc = {}
    atoms96 = None; kk96 = None
    for w in WALLS:
        t0 = time.time()
        cw = cache_wall(Y, TH, w)
        yh0, c0, gi0 = solve_pen(cw, np.zeros((Tn, nb)), 0.0)      # tau=0 == the model of record
        base_auc[w] = auc_at(Y, yh0, w)
        A0 = c0[:, 1:]
        U0 = A0 / np.maximum(np.linalg.norm(A0, axis=1), 1e-12)[:, None]
        (yhd, kk, gid, bd, sd, rd_), G = fit_dict(Y, TH, w, K, RD, U0, TW)
        dict_auc[w] = auc_at(Y, yhd, w)
        T_atom = sd[:, None] * G[kk]                               # the dictionary's implied arrows
        T_field = build_targets(A0, flds, "signed")[0]              # pool_refit's select-9 winner
        if w == w96:
            atoms96, kk96 = G, kk
        for nm, tg in (("atom", T_atom), ("field", T_field)):
            for tau in TAUS:
                if tau == 0.0:
                    auc[(nm, tau)] = dict(base_auc); dfs[(nm, tau)] = edf(cw, gi0, 0.0) + 1.0
                    continue
                yh, c, gi = solve_pen(cw, tg, tau)
                auc.setdefault((nm, tau), {})[w] = auc_at(Y, yh, w)
                if w == w96:
                    dfs[(nm, tau)] = edf(cw, gi, tau) + 1.0
        print(f"  wall {yr(w)}  free {base_auc[w]:+.4f}  dict(K={K}) {dict_auc[w]:+.4f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del cw

    def agg(a):
        return (float(np.mean([a[w] for w in SEL])), float(np.mean([a[w] for w in HELD])),
                float(np.mean([a[w] for w in WALLS])), a[w96])

    b_sel, b_held, b_all, b_96 = agg(base_auc)
    d_sel, d_held, d_all, d_96 = agg(dict_auc)
    print(f"\n  BASELINE  9 p/topic · {9*Tn:5d} params · sel {b_sel:+.4f} held {b_held:+.4f} "
          f"all {b_all:+.4f} 1996 {b_96:+.4f}", flush=True)
    assert abs(b_all - 0.8751) < 5e-4 and abs(b_96 - 0.7990) < 5e-4, (b_all, b_96)
    print(f"  DICT K={K} 4 p/topic · {4*Tn+7*K:5d} params · sel {d_sel:+.4f} held {d_held:+.4f} "
          f"all {d_all:+.4f} 1996 {d_96:+.4f}", flush=True)

    rows = []
    for (nm, tau), a in auc.items():
        s, h, al, w9 = agg(a)
        # honest parameter count: the fitted 8 numbers' ridge df + the tuning, plus the per-topic
        # numbers the PRIOR itself carries (gain + pointer/gauge = 2), faded in as the prior takes
        # over, plus the shared spectra amortised over the 251 topics.
        e = dfs[(nm, tau)]
        take = (9.0 - e) / 5.0                                     # 0 at tau=0, 1 at the fully-pooled end
        glob = (7 * K if nm == "atom" else 7 * NF) / Tn
        ppt = e + 2.0 * min(max(take, 0.0), 1.0)
        rows.append(dict(prior=nm, tau=tau, sel=s, held=h, all=al, w1996=w9, edf=round(e, 3),
                         params_per_topic=round(ppt, 3), params_total=round(ppt * Tn + glob * Tn, 1),
                         per_wall={yr(w): round(a[w], 4) for w in WALLS}))

    for nm in ("atom", "field"):
        print(f"\n  TAU PATH -- prior = {nm}   (select-9 · held-3 · all-12 · 1996 · eff params/topic)",
              flush=True)
        for r in sorted([x for x in rows if x["prior"] == nm], key=lambda x: x["tau"]):
            print(f"    tau {r['tau']:<8g} sel {r['sel']:+.4f}  held {r['held']:+.4f}  "
                  f"all {r['all']:+.4f}  1996 {r['w1996']:+.4f}  ·  {r['params_per_topic']:5.2f} p/topic "
                  f"({r['params_total']:6.0f} total)", flush=True)

    best = max(rows, key=lambda r: r["sel"])
    print(f"\n  SELECTED ON THE FIRST NINE ORIGINS ONLY: prior={best['prior']} tau={best['tau']:g}",
          flush=True)
    for tag, k_, bl in (("select9", "sel", b_sel), ("HELD3", "held", b_held),
                        ("all12", "all", b_all), ("1996", "w1996", b_96)):
        print(f"    {tag:8s} {best[k_]:+.4f}  (baseline {bl:+.4f}, delta {best[k_]-bl:+.4f})", flush=True)
    wins = sum(1 for w in WALLS if best["per_wall"][yr(w)] > round(base_auc[w], 4))
    print(f"    wins {wins}/12 origins · effective {best['params_per_topic']:.2f} params/topic "
          f"({best['params_total']:.0f} total vs the baseline's {9*Tn})", flush=True)
    print(f"    per wall {best['per_wall']}", flush=True)
    print(f"    baseline {{" + ", ".join(f"{yr(w)}: {base_auc[w]:.4f}" for w in WALLS) + "}", flush=True)

    print(f"\n  PARETO (select-9 chooses, all-12 reported; * = at or above the baseline's {b_all:+.4f}):",
          flush=True)
    for r in sorted(rows, key=lambda x: x["params_per_topic"]):
        mark = "*" if r["all"] >= b_all else " "
        print(f"   {mark} {r['prior']:5s} tau {r['tau']:<8g} {r['params_per_topic']:5.2f} p/topic "
              f"({r['params_total']:6.0f}) · all12 {r['all']:+.4f} · held3 {r['held']:+.4f}", flush=True)

    print(f"\n  the {K} shared spectra @1996 (" + " ".join(b[:2] for b in af.BODIES) + "), usage:",
          flush=True)
    use = np.bincount(kk96, minlength=K)
    for k in np.argsort(-use):
        print(f"    n={use[k]:3d}  " + " ".join(f"{v:+.3f}" for v in atoms96[k]), flush=True)

    wall_s = time.time() - t_all
    res = dict(model="hierarchical: free fits shrunk toward a K-atom shared spectrum (tau path)",
               K=K, als_rounds=RD, atom_weighting=TW, taus=TAUS,
               baseline=dict(sel=b_sel, held=b_held, all=b_all, w1996=b_96,
                             params_per_topic=9, params_total=9 * Tn,
                             per_wall={yr(w): round(base_auc[w], 4) for w in WALLS}),
               dictionary=dict(sel=d_sel, held=d_held, all=d_all, w1996=d_96,
                               params_per_topic=4, params_total=4 * Tn + 7 * K,
                               per_wall={yr(w): round(dict_auc[w], 4) for w in WALLS}),
               walls=[yr(w) for w in WALLS], select=[yr(w) for w in SEL], held=[yr(w) for w in HELD],
               path=rows, selected=best, wins_vs_baseline=wins, deterministic=True,
               wall_clock_s=round(wall_s, 1))
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  wall clock {wall_s:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
