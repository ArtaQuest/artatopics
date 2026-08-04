#!/usr/bin/env python3
"""adstopics SPEC-model improvement round — everything INSIDE the model of record.

The model is fixed (operator spec): y_hat(t) = Σ_i w_i · sinc(f_i · wrap(x_i(t) − p)), 12 bodies,
25 positive parameters, and EVERY fit descends from 12 phase initializations, one per sign centre
(15°..345°), best validation checkpoint wins. Nothing outside the model — no level track, no
intercept, no other kernels. The arms vary only HOW the spec model is fitted and selected:

  A  baseline        stage-1 sinc12 → prune-4 → refit (the current atlas protocol)
  B  clip            A + predictions clipped to the physical Trends range [0,100]
  C  huber           A fitted with Huber loss (robust to the spike months that yank MSE)
  D  prune-k(val)    per-topic surviving-lamp count k ∈ {2,4,6,8,12} chosen on VALIDATION
  E  D + clip        the composed protocol
  Judge: FUTURE test window (recency year excluded), full-tier population; MEAN + median + >0.

  python3 analysis/adstopics/spec_experiments.py [N_topics]
→ analysis/adstopics/spec_results.csv
"""
import importlib.util as u, os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m

tf = _load("analysis/trends_fit.py", "tf")
r5 = _load("analysis/topic500_reference_solution.py", "r5")
ex = _load("analysis/adstopics/experiments.py", "ex")
co = _load("analysis/adstopics/combo_experiments.py", "co")

KS = [2, 4, 6, 8, 12]


def fit_sinc(Ys, X, dev, body_mask=None, huber=False):
    """The spec fit — thin wrapper so the Huber arm shares co.fit_vm's 12-start machinery."""
    if not huber:
        return co.fit_vm(Ys, X, dev, body_mask=body_mask, kernel="sinc")
    return co.fit_vm(Ys, X, dev, body_mask=body_mask, kernel="sinc", loss="huber")


def contributions(par, X, a, T):
    C = np.zeros((T, ex.NBX))
    for i in range(T):
        z = np.sinc(np.deg2rad((X[:a] - par["p"][i] + 180.0) % 360.0 - 180.0) * par["kappa"][i][None, :])
        C[i] = (par["w"][i][None, :] * z).std(0)
    return C


def main():
    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    names, Ys, X = ex.load_topics(n_topics)
    n = len(Ys[0]); a, b = ex.split3(n)
    T = len(Ys)
    print(f"[spec] {T} topics · {n} months (recency year excluded)")
    dev = r5._device()

    def scores(P, clip=False):
        val, test = [], []
        for i, y in enumerate(Ys):
            pr = np.clip(P[i], 0.0, 100.0) if clip else P[i]
            sstv = max(((y[a:b] - y[a:b].mean()) ** 2).sum(), 1e-9)
            sst = max(((y[b:] - y[b:].mean()) ** 2).sum(), 1e-9)
            val.append(1 - ((y[a:b] - pr[a:b]) ** 2).sum() / sstv)
            test.append(1 - ((y[b:] - pr[b:]) ** 2).sum() / sst)
        return np.array(val), np.array(test)

    rows = []
    def report(tag, val, test):
        rows.append(dict(arm=tag, mean_test=float(test.mean()), median_test=float(np.median(test)),
                         frac_pos=float((test > 0).mean()),
                         clamped=float(np.clip(test, 0, 1).mean() * 100)))
        print(f"  {tag:16s} TEST mean {test.mean():8.3f} · med {np.median(test):6.3f} · "
              f">0 {(test>0).mean()*100:4.1f}% · clamped {np.clip(test,0,1).mean()*100:5.1f}", flush=True)

    # stage 1 (shared): all 12 lamps, MSE
    P12, par1 = fit_sinc(Ys, X, dev)
    C = contributions(par1, X, a, T)

    # per-k pruned refits (shared by arms A/D/E)
    preds_k = {12: P12}
    for k in (2, 4, 6, 8):
        m = np.zeros_like(C)
        idx = np.argsort(-C, axis=1)[:, :k]
        for i in range(T):
            m[i, idx[i]] = 1.0
        preds_k[k], _ = fit_sinc(Ys, X, dev, body_mask=m)

    vA, tA = scores(preds_k[4]);            report("A_prune4", vA, tA)
    vB, tB = scores(preds_k[4], clip=True); report("B_prune4+clip", vB, tB)

    # C: Huber-fitted prune-4
    P12h, par1h = fit_sinc(Ys, X, dev, huber=True)
    Ch = contributions(par1h, X, a, T)
    mh = np.zeros_like(Ch)
    idxh = np.argsort(-Ch, axis=1)[:, :4]
    for i in range(T):
        mh[i, idxh[i]] = 1.0
    P4h, _ = fit_sinc(Ys, X, dev, body_mask=mh, huber=True)
    vC, tC = scores(P4h, clip=True);        report("C_huber4+clip", vC, tC)

    # D/E: per-topic k chosen on validation
    # F/G: the 13-param spec (f frozen at 1) — the operator head-to-head showed frozen f
    # generalizes far better than trained f; test whether pruning compounds the gain
    PF, parF = co.fit_vm(Ys, X, dev, kernel="sinc", intercept=False, fixed_f=1.0)
    vF, tF = scores(PF, clip=True);         report("F_13p+clip", vF, tF)
    CF = contributions(parF, X, a, T)
    mF = np.zeros_like(CF)
    idxF = np.argsort(-CF, axis=1)[:, :4]
    for i in range(T):
        mF[i, idxF[i]] = 1.0
    PG, _ = co.fit_vm(Ys, X, dev, kernel="sinc", intercept=False, fixed_f=1.0, body_mask=mF)
    vG, tG = scores(PG, clip=True);         report("G_13p_prune4+clip", vG, tG)

    vals = {k: scores(preds_k[k])[0] for k in KS}
    pick = np.array([max(KS, key=lambda k: vals[k][i]) for i in range(T)])
    PD = np.stack([preds_k[pick[i]][i] for i in range(T)])
    vD, tD = scores(PD);                    report("D_pruneK(val)", vD, tD)
    vE, tE = scores(PD, clip=True);         report("E_pruneK+clip", vE, tE)
    print(f"  k picks: {dict(zip(*np.unique(pick, return_counts=True)))}")

    pd.DataFrame(rows).to_csv("analysis/adstopics/spec_results.csv", index=False)

if __name__ == "__main__":
    main()
