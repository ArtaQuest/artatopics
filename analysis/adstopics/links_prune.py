#!/usr/bin/env python3
"""Final composition round: the two winning smooth links (cos, gauss; 14p frozen-shape + bias)
with prune-4 refit and the physical [0,100] clip; plus a trained-width gauss (26p) control.
Judge: future test window, recency year excluded.
  python3 analysis/adstopics/links_prune.py [N_topics]
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

def kernel_vals(kern, par, X, a):
    z = np.deg2rad((X[:a] - par["p"][:, None, None] + 180.0) % 360.0 - 180.0)  # unused; per-topic below
    return None

def main():
    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    names, Ys, X = ex.load_topics(n_topics)
    n = len(Ys[0]); a, b = ex.split3(n)
    T = len(Ys)
    print(f"[links_prune] {T} topics · {n} months")
    dev = r5._device()
    rows = []
    def report(tag, P, clip=True):
        test = []
        for i, y in enumerate(Ys):
            pr = np.clip(P[i], 0.0, 100.0) if clip else P[i]
            sst = max(((y[b:] - y[b:].mean()) ** 2).sum(), 1e-9)
            test.append(1 - ((y[b:] - pr[b:]) ** 2).sum() / sst)
        test = np.array(test)
        rows.append(dict(arm=tag, mean_test=float(test.mean()), median_test=float(np.median(test)),
                         frac_pos=float((test > 0).mean()),
                         clamped=float(np.clip(test, 0, 1).mean() * 100)))
        print(f"  {tag:22s} TEST mean {test.mean():8.3f} · med {np.median(test):6.3f} · "
              f">0 {(test>0).mean()*100:4.1f}% · clamped {np.clip(test,0,1).mean()*100:5.1f}", flush=True)

    for kern in ("cos", "gauss"):
        P1, par1 = co.fit_vm(Ys, X, dev, kernel=kern, intercept=True, fixed_f=1.0)
        report(f"14p_{kern}+b+clip", P1)
        # contributions with the kernel's own shape
        C = np.zeros((T, ex.NBX))
        for i in range(T):
            z = np.deg2rad((X[:a] - par1["p"][i] + 180.0) % 360.0 - 180.0)
            K = np.cos(z) if kern == "cos" else np.exp(-z ** 2)
            C[i] = (par1["w"][i][None, :] * K).std(0)
        m = np.zeros_like(C)
        idx = np.argsort(-C, axis=1)[:, :4]
        for i in range(T):
            m[i, idx[i]] = 1.0
        P4, _ = co.fit_vm(Ys, X, dev, kernel=kern, intercept=True, fixed_f=1.0, body_mask=m)
        report(f"{kern}+b_prune4+clip", P4)
    Pw, _ = co.fit_vm(Ys, X, dev, kernel="gauss", intercept=True)   # trained widths (26p) control
    report("26p_gauss_trainedw+clip", Pw)
    pd.DataFrame(rows).to_csv("analysis/adstopics/links_prune_results.csv", index=False)

if __name__ == "__main__":
    main()
