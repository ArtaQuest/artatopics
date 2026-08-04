#!/usr/bin/env python3
"""Head-to-head requested by the operator: mean FUTURE test R² of
     25-param  y = Σ w_i sinc(f_i(x_i − p))   (trained per-body frequencies)
  vs 13-param  y = Σ w_i sinc(x_i − p)        (f_i ≡ 1 fixed)
Both: raw series, NO intercept, positive weights, 12 sign-centre phase inits, recency yr excluded.
  python3 analysis/adstopics/compare_f.py [N_topics]
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

def main():
    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    names, Ys, X = ex.load_topics(n_topics)
    n = len(Ys[0]); a, b = ex.split3(n)
    print(f"[compare_f] {len(Ys)} topics · {n} months (recency year excluded)")
    dev = r5._device()
    rows = []
    def report(tag, P):
        test = []
        for i, y in enumerate(Ys):
            sst = max(((y[b:] - y[b:].mean()) ** 2).sum(), 1e-9)
            test.append(1 - ((y[b:] - P[i][b:]) ** 2).sum() / sst)
        test = np.array(test)
        rows.append(dict(arm=tag, mean_test=float(test.mean()), median_test=float(np.median(test)),
                         frac_pos=float((test > 0).mean()),
                         clamped=float(np.clip(test, 0, 1).mean() * 100)))
        print(f"  {tag:22s} TEST mean {test.mean():8.3f} · med {np.median(test):6.3f} · "
              f">0 {(test>0).mean()*100:4.1f}% · clamped {np.clip(test,0,1).mean()*100:5.1f}", flush=True)
    if os.environ.get("AQ_LINKS13"):
        # operator round 2: 14-param family (12 w + p + bias b), f frozen — three links
        for kern, tag in (("sinc", "14p_sinc+b"), ("cos", "14p_cos+b"), ("gauss", "14p_gauss+b")):
            P, _ = co.fit_vm(Ys, X, dev, kernel=kern, intercept=True, fixed_f=1.0)
            report(tag, P)
        pd.DataFrame(rows).to_csv("analysis/adstopics/compare_links13_results.csv", index=False)
        return
    P25, _ = co.fit_vm(Ys, X, dev, kernel="sinc", intercept=False)
    report("25p_trained_f", P25)
    P13, _ = co.fit_vm(Ys, X, dev, kernel="sinc", intercept=False, fixed_f=1.0)
    report("13p_fixed_f1", P13)
    pd.DataFrame(rows).to_csv("analysis/adstopics/compare_f_results.csv", index=False)

if __name__ == "__main__":
    main()
