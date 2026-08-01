#!/usr/bin/env python3
"""30-yr DETECTOR × ROSTER cross (operator 2026-07-24) — the interaction the single-axis battery missed.
The battery showed: lower detector power → higher AUC / lower median skill; dropping the FAST bodies
(mars 1.9y, jupiter 12y) and CHIRON helps; saturn+node are load-bearing. This crosses them and scores
BOTH the pooled 30-yr AUC (the operator's headline) and the median per-topic skill (is the typical field
beaten?), seed-checked, so the pick maximises AUC without a negative-median blowup.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/arxiv_30yr_cross.py
"""
import importlib.util as u, json, os
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
A = _load("analysis/arxivtopics/arxiv_30yr_ablation.py", "A30")
S = A.S

ROSTERS = {
    "record8":      A.REC,
    "drop-chiron7": [b for b in A.REC if b != "chiron"],
    "drop-fast6":   ["saturn", "uranus", "neptune", "pluto", "node", "chiron"],   # drop mars+jupiter (=slow6)
    "core5":        ["saturn", "uranus", "neptune", "pluto", "node"],             # drop mars+jupiter+chiron
    "core5+jup":    ["jupiter", "saturn", "uranus", "neptune", "pluto", "node"],
}
DETS = ["relu2", "relu1_5", "relu1"]
SEEDS = (7, 1, 2, 3)

rows = []
print(f"== 30-yr DETECTOR × ROSTER cross · wall {A.W30} ==", flush=True)
for det in DETS:
    for rn, r in ROSTERS.items():
        aucs, sks, pcs = [], [], []
        for sd in SEEDS:
            s, p, a = S.bench(A.fit(det=det, bodies=r, seed=sd), wall=A.W30)
            aucs.append(a); sks.append(s); pcs.append(p)
        med_a, med_s, med_p = float(np.median(aucs)), float(np.median(sks)), float(np.median(pcs))
        rows.append((f"{det} · {rn}", med_a, med_s, med_p, min(aucs), max(aucs), det, r))
        print(f"  {det:8s} {rn:13s} AUC med {med_a:+.4f} [{min(aucs):+.4f}..{max(aucs):+.4f}] · "
              f"skill {med_s:+.4f} · {med_p:.1f}%>0", flush=True)

print("\n  LEAGUE by 30-yr AUC (seed-median):", flush=True)
for t, a, s, p, lo, hi, det, r in sorted(rows, key=lambda x: -x[1]):
    print(f"    {a:+.4f} AUC · {s:+.4f} skill · {p:.1f}%>0 · {t}", flush=True)
print("\n  LEAGUE by MEDIAN SKILL (the typical field):", flush=True)
for t, a, s, p, lo, hi, det, r in sorted(rows, key=lambda x: -x[2]):
    print(f"    {s:+.4f} skill · {a:+.4f} AUC · {p:.1f}%>0 · {t}", flush=True)

# best AUC with a NON-NEGATIVE median (a headline model must beat the mean for most topics)
ok = [r for r in rows if r[2] > 0.05]
best = max(ok, key=lambda x: x[1]) if ok else max(rows, key=lambda x: x[1])
print(f"\n  PICK (max AUC with median skill>0.05): {best[0]}  AUC {best[1]:+.4f} skill {best[2]:+.4f}", flush=True)
json.dump({"pick": best[0], "det": best[6], "roster": best[7], "auc30": best[1], "skill30": best[2],
           "rows": [(t, a, s, p) for t, a, s, p, lo, hi, det, r in sorted(rows, key=lambda x: -x[1])]},
          open("analysis/arxivtopics/arxiv_30yr_cross.json", "w"), indent=1)
print("CROSSDONE", flush=True)
