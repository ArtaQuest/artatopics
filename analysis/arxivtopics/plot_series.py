#!/usr/bin/env python3
"""Time series of the data and the deployed model's fit (operator 2026-08-07).

Exports, from the HONEST fit (trained on <=1995 only; everything after is pure forecast, the sky
being known in advance; extended to 2055):
  figs/fit_multiples.png  -- six recognisable fields, actual vs model, the wall marked
  docs/series.js          -- every field's actual + model series for the page's interactive chart

  python3 analysis/arxivtopics/plot_series.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arxiv_fit as af
import global_phasor as GP

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
names, Y, labels, future = GP.names, GP.Y, GP.labels, GP.future
n, ne = GP.n, GP.ne
wall = n - 30
P_exact = GP.fit_wall(wall)[3]        # the deployed model: level + gain per field, global arrows
years_all = [int(y) for y in (labels + future)]
starts = af.META["topic_valid"].argmax(1)

# ── the page's data: every field, from its own start, to 2055 ────────────────────────────────────
S = {}
for j, nm in enumerate(names):
    t0 = int(starts[j])
    S[nm] = {"y0": years_all[t0], "wall": wall - t0,
             "actual": [round(float(v), 4) for v in Y[j, t0:n]],
             "model": [round(float(v), 4) for v in P_exact[j, t0:ne]]}
os.makedirs(os.path.join(REPO, "docs"), exist_ok=True)
with open(os.path.join(REPO, "docs", "series.js"), "w") as f:
    f.write("const SERIES = " + json.dumps(S, separators=(",", ":")) + ";\n")
sz = os.path.getsize(os.path.join(REPO, "docs", "series.js")) // 1024
print(f"docs/series.js written ({sz}KB, {len(S)} fields)")

# ── the static multiples: six recognisable fields ────────────────────────────────────────────────
PICK = ["Artificial Intelligence", "Molecular Biology", "Astronomy and Astrophysics",
        "Surgery", "Electrical and Electronic Engineering", "Organic Chemistry"]
BLUE, GOLD, INK, MUT = "#1746DC", "#8A6D0B", "#1a2330", "#6b7686"
fig, axes = plt.subplots(3, 2, figsize=(9.6, 8.4), dpi=150)
for ax, nm in zip(axes.ravel(), PICK):
    j = names.index(nm)
    t0 = int(starts[j])
    yrs_a = years_all[t0:n]; yrs_m = years_all[t0:ne]
    ax.plot(yrs_a, Y[j, t0:n], color=BLUE, lw=1.4, label="actual")
    ax.plot(yrs_m[:wall - t0 + 1], P_exact[j, t0:wall + 1], color=GOLD, lw=1.8, label="model, fitted")
    ax.plot(yrs_m[wall - t0:], P_exact[j, wall:ne], color=GOLD, lw=1.8, ls="--", label="model, forecast")
    ax.axvline(years_all[wall], color=MUT, lw=0.8, ls=":")
    ax.set_title(nm, fontsize=9.5, color=INK)
    ax.tick_params(labelsize=7.5, colors=MUT)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.12)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.suptitle("Share of each year's citations — the data (blue) and the deployed model (gold);\n"
             "fitted on years left of the dotted line only, pure forecast to the right, extended to 2055",
             fontsize=10.5, color=INK)
fig.tight_layout(rect=[0, 0.045, 1, 0.92])
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figs", "fit_multiples.png"))
print("figs/fit_multiples.png written")
