#!/usr/bin/env python3
"""Figures for the per-topic trending classifiers. Palette validated (dataviz six checks):
blue #1746DC + dark-gold #8A6D0B on white; every bar direct-labelled; text in ink, never series color."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "trend_binary.json")))
C = json.load(open(os.path.join(HERE, "trend_binary_controls.json")))
BLUE, GOLD, INK, MUT = "#1746DC", "#8A6D0B", "#1a2330", "#6b7686"
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)

tr = np.array([t["train"] for t in R["topics"].values()])
te = np.array([t["test"] for t in R["topics"].values()])

# ── Fig 1: train vs shuffled-test accuracy, one dot per topic ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=160)
ax.scatter(tr, te, s=14, c=BLUE, alpha=0.55, linewidths=0)
lo, hi = 0.25, 1.0
ax.plot([lo, hi], [lo, hi], color=MUT, lw=1, ls="--", zorder=0)
ax.axhline(0.5, color=MUT, lw=1, ls=":", zorder=0)
ax.axvline(0.5, color=MUT, lw=1, ls=":", zorder=0)
ax.annotate("chance", (0.505, 0.26), color=MUT, fontsize=8)
ax.annotate("train = test", (0.80, 0.845), color=MUT, fontsize=8, rotation=38)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("train accuracy", color=INK); ax.set_ylabel("held-out accuracy (shuffled 90/10)", color=INK)
ax.set_title("251 per-topic trending classifiers — balanced binary, sky features", color=INK, fontsize=11)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(colors=MUT); ax.grid(alpha=0.15)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "trend_binary_scatter.png")); plt.close(fig)

# ── Fig 2: the controls — rhythm or clock ────────────────────────────────────────────────────────
arms = list(C.keys())
sh = [C[a]["shuffled"] for a in arms]; tp = [C[a]["temporal"] for a in arms]
fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=160)
x = np.arange(len(arms)); wdt = 0.36
b1 = ax.bar(x - wdt/2, sh, wdt, color=BLUE, label="shuffled test", edgecolor="white", linewidth=2)
b2 = ax.bar(x + wdt/2, tp, wdt, color=GOLD, label="temporal test", edgecolor="white", linewidth=2)
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width()/2, b.get_height() + 0.012),
                    ha="center", fontsize=8.5, color=INK)
ax.axhline(0.5, color=MUT, lw=1, ls=":")
ax.annotate("chance", (3.36, 0.505), color=MUT, fontsize=8)
ax.set_xticks(x, [a.replace(" (the model)", "\n(the model)").replace(" (mars, jupiter)", "\n(mars, jupiter)")
                   .replace(" (ur, ne, pl)", "\n(uranus, neptune, pluto)").replace(" [1, t]", "\n[1, t]") for a in arms],
              fontsize=8.5, color=INK)
ax.set_ylim(0, 0.85); ax.set_ylabel("mean held-out accuracy", color=INK)
ax.set_title("Rhythm or clock? Fast bodies are chance; a bare year beats the sky", color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(colors=MUT); ax.grid(axis="y", alpha=0.15)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "trend_binary_controls.png")); plt.close(fig)
print("FIGSDONE", flush=True)
