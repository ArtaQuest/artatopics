#!/usr/bin/env python3
"""TRIPLE-CHECK of the round-2 scoring metric (operator: "tripple check everyghign and ensure
robustness and fairness"). Every claim the harness makes about sign_information() is tested here:
exact boundary values, monotonicity, grid-independence (the grid must not be a tuning knob),
invariance to arbitrary sign-frame choices, correct behaviour on null models that carry NO phase
information, and resistance to the obvious gaming vectors. Run it before trusting any leaderboard.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/verify_phase_metric.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import comp_harness as CH

W = WALL_OUTER
SYNTH_T = 0.02   # synthetic-curve noise scale: wells below are ~0.9 deep, so T must be << that
# For the SYNTHETIC curves below we pin the common temperature to 1.0 so the analytic answers are exact.
CH._NOISE[(W, 0.75)] = np.full(Tn, SYNTH_T)

OK = []
def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}", flush=True)

G = PHASE_GRID
M = len(G)
print("═══ A. EXACT BOUNDARY VALUES (synthetic curves, analytic answers) ═══", flush=True)
# a flat curve = the model's loss does not care where the sky is = zero information
flat = np.ones((Tn, M))
r = sign_information(flat, np.zeros(Tn), W)
check("uniform tuning curve → info = 0", abs(r["info"]) < 1e-9, f"info={r['info']}")

# a curve with one deep cell → all mass in one sign → maximal information
delta = np.ones((Tn, M)); delta[:, M // 2] = 0.0            # min at δ=0 → phase stays as declared
r = sign_information(delta, np.zeros(Tn), W)
check("single-cell well → info = 1", abs(r["info"] - 1.0) < 1e-6, f"info={r['info']}")

# mass split evenly over exactly 2 signs → H = 1 bit → info = 1 − 1/log2(12)
two = np.full((Tn, M), 50.0)
for i, d in enumerate(G):
    if (0 <= (0 - d) % 360 < 30) or (30 <= (0 - d) % 360 < 60): two[:, i] = 0.0
r = sign_information(two, np.zeros(Tn), W)
exp2 = 1 - 1 / np.log2(12)
check("even mass over 2 signs → 1 − 1/log2(12)", abs(r["info"] - exp2) < 0.02, f"info={r['info']} expected≈{exp2:.4f}")

# evenly over 3 signs → H = log2(3)
three = np.full((Tn, M), 50.0)
for i, d in enumerate(G):
    if (0 <= (0 - d) % 360 < 90): three[:, i] = 0.0
r = sign_information(three, np.zeros(Tn), W)
exp3 = 1 - np.log2(3) / np.log2(12)
check("even mass over 3 signs → 1 − log2(3)/log2(12)", abs(r["info"] - exp3) < 0.02, f"info={r['info']} expected≈{exp3:.4f}")

print("═══ A2. THE SCORED PAIR — degenerate strategies must score 0, the ideal must score 1 ═══", flush=True)
# every topic sharp but ALL IN ONE SIGN → collapse → NMI must be 0
sharp = np.ones((Tn, M)); sharp[:, M // 2] = 0.0
r = sign_mutual_information(sharp, np.zeros(Tn), W)
check("sharp-but-collapsed atlas → NMI = 0", r["nmi"] < 1e-9, f"nmi={r['nmi']} (per-topic {r['per_topic_bits']} bits, across {r['across_topic_bits']})")
# every topic uniform → NMI 0
r = sign_mutual_information(np.ones((Tn, M)), np.linspace(0, 359, Tn), W)
check("uninformative atlas → NMI = 0", r["nmi"] < 1e-9, f"nmi={r['nmi']}")
# sharp AND evenly spread over 12 signs → NMI = 1
ph12 = np.tile(np.arange(12) * 30.0 + 15.0, Tn // 12 + 1)[:Tn]
r = sign_mutual_information(sharp, ph12, W)
check("sharp + evenly spread over 12 signs → NMI = 1", abs(r["nmi"] - 1.0) < 1e-6, f"nmi={r['nmi']}")
# peak-to-mean confidence boundaries
check("flat curve → confidence 0", abs(phase_confidence(np.ones((Tn, M)), W)["conf"]) < 1e-9)
check("single-cell peak → confidence ≈ 1", phase_confidence(sharp, W)["conf"] > 0.99,
      f"conf={phase_confidence(sharp, W)['conf']}")

print("═══ B. MONOTONICITY — narrower well ⇒ strictly more information ═══", flush=True)
infos = []
for kappa in (0.05, 0.2, 1.0, 4.0, 16.0, 64.0):
    L = np.tile(1.0 - 0.9 * np.exp(kappa * (np.cos(np.deg2rad(G)) - 1.0)), (Tn, 1))
    infos.append(sign_information(L, np.full(Tn, 15.0), W)["info"])   # mid-sign, not a boundary
check("info increases monotonically with concentration (mid-sign phase)", all(b >= a - 1e-9 for a, b in zip(infos, infos[1:])),
      " → ".join(f"{v:.3f}" for v in infos))

print("═══ C. GRID-INDEPENDENCE — the δ grid must NOT be a tuning knob ═══", flush=True)
gs, vals, cvals = [1.0, 2.0, 3.0, 5.0, 10.0, 15.0], [], []
for step in gs:
    g = np.arange(-180.0, 180.0, step)
    L = np.tile(1.0 - 0.9 * np.exp(3.0 * (np.cos(np.deg2rad(g)) - 1.0)), (Tn, 1))
    CH._NOISE[(W, 0.75)] = np.full(Tn, SYNTH_T)
    vals.append(sign_information(L, np.full(Tn, 15.0), W, grid=g)["info"])
    cvals.append(phase_confidence(L, W, grid=g)["conf"])
fine = max(vals[:4]) - min(vals[:4])          # 1°..5°, the converged regime (harness ships 1°)
check("sign info is CONVERGED over the fine grids the harness uses (1°-5°)", fine < 0.02,
      " ".join(f"{s:g}°={v:.4f}" for s, v in zip(gs, vals)) + f" | fine-spread={fine:.4f}")
cfine = max(cvals) - min(cvals)
check("peak-to-mean confidence is grid-independent across 1°..15°", cfine < 0.03,
      " ".join(f"{s:g}°={v:.4f}" for s, v in zip(gs, cvals)) + f" spread={cfine:.4f}")

print("═══ D. FAIRNESS — no privileged sign, no privileged frame ═══", flush=True)
# The twelve signs must be interchangeable: shifting a topic's phase by a WHOLE sign (30°) relabels
# its sign but changes nothing about the geometry, so the score must be EXACTLY unchanged.
rot = []
for shift in (0.0, 30.0, 60.0, 90.0, 180.0, 330.0):
    Lr_ = np.tile(1.0 - 0.9 * np.exp(3.0 * (np.cos(np.deg2rad(G)) - 1.0)), (Tn, 1))
    rot.append(sign_information(Lr_, np.full(Tn, 15.0 + shift), W)["info"])
check("no privileged sign — a whole-sign (30°) relabel leaves the score identical",
      max(rot) - min(rot) < 1e-9, " ".join(f"{v:.4f}" for v in rot))
# and within a sign the score DOES vary — decisiveness is the point of the metric, not a bug
within = [sign_information(np.tile(1.0 - 0.9 * np.exp(3.0 * (np.cos(np.deg2rad(G)) - 1.0)), (Tn, 1)),
                           np.full(Tn, off), W)["info"] for off in (0.0, 7.0, 15.0, 23.0, 29.0)]
check("within a sign, a mid-sign phase scores higher than a boundary one (intended)",
      within[2] > within[0] and within[2] > within[4],
      "offsets 0/7/15/23/29° → " + " ".join(f"{v:.3f}" for v in within))
# a well straddling a boundary is GENUINELY ambiguous — it must score lower, and that is correct
Lb = np.tile(1.0 - 0.9 * np.exp(3.0 * (np.cos(np.deg2rad(G)) - 1.0)), (Tn, 1))
centred = sign_information(Lb, np.full(Tn, 15.0), W)["info"]     # mid-sign
straddle = sign_information(Lb, np.zeros(Tn), W)["info"]     # exactly on a boundary
check("a boundary-straddling phase scores lower (honest ambiguity, not a bug)", straddle <= centred + 1e-9,
      f"mid-sign={centred:.4f} boundary={straddle:.4f}")

print("═══ E. NULL MODELS — anything with no phase information must score ~0 ═══", flush=True)
persist = np.repeat(Y[:, WALL_OUTER - 1][:, None], n, 1)      # carry-forward: no sky at all
Lp = phase_curve(lambda d: persist, WALL_OUTER)
CH._NOISE.pop((W,0.75), None)
rp = sign_information(Lp, np.zeros(Tn), W)
check("sky-free persistence model → info = 0", rp["info"] < 1e-6, f"info={rp['info']}")
rng = np.random.RandomState(0)
Lr = 1.0 + 0.001 * rng.randn(Tn, M)                            # pure noise curve
rr = sign_information(Lr, rng.uniform(0, 360, Tn), W)
check("noise-only tuning curve → info ≈ 0", rr["info"] < 0.05, f"info={rr['info']}")

print("═══ F. THE REAL MODEL (deployed v9) + GAMING PROBES ═══", flush=True)
yh, pdel, ph = champion_with_phase(WALL_OUTER, seed=7)
ev = evaluate(yh, WALL_OUTER)
L = phase_curve(pdel, WALL_OUTER)
si = sign_information(L, ph, W)
mi = sign_mutual_information(L, ph, W)
pc = phase_confidence(L, W)
ai = angle_information(L, W)
f1 = global_f1(ev["auc"], mi["nmi"])
print(f"    v9 OUTER: AUC {ev['auc']}  NMI {mi['nmi']}  F1 {f1}")
print(f"    per-topic conf (peak/mean) {pc['conf']} (peak is {pc['peak_over_mean_median']}x the mean) · "
      f"per-topic {mi['per_topic_bits']} bits, across-topic {mi['across_topic_bits']}/{mi['max_bits']} bits")
print(f"    diagnostics: agreement {si['sign_agreement']} · top-sign mass {si['top_sign_mass_median']} · "
      f"diversity {sign_diversity(ph)} · continuous-angle info {ai['angle_info']} ({ai['angle_bits_median']} bits)")
check("declared sign == argmax of its own sign posterior (boundary topics differ)", si["sign_agreement"] > 0.85,
      f"agreement={si['sign_agreement']}")
check("topics spread across the signs (no collapse)", sign_diversity(ph) > 0.5, f"diversity={sign_diversity(ph)}")

# GAMING PROBE 1: inflate the model's amplitudes — does info rise for free?
import torch as T
def scaled_predict(scale):
    def f(d):
        base = pdel(d)
        return np.clip(base * scale, 0, None)
    return f
for sc in (0.5, 2.0, 5.0):
    Ls = phase_curve(scaled_predict(sc), WALL_OUTER)
    print(f"    amplitude ×{sc}: NMI {sign_mutual_information(Ls, ph, W)['nmi']:.4f} conf {phase_confidence(Ls, W)['conf']:.4f} "
          f"(AUC {evaluate(np.clip(yh*sc,0,None), WALL_OUTER)['auc']:+.4f})")
# GAMING PROBE 2: collapse every topic onto one phase — diversity guard must fire
collapsed = np.zeros(Tn)
print(f"    phase collapse on the REAL curve → NMI {sign_mutual_information(L, collapsed, W)['nmi']:.4f} "
      f"(diversity {sign_diversity(collapsed):.4f})")
check("collapse costs most of the score on the metric ITSELF (no external guard needed)",
      sign_mutual_information(L, collapsed, W)["nmi"] < 0.5 * mi["nmi"],
      f"collapsed {sign_mutual_information(L, collapsed, W)['nmi']:.3f} vs honest {mi['nmi']:.3f}")

print("═══ F2. THE AUDIT FIXES — capacity must not buy free sharpness ═══", flush=True)
lmin_r = L.min(1, keepdims=True)
L_overfit = (L - lmin_r) + lmin_r * 0.1        # same well depth, 10x lower floor = pure overfitting
nmi_of = sign_mutual_information(L_overfit, ph, W)["nmi"]
check("overfitting the floor buys NO extra phase score", abs(nmi_of - mi["nmi"]) < 0.02,
      f"honest {mi['nmi']:.4f} vs overfit-floor {nmi_of:.4f} (was 0.4406 -> 0.7048 before the fix)")
# Noise-referenced BY DESIGN: a model whose loss barely moves when the sky rotates has not identified
# the phase, however well it fits. So shrinking the whole curve toward flat MUST lower the score.
L_flatter = (L - L.min(1, keepdims=True)) * 0.1 + L.min(1, keepdims=True)
nmi_flat = sign_mutual_information(L_flatter, ph, W)["nmi"]
check("a 10x flatter response to rotation scores LOWER (noise-referenced by design)",
      nmi_flat < mi["nmi"], f"honest {mi['nmi']:.4f} vs flattened {nmi_flat:.4f}")
# amplitude inflation raises NMI but destroys AUC — the harmonic mean must kill it
f1_amp = global_f1(evaluate(np.clip(yh * 5.0, 0, None), W)["auc"],
                   sign_mutual_information(phase_curve(scaled_predict(5.0), W), ph, W)["nmi"])
check("amplitude-inflation gaming dies on the F1 (AUC collapses)", f1_amp < 0.05,
      f"F1 of the x5 amplitude model = {f1_amp:.4f} vs honest {global_f1(ev['auc'], mi['nmi']):.4f}")
mask_diff = int((TV[:, :W] != train_mask(W)).sum())
check("training mask is leak-free (no peeking past the wall)", mask_diff == 0, f"cells differing = {mask_diff}")

print("═══ G. SEED ROBUSTNESS of the metric on the real model ═══", flush=True)
res = [(ev["auc"], mi["nmi"], f1, ph)]
for sd in (11, 23):
    y2, p2, h2 = champion_with_phase(WALL_OUTER, seed=sd)
    L2 = phase_curve(p2, WALL_OUTER); m2 = sign_mutual_information(L2, h2, W)
    e2 = evaluate(y2, WALL_OUTER)
    res.append((e2["auc"], m2["nmi"], global_f1(e2["auc"], m2["nmi"]), h2))
for a, i, f, _ in res: print(f"    seed: AUC {a:+.4f}  NMI {i:.4f}  F1 {f:.4f}")
f1s = [r[2] for r in res]
check("F1 stable across seeds (spread < 0.03)", max(f1s) - min(f1s) < 0.03, f"spread={max(f1s)-min(f1s):.4f}")
phs = [r[3] for r in res]
check("phase stable across seeds (>0.9)", phase_stability(phs) > 0.9, f"stability={phase_stability(phs)}")
check("sign assignment reproducible across seeds (>0.75; boundary topics flip — a REAL caveat)",
      sign_agreement_across_seeds(phs) > 0.75, f"sign agreement={sign_agreement_across_seeds(phs)} "
      f"— ~{100*(1-sign_agreement_across_seeds(phs)):.0f}% of topics change sign between seeds, which is "
      f"exactly what training the KL term should improve")

print(f"\n═══ RESULT: {sum(OK)}/{len(OK)} checks passed ═══", flush=True)
import json
json.dump({"auc": ev["auc"], "nmi": mi["nmi"], "conf_peak_over_mean": pc["conf"],
           "sign_info": si["info"], "f1": f1,
           "sign_agreement": si["sign_agreement"], "diversity": sign_diversity(ph),
           "angle_info": ai["angle_info"], "stability": phase_stability(phs),
           "sign_stability": sign_agreement_across_seeds(phs),
           "checks_passed": f"{sum(OK)}/{len(OK)}"},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "round2_baseline.json"), "w"), indent=1)
print("VERIFYDONE", flush=True)
