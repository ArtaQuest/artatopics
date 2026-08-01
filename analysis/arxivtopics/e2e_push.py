#!/usr/bin/env python3
"""ADVERSARIAL ROUND 2 on the end-to-end embedding model (operator: "adversarially improve the
training and embedding layer design choices").

Round 1 left three tells that a search has stopped too early, and one clear failure mode:
  * steps=16000, dropout=0.15 and lr=0.005 were all KEPT AT THE EDGE OF THEIR GRID — the search
    terminated because the grid ran out, not because the optimum was found. Extend all three.
  * OUTER seed spread was 0.039 (v10's is 0.002) — the model is optimisation-sensitive, so the single
    biggest available win is variance reduction, i.e. averaging over seeds. Round 1 of the earlier
    competition showed a seed ensemble merely ties for the LOW-variance v10; for a HIGH-variance model
    it should genuinely help. Test it honestly: select the ensemble size on the inner wall.
  * head="mlp" scored -0.96 — a learned predictor on the rotated phases is catastrophic, so the
    physical rectified square-law stays. (Retested here at a lower LR in case it was just diverging.)

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/e2e_push.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import importlib.util as u

_s = u.spec_from_file_location("e2e", os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_embed.py"))
e2e = u.module_from_spec(_s); _s.loader.exec_module(e2e)

BEST = dict(e2e.DEFAULT)
BEST.update(dim=64, decoder="mlp", depth=2, head="physical", emb_init="random",
            emb_norm="none", dropout=0.15, wd=0.0, lr=5e-3, steps=16000)

if __name__ == "__main__":
    print("═══ A. PUSH THE GRID BOUNDARIES the round-1 search stopped at ═══", flush=True)
    cur, best = dict(BEST), e2e.inner(BEST)
    print(f"  round-1 best                    inner {best:+.4f}", flush=True)
    for axis, alts in [("steps", [24000, 32000]), ("dropout", [0.25, 0.35]), ("lr", [3e-3, 2e-3]),
                       ("dim", [96, 192]), ("depth", [3])]:
        won = None
        for v in alts:
            a = e2e.inner({**cur, axis: v})
            flag = ""
            if a > best + 1e-4: won, best, flag = v, a, "  ← kept"
            print(f"   {axis:8s}= {str(v):8s} inner {a:+.4f}{flag}", flush=True)
        if won is not None: cur[axis] = won
    print(f"  extended best inner {best:+.4f} · cfg "
          f"{ {k: cur[k] for k in ('dim','depth','dropout','lr','steps')} }", flush=True)

    print("\n═══ B. RETEST the learned astro predictor at a safe LR (round 1 gave -0.96) ═══", flush=True)
    for lr in (1e-3, 3e-4):
        a = e2e.inner({**cur, "head": "mlp", "lr": lr})
        print(f"   head=mlp lr={lr:<7g} inner {a:+.4f}", flush=True)

    print("\n═══ C. SEED ENSEMBLE — the model's seed spread is 20x the deployed model's ═══", flush=True)
    P_in = [e2e.fit(WALL_INNER, cur, seed=s) for s in (7, 11, 23, 3, 42)]
    sizes = {}
    for k in (1, 2, 3, 5):
        sizes[k] = evaluate(np.mean(P_in[:k], 0), WALL_INNER)["auc"]
        print(f"   ensemble of {k}   inner {sizes[k]:+.4f}", flush=True)
    K = max(sizes, key=lambda k: sizes[k])
    print(f"  chosen on inner: ensemble of {K}", flush=True)

    print("\n═══ OUTER WALL — fitted once with the frozen configuration ═══", flush=True)
    P_out = [e2e.fit(WALL_OUTER, cur, seed=s) for s in (7, 11, 23, 3, 42)[:max(K, 3)]]
    singles = [evaluate(p, WALL_OUTER) for p in P_out]
    ens = evaluate(np.mean(P_out[:K], 0), WALL_OUTER)
    print(f"  E2E singles      OUTER AUC median {np.median([r['auc'] for r in singles]):+.4f} "
          f"[{min(r['auc'] for r in singles):+.4f}..{max(r['auc'] for r in singles):+.4f}]", flush=True)
    print(f"  E2E ensemble({K}) OUTER AUC {ens['auc']:+.4f} · skill {ens['skill']:+.4f} · {ens['pct']:.1f}%>0", flush=True)
    print(f"  deployed v10     OUTER AUC +0.8174 · skill +0.5448 · 72.9%>0", flush=True)
    print(f"  → {'E2E ENSEMBLE WINS' if ens['auc'] > 0.8174 else 'deployed v10 holds'}", flush=True)
    json.dump({"cfg": {k: str(v) for k, v in cur.items()}, "inner": best, "ens_size": K,
               "outer_ens": ens, "outer_singles": [r["auc"] for r in singles]},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_push_result.json"), "w"), indent=1)
    print("PUSHDONE", flush=True)
