#!/usr/bin/env python3
"""FINAL EXHAUSTIVE ABLATION (operator 2026-07-24): pick the best 12-yr-AUC model, full stop.

Staged search over the whole family, sharing ONE fit implementation with the roster/factor sweep
(arxiv_sweep_rosters_factors.fit — independent yearly record base: per-topic non-zero-suffix mask,
√N year weights, 12-yr wall 314, per-topic valid-train-mean baseline):

  STAGE A — full 4×2×2×2×2 factorial on the record-8 roster (64 fits):
            arch {indep · pie · tides1 · tides2} × magnitude {|·|² · cosine-sum}
            × space {√ · log1p} × loss {L1 · L2} × 2nd-harmonic {off · on}
  STAGE B — rosters {no-mars 7 · +sun 9 · +mercury 9 · +venus 9 · all 11} on the top-3 of A
  STAGE C — distance factors {1/r · 1/r² · 1/r³} on the top-3 so far
  STAGE D — seeds {1,2,3,5,11} on the top-6 so far; WINNER = best MEDIAN 12-yr AUC across
            all its seeds (seed-median, not a seed lottery), tie-break median skill.
  FINALE  — the winner refit at the 4- and 8-year walls for the honest-wall trio.

Every fit ever run is appended to final_exhaustive.csv (tag, config, skill, %>0, auc).

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/arxiv_final_exhaustive.py
"""
import importlib.util as u, itertools, json, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
def _load(p, n):
    s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); return m
S = _load("analysis/arxivtopics/arxiv_sweep_rosters_factors.py", "S")

CSV = "analysis/arxivtopics/final_exhaustive.csv"
open(CSV, "w").write("tag,arch,magnitude,space,loss,harm2,roster,fmode,seed,skill,pct,auc\n")

CFG, AUC, SEEDS = {}, {}, {}
def key_of(kw):
    b = kw.get("bodies") or S.RECORD
    return "|".join([kw.get("arch", "indep"), "mag" if kw.get("magnitude", True) else "cos",
                     kw.get("space", "sqrt"), kw.get("lk", "l1"),
                     "h2" if kw.get("harm2", False) else "h1",
                     "+".join(b) if b != S.RECORD else "rec8", kw.get("fmode", "none")])

def ev(**kw):
    k = key_of(kw); sd = kw.get("seed", 7)
    if (k, sd) in AUC: return AUC[(k, sd)]
    s, p, a = S.bench(S.fit(**kw))
    AUC[(k, sd)] = a; CFG[k] = {x: v for x, v in kw.items() if x != "seed"}
    SEEDS.setdefault(k, {})[sd] = (s, p, a)
    b = kw.get("bodies") or S.RECORD
    open(CSV, "a").write(f"{k},{kw.get('arch','indep')},{kw.get('magnitude',True)},{kw.get('space','sqrt')},"
                         f"{kw.get('lk','l1')},{kw.get('harm2',False)},{'+'.join(b)},{kw.get('fmode','none')},"
                         f"{sd},{s:.4f},{p:.1f},{a:.4f}\n")
    print(f"  {k:52s} seed{sd:>2} skill {s:+.4f} ({p:.1f}%>0) · AUC {a:+.4f}", flush=True)
    return a

def top(nk):
    return [k for k, _ in sorted(((k, max(v[2] for v in SEEDS[k].values())) for k in SEEDS),
                                 key=lambda x: -x[1])[:nk]]

print(f"== FINAL EXHAUSTIVE · {S.Tn}×{S.n} · 12yr wall {S.W12} ==", flush=True)
print("== STAGE A · 64-config factorial (record 8) ==", flush=True)
for arch, mag, sp, lk, h2 in itertools.product(["indep", "pie", "tides1", "tides2"], [True, False],
                                               ["sqrt", "log1p"], ["l1", "l2"], [False, True]):
    ev(arch=arch, magnitude=mag, space=sp, lk=lk, harm2=h2)

print("== STAGE B · rosters on top-3 ==", flush=True)
ROSTERS = [[b for b in S.RECORD if b != "mars"], ["sun"] + S.RECORD, ["mercury"] + S.RECORD,
           ["venus"] + S.RECORD, S.ALL_BODIES]
for k in top(3):
    for r in ROSTERS: ev(bodies=r, **CFG[k])

print("== STAGE C · distance factors on top-3 ==", flush=True)
for k in top(3):
    for fm in ("1/r", "1/r2", "1/r3"):
        kw = dict(CFG[k]); kw["fmode"] = fm; ev(**kw)

print("== STAGE D · 5 extra seeds on top-6 ==", flush=True)
for k in top(6):
    for sd in (1, 2, 3, 5, 11): ev(seed=sd, **CFG[k])

med = {k: float(np.median([v[2] for v in SEEDS[k].values()])) for k in SEEDS if len(SEEDS[k]) >= 4}
league = sorted(med, key=lambda k: (-med[k], -float(np.median([v[0] for v in SEEDS[k].values()]))))
print("\n  LEAGUE (seed-MEDIAN 12yr AUC, ≥4 seeds):", flush=True)
for k in league:
    aucs = sorted(v[2] for v in SEEDS[k].values())
    print(f"    {med[k]:+.4f} med · [{aucs[0]:+.4f}..{aucs[-1]:+.4f}] · {k}", flush=True)

WIN = league[0]
print(f"\n  WINNER: {WIN}  med12 {med[WIN]:+.4f}", flush=True)
print("== FINALE · winner at the 4/8-yr walls (seed 7) ==", flush=True)
walls = {}
for w, lab in ((S.n - 4, "4yr"), (S.n - 8, "8yr"), (S.n - 12, "12yr")):
    s, p, a = S.bench(S.fit(wall=w, **CFG[WIN]), wall=w)
    walls[lab] = dict(skill=round(s, 4), pct=round(p, 1), auc=round(a, 4))
    print(f"  {lab}: skill {s:+.4f} ({p:.1f}%>0) · AUC {a:+.4f}", flush=True)
json.dump({"winner": WIN, "cfg": {x: (v if not isinstance(v, list) else v) for x, v in CFG[WIN].items()},
           "median12": med[WIN], "seeds": {str(sd): v for sd, v in SEEDS[WIN].items()}, "walls": walls},
          open("analysis/arxivtopics/final_exhaustive_winner.json", "w"), indent=1)
print("EXHAUSTIVE DONE", flush=True)
