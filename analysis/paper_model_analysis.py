#!/usr/bin/env python3
"""Score every atlas field with EXACTLY the model of paper #17 ("Reading a Competition's Ceiling
off Two Agents' Convergence", journal/skills-ceiling) — the method-ladder's winning rung (g):

    an equal blend of TWO decorrelated ensemble lineages, each a per-field OOF-NNLS stack of a
    member library (lineage A = linear/PCHIP/Akima/Gaussian/kNN, lineage B = Akima/spline/wide-
    Gaussian/kNN5, both + the field's own CV-tuned harmonic + a fixed annual harmonic), with the
    per-field NNLS weights shrunk toward the POOLED global stack at the paper's shrink-sweep peak
    (SHRINK = 0.20), predictions clipped to [0, 100].

All machinery is IMPORTED from the paper's reproduction script (journal/skills-ceiling/code/
reproduce.py) so the model here cannot drift from the published one: dedup, LIB_A/LIB_B,
cv_period/make_harm/m_annual, loo_oof, r2, HOLDOUT_FRAC.

Two outputs per field (series via trends_fit.load_y, 258 months = 2004→now minus the last year):

  paperR2  holdout R² under the paper's protocol — hide a seeded random 20% of the months
           (deterministic per-field seed = crc32(key)), fit the whole rung-(g) pipeline on the
           visible 80%, score the blend on the hidden 20%. The fleet mean/median is directly
           comparable to the paper's local ceiling (mean 0.8813, median 0.9385); the atlas trains
           a little denser (206/258 visible vs the paper's 165/258), so slightly higher is expected.
  fitted   the CHART fit — the same rung-(g) blend fitted on ALL months (no holdout), evaluated at
           every month index, clipped 0-100 + rounded 2dp → written into the registries as
           r['fitted'] (the field page's gold curve; the sidereal original stays in sid_fitted).

Writes analysis/_paper_model.json {key: {"paperR2": …}} and updates analysis/_fields_weekly.json +
analysis/_topics_weekly.json in place (keys mapped back through merge_atlas's '-t' collision rule).
Run AFTER reclassify_best_fit.py and BEFORE merge_atlas in the rebuild chain.

  python3 analysis/paper_model_analysis.py             # score the atlas + write registries
  python3 analysis/paper_model_analysis.py --validate  # rerun the PAPER's own 500-topic geometry
                                                       # through this runner — must reproduce the
                                                       # published rung (g): mean 0.8813, median 0.9385
"""
import importlib.util as _u
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PAPER_CODE = os.path.join(REPO, "journal", "skills-ceiling", "code", "reproduce.py")

ALLFIT = os.path.join(REPO, "analysis", "_allfit.json")
REG_SK = os.path.join(REPO, "analysis", "_fields_weekly.json")
REG_TP = os.path.join(REPO, "analysis", "_topics_weekly.json")
OUT = os.path.join(REPO, "analysis", "_paper_model.json")
SHRINK = 0.20        # the paper's shrink-sweep peak (reproduce.py → _results.json shrink_peak)
PAPER_MEAN, PAPER_MEDIAN = 0.8813, 0.9385   # _results.json ladder_g_blend / g_median_r2


def _load(name, path):
    spec = _u.spec_from_file_location(name, path)
    mod = _u.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Loaded at module level so spawned workers (which re-import this module) get it too.
paper = _load("skills_ceiling_reproduce", PAPER_CODE)


def _lineage_parts(um, uy, mem, q):
    """The per-topic body of the paper's lineage_fit(): leave-one-month-out OOF → per-field NNLS
    weights, plus the member-prediction matrix at the query months q. Returns (w, full, oofX, oofY);
    oofX/oofY feed the pooled GLOBAL stack (None when the field is too short to stack, as in the paper)."""
    if len(um) >= 6:
        oof = paper.loo_oof(um, uy, mem)
        w, _ = nnls(oof, uy)
        if w.sum() < 1e-9:
            w = np.ones(len(mem))
        ox, oy = oof, uy
    else:
        w, ox, oy = np.ones(len(mem)), None, None
    return w / w.sum(), np.column_stack([f(um, uy, q) for f in mem]), ox, oy


def field_worker(job):
    """Phase 1 (parallel, per field): everything shrink/global-independent, per pass × lineage.
    The split (tr/hold) is decided by the CALLER so the atlas (per-field crc32 seed) and the
    --validate rerun (the paper's single sequential rng) share one code path."""
    key, y, mon, tr, hold, do_full = job
    y = np.asarray(y, float)
    mon = np.asarray(mon)
    res = {"y_hold": y[hold]}
    passes = []
    um, uy = paper.dedup(mon[tr], y[tr])
    passes.append(("hold", um, uy, paper.cv_period(um, uy), mon[hold]))   # the paper's protocol
    if do_full:                                                           # the chart fit
        umf, uyf = paper.dedup(mon, y)
        passes.append(("full", umf, uyf, paper.cv_period(umf, uyf), mon))
    for pas, u_, y_, pn, q in passes:
        for lin, base in (("A", paper.LIB_A), ("B", paper.LIB_B)):
            mem = list(base) + [paper.make_harm(*pn), paper.m_annual]
            res[(pas, lin)] = _lineage_parts(u_, y_, mem, q)
    return key, res


def run_fleet(jobs, workers=None):
    """Phase 1 in parallel, then phase 2 (the paper's pooled global stack + shrink + equal blend).
    Returns ({key: holdout blend pred}, {key: full blend pred or None}, {key: y_hold})."""
    keys = [j[0] for j in jobs]
    results = {}
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (key, res) in enumerate(ex.map(field_worker, jobs, chunksize=4), 1):
            results[key] = res
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)} fields fitted…", flush=True)
    do_full = ("full", "A") in results[keys[0]]
    blends = {"hold": {}, "full": {}}
    for pas in ("hold", "full") if do_full else ("hold",):
        preds = {}
        for lin in ("A", "B"):
            # the paper's GLOBAL stack: one pooled NNLS over every field's OOF (deterministic key order)
            allX = np.vstack([results[k][(pas, lin)][2] for k in keys
                              if results[k][(pas, lin)][2] is not None])
            ally = np.concatenate([results[k][(pas, lin)][3] for k in keys
                                   if results[k][(pas, lin)][3] is not None])
            gw, _ = nnls(allX, ally)
            gw = gw / gw.sum()
            for k in keys:   # the paper's lineage_apply: per-lineage predictions are clipped too
                w, full, _, _ = results[k][(pas, lin)]
                preds.setdefault(k, {})[lin] = np.clip(full @ ((1 - SHRINK) * w + SHRINK * gw), 0.0, 100.0)
        for k in keys:   # rung (g): the equal two-lineage blend, clipped like the paper
            blends[pas][k] = np.clip(0.5 * preds[k]["A"] + 0.5 * preds[k]["B"], 0.0, 100.0)
    return blends["hold"], blends["full"], {k: results[k]["y_hold"] for k in keys}


def split_paper(rng, n):
    """The paper's holdout rule: a random 20% of the months (rng.permutation; ≥1)."""
    perm = rng.permutation(n)
    n_hold = max(1, int(round(paper.HOLDOUT_FRAC * n)))
    return perm[n_hold:], perm[:n_hold]                                   # (tr, hold)


def validate():
    """Rerun the PAPER's own 500 competition topics (decoded months, the paper's single sequential
    rng split) through this runner — it must reproduce the published rung (g) numbers exactly."""
    code_dir = os.path.dirname(PAPER_CODE)
    os.chdir(code_dir)                    # the loaders use bare relative paths to the cached files
    months, grid = paper.load_ephemeris()
    train = paper.load_train()
    rng = np.random.default_rng(paper.SEED)
    jobs = []
    for s, df in train.items():           # sorted order — the paper consumes ONE rng sequentially
        y = df["target"].to_numpy(float)
        mon, _ = paper.decode_months(df[paper.BODIES].to_numpy(int), grid)
        tr, hold = split_paper(rng, len(df))
        jobs.append((s, y, mon, tr, hold, False))
    print(f"[validate] {len(jobs)} competition topics — rerunning the paper's rung (g)…")
    hold_pred, _, y_hold = run_fleet(jobs)
    r2s = np.array([paper.r2(y_hold[k], hold_pred[k]) for k, *_ in jobs])
    print(f"[validate] rung (g) rerun: mean {r2s.mean():.4f} (paper {PAPER_MEAN}), "
          f"median {np.median(r2s):.4f} (paper {PAPER_MEDIAN})")
    ok = abs(r2s.mean() - PAPER_MEAN) < 5e-4 and abs(np.median(r2s) - PAPER_MEDIAN) < 5e-4
    print("[validate] " + ("MATCHES the published numbers." if ok else "MISMATCH — pipeline drifted!"))
    return 0 if ok else 1


def main():
    os.chdir(REPO)
    tf = _load("tf", os.path.join(HERE, "trends_fit.py"))
    reg = json.load(open(ALLFIT))
    jobs, skip = [], 0
    for key, rec in reg.items():
        _, y = tf.load_y(rec.get("label", key))
        if y is None or len(y) < 36:
            skip += 1
            continue
        y = np.asarray(y, float)
        rng = np.random.default_rng(zlib.crc32(key.encode()))             # deterministic per field
        tr, hold = split_paper(rng, len(y))
        jobs.append((key, y, np.arange(len(y)), tr, hold, True))
    print(f"[paper-model] {len(jobs)} fields with a usable series ({skip} skipped of {len(reg)} keys)")
    hold_pred, full_pred, y_hold = run_fleet(jobs)

    keys = [j[0] for j in jobs]
    out, fitted = {}, {}
    for k in keys:
        out[k] = {"paperR2": round(float(paper.r2(y_hold[k], hold_pred[k])), 4)}
        fitted[k] = [round(float(v), 2) for v in full_pred[k]]
    json.dump(out, open(OUT, "w"), indent=0)
    r2s = np.array([out[k]["paperR2"] for k in keys])
    print(f"[paper-model] fleet holdout R² — mean {r2s.mean():.4f}  median {np.median(r2s):.4f}  "
          f"(paper rung (g): mean {PAPER_MEAN}, median {PAPER_MEDIAN})")
    print(f"[paper-model] min {r2s.min():.4f} · {(r2s < 0).sum()} fields < 0 · "
          f"{(r2s > 0.9).sum()} fields > 0.9 → {OUT}")

    # ── write the chart fit into the SOURCE registries (undo merge_atlas's '-t' suffix rule) ──
    sk = json.load(open(REG_SK))
    tp = json.load(open(REG_TP))
    mapping, assigned = {}, set()
    for k in sk:
        mapping[k] = ("sk", k)
        assigned.add(k)
    for k in tp:
        kk = k
        while kk in assigned:
            kk += "-t"
        assigned.add(kk)
        mapping[kk] = ("tp", k)
    n_sk = n_tp = 0
    for k, vec in fitted.items():
        src, orig = mapping[k]
        if src == "sk":
            sk[orig]["fitted"] = vec
            n_sk += 1
        else:
            tp[orig]["fitted"] = vec
            n_tp += 1
    json.dump(sk, open(REG_SK, "w"), indent=0)
    json.dump(tp, open(REG_TP, "w"), indent=0)
    print(f"[paper-model] chart fit written: {n_sk} skills + {n_tp} topics (sid_fitted backups untouched)")
    # chart-fit sanity: R² between the stored fit and the actual series for a probe field
    probe = "machine-learning" if "machine-learning" in fitted else keys[0]
    _, y = tf.load_y(reg[probe].get("label", probe))
    print(f"[paper-model] probe {probe}: paperR2 {out[probe]['paperR2']:.4f} · "
          f"chart-fit R² vs actual {paper.r2(np.asarray(y, float), np.asarray(fitted[probe], float)):.4f}")


if __name__ == "__main__":
    sys.exit(validate()) if "--validate" in sys.argv else main()
