#!/usr/bin/env python3
"""IS THERE ANY CLUSTER STRUCTURE TO CLUSTER? -- a forecast-free look at the baseline's own spectra.

The AUC curves in archetypes.py answer "does a K-archetype dictionary forecast as well as 251 free
fits". This asks the prior question, with no forecast in it at all: do the 251 fitted arrow-vectors
a_j actually LIE near a few directions? If they are spread over the sphere, no dictionary of any
fitting procedure can be small, and the AUC result is explained rather than merely observed.

Three forecast-free measurements at the 1996 wall (train years only, so nothing here sees the test):
  1. PCA of the unit-normalised spectra -- how many directions carry the variance.
  2. k-means inertia on the projective metric d(u,v) = 1-|u.v| vs the same statistic on spectra with
     the topic labels SHUFFLED (a null with identical marginals but no field-specific structure).
     If real clusters exist, real inertia must fall well below the shuffled null.
  3. The sign-pattern census: how the 251 topics spread over the 64 sign classes, and whether the
     class a topic lands in has anything to do with its OpenAlex domain (mutual information vs a
     permutation null).

  python3 analysis/arxivtopics/archetypes_geometry.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import archetypes as AR
from collections import Counter

W96 = AR.WALLS[-1]
NB, Tn = AR.NB, AR.Tn


def kmeans_proj(U, K, iters=50):
    """Deterministic k-means under d = 1-|u.v| (sign is a gauge). Farthest-point init, as in the fit."""
    C = AR.seed(U, K)
    for _ in range(iters):
        S = np.abs(U @ C.T)
        lab = S.argmax(1)
        newC = C.copy()
        for k in range(K):
            idx = np.where(lab == k)[0]
            if idx.size == 0:
                continue
            V = U[idx] * np.sign(U[idx] @ C[k])[:, None]     # align to the centre's gauge
            m = V.mean(0); nz = np.linalg.norm(m)
            if nz > 1e-12:
                newC[k] = m / nz
        if np.allclose(newC, C):
            break
        C = newC
    S = np.abs(U @ C.T)
    return float((1.0 - S.max(1)).mean()), S.argmax(1)


if __name__ == "__main__":
    P = AR.Wall(W96)
    _, A, g, b = AR.baseline_spectra(P)
    U = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    print("=== GEOMETRY OF THE 251 FITTED SPECTRA (1996 wall, train years only) ===", flush=True)

    # 1. PCA on the sign-gauge-free second moment
    M = U.T @ U / Tn
    ev = np.sort(np.linalg.eigvalsh(M))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    print("  PCA of the unit spectra (second moment, gauge-free):", flush=True)
    print("    eigenvalues " + " ".join(f"{v:.3f}" for v in ev), flush=True)
    print("    cumulative  " + " ".join(f"{v:.3f}" for v in cum), flush=True)
    print(f"    dims for 90% of the variance: {int(np.searchsorted(cum, 0.90)) + 1} of {NB}",
          flush=True)

    # 2. clustering tightness vs a label-shuffled null
    print("\n  k-means inertia under d=1-|u.v|  (real vs column-shuffled null, 20 shuffles):",
          flush=True)
    rng = np.random.RandomState(0)
    rows = {}
    for K in [2, 3, 4, 6, 8, 12, 20]:
        real, lab = kmeans_proj(U, K)
        null = []
        for s in range(20):
            Ush = np.stack([U[rng.permutation(Tn), i] for i in range(NB)], 1)
            Ush /= np.maximum(np.linalg.norm(Ush, axis=1, keepdims=True), 1e-12)
            null.append(kmeans_proj(Ush, K)[0])
        z = (real - np.mean(null)) / max(np.std(null), 1e-9)
        rows[K] = dict(real=real, null_mean=float(np.mean(null)), null_sd=float(np.std(null)),
                       z=float(z), sizes=np.bincount(lab, minlength=K).tolist())
        print(f"    K={K:<3d} real {real:.4f}   null {np.mean(null):.4f} +- {np.std(null):.4f}"
              f"   z {z:+.2f}   sizes {np.bincount(lab, minlength=K).tolist()}", flush=True)

    # 3. sign-pattern census and whether it tracks the OpenAlex domain
    pat = np.sign(A); pat[pat == 0] = 1
    canon = (pat * pat[:, :1]).astype(int)
    keys = [tuple(r) for r in canon]
    cnt = Counter(keys)
    sizes = sorted(cnt.values(), reverse=True)
    print(f"\n  sign-pattern census: {len(cnt)} of 64 classes occupied; largest {sizes[0]}, "
          f"top-5 {sum(sizes[:5])}/{Tn}, singletons {sum(1 for v in sizes if v == 1)}", flush=True)
    ent = -sum((v / Tn) * np.log2(v / Tn) for v in sizes)
    print(f"    entropy of the sign class {ent:.2f} bits (uniform over 64 would be 6.00)", flush=True)

    dom = [AR.af.META["domain"][n] for n in AR.names]
    doms = sorted(set(dom))
    kidx = {k: i for i, k in enumerate(sorted(cnt))}
    lab = np.array([kidx[k] for k in keys])
    dlab = np.array([doms.index(d) for d in dom])

    def mi(a, b):
        na, nb = a.max() + 1, b.max() + 1
        J = np.zeros((na, nb))
        for x, y in zip(a, b):
            J[x, y] += 1
        J /= J.sum()
        pa, pb = J.sum(1, keepdims=True), J.sum(0, keepdims=True)
        nzm = J > 0
        return float((J[nzm] * np.log2(J[nzm] / (pa @ pb)[nzm])).sum())

    real_mi = mi(lab, dlab)
    nulls = [mi(lab, dlab[rng.permutation(Tn)]) for _ in range(200)]
    print(f"    MI(sign class ; OpenAlex domain) = {real_mi:.3f} bits   "
          f"permutation null {np.mean(nulls):.3f} +- {np.std(nulls):.3f}  "
          f"z {(real_mi-np.mean(nulls))/max(np.std(nulls),1e-9):+.2f}", flush=True)

    # which bodies lead, and do domains agree?
    lead = np.abs(A).argmax(1)
    print("\n  dominant body by domain:", flush=True)
    for d in doms:
        idx = [i for i in range(Tn) if dom[i] == d]
        c = Counter(AR.BODIES[lead[i]] for i in idx)
        print(f"    {d[:34]:34s} n={len(idx):<4d} " +
              " ".join(f"{b} {v}" for b, v in c.most_common(4)), flush=True)

    json.dump(dict(wall=int(AR.YEARS[W96]), eigenvalues=ev.tolist(), cumulative=cum.tolist(),
                   dims_for_90pct=int(np.searchsorted(cum, 0.90)) + 1,
                   kmeans=rows, sign_classes_occupied=len(cnt), sign_class_sizes=sizes,
                   sign_entropy_bits=float(ent), mi_signclass_domain=real_mi,
                   mi_null_mean=float(np.mean(nulls)), mi_null_sd=float(np.std(nulls))),
              open("analysis/arxivtopics/archetypes_geometry.json", "w"), indent=1)
    print("\n  saved -> analysis/arxivtopics/archetypes_geometry.json", flush=True)
    print("GEOMDONE", flush=True)
