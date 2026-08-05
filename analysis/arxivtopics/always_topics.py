#!/usr/bin/env python3
"""ONLY THE TOPICS THAT HAVE ALWAYS EXISTED (operator 2026-08-04).

No topic in the record is valid from 1700 — the citation record effectively dawns in the 1800s, and
the emergence cohorts are: 192 topics valid from the 1800s, 53 more from 1900-1949, 6 from 1950-79.
"Always existed" therefore means the 192 pre-1900 topics; "appeared recently" is the 59 that emerged
in the 1900s. The distribution is renormalised over the 192, so q(·|t) is again a proper
distribution over the topic set being modelled.

The point of the cut: the emergent topics carry exactly the secular drift (a new field's share
climbing from zero) that a function of planetary angles cannot reproduce and that carry-forward
persistence copies for free. On the always-existing set the arena is as stationary as this data
gets — if the sky family is ever going to beat persistence at the distribution goal, it is here.

  python3 analysis/arxivtopics/always_topics.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
import arxiv_fit as af

names, Yv, labels_y, future = af.load_lunar()
TH, _ = af.sky_lunar(labels_y + future)
n = Yv.shape[1]; ne = TH.shape[0]
tv = af.META["topic_valid"]
start = tv.argmax(1)
SUB = np.where(np.array([int(labels_y[s]) for s in start]) < 1900)[0]
J = len(SUB)
Qs = Yv[SUB] / 100.0
Q = Qs / np.maximum(Qs.sum(0, keepdims=True), 1e-12)         # renormalised over the 192
NW = np.clip(af.META["evidence"], 0, None) ** 0.75
WALLS = list(range(n - 63, n - 29, 3))


def features(kind):
    if kind == "sin":    return np.concatenate([np.ones((ne, 1)), np.sin(TH)], 1)
    return np.concatenate([np.ones((ne, 1)), np.sin(TH), np.cos(TH)], 1)


def solve(Z, wall):
    """Anchored analytic amplitude solve on the subset (same construction as multihead_ce)."""
    d = Z.shape[1]
    Zt = Z[:wall]; w = NW[:wall] / NW[:wall].sum()
    S = np.sqrt(Q[:, :wall])
    G0 = Zt.T @ (Zt * w[:, None]) + 1e-10 * np.eye(d)
    B0 = (S * w[None, :]) @ Zt
    hz = min(wall + af.HORIZON, ne)
    Za = Z[wall:hz]; Ga = Za.T @ Za
    m = np.maximum(S[:, max(0, wall - af.ANCHOR_K):].mean(1), 1e-4)
    aw = af.LAM_HORIZON / (m ** 2) / max(hz - wall, 1)
    G = G0[None] + aw[:, None, None] * Ga[None]
    B = B0 + (aw * m)[:, None] * Za.sum(0)[None, :]
    return np.linalg.solve(G, B[:, :, None])[:, :, 0]


def predict(U, Z):
    A = U @ Z.T; Y = A ** 2
    return Y / np.maximum(Y.sum(0, keepdims=True), 1e-12)


def kl_at(P, wall):
    hi = min(wall + af.HORIZON, n)
    w = NW[wall:hi]; w = w / w.sum()
    Hq = float(-(np.where(Q[:, wall:hi] > 0, Q[:, wall:hi] * np.log(np.clip(Q[:, wall:hi], 1e-12, None)), 0)).sum(0) @ w)
    pd = np.clip(P[:, wall:hi], 1e-12, None); pd = pd / pd.sum(0, keepdims=True)
    return float(-(Q[:, wall:hi] * np.log(pd)).sum(0) @ w) - Hq


def main():
    print(f"═══ ALWAYS-EXISTING TOPICS ONLY · {J} of 251 (valid since the 1800s) ═══", flush=True)
    print(f"    their share of 2025 citations before renormalising: {Qs[:, -1].sum()*100:.1f}%", flush=True)
    t0 = time.time()
    board = {}
    # baselines
    board["persistence"] = [kl_at(np.repeat(Q[:, w-1:w], 1, 1) * np.ones((J, n)), w) for w in WALLS]
    board["train-mean"] = []
    for w in WALLS:
        tw = NW[:w]
        md = (Q[:, :w] * tw[None]).sum(1, keepdims=True) / tw.sum()
        board["train-mean"].append(kl_at(np.repeat(md, n, 1), w))
    # the per-topic record model, restricted to the subset and renormalised
    board["record renormalised"] = []
    for w in WALLS:
        Pr, _ = af.fit_final(Yv, TH, w)
        Ps = Pr[SUB] / np.maximum(Pr[SUB].sum(0, keepdims=True), 1e-12)
        board["record renormalised"].append(kl_at(Ps, w))
    # the Born family, analytic anchored
    for kind, lab in (("sin", "Born sin (no phases)"), ("sincos", "Born free p_ji")):
        Z = features(kind)
        board[lab] = [kl_at(predict(solve(Z, w), Z), w) for w in WALLS]
    print(f"  [{time.time()-t0:.0f}s] held-out KL over twelve origins (lower better):", flush=True)
    order = sorted(board, key=lambda k: np.mean(board[k]))
    for k in order:
        r = np.array(board[k])
        print(f"    {k:24s} mean {r.mean():.4f} · 1996 {r[-1]:.4f}   per-origin " +
              " ".join(f"{v:.3f}" for v in r), flush=True)
    best_sky = min((k for k in board if k != "persistence" and k != "train-mean"), key=lambda k: np.mean(board[k]))
    d = np.array(board[best_sky]) - np.array(board["persistence"])
    print(f"\n  best sky model ({best_sky}) − persistence: {d.mean():+.4f} · wins {int((d<0).sum())}/12", flush=True)
    json.dump({"subset": int(J), "definition": "topic_valid start < 1900",
               "board": {k: [round(float(v), 4) for v in board[k]] for k in board},
               "means": {k: round(float(np.mean(board[k])), 4) for k in board}},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "always_topics.json"), "w"), indent=1)
    print("ALWDONE", flush=True)


if __name__ == "__main__":
    main()
