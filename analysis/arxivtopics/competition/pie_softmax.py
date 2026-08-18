#!/usr/bin/env python3
"""SOFT CLASSIFICATION OF NOW — given the date, what is the pie?

Not "will this field trend next year" but: hand the model a year and it returns a probability
distribution over all 251 fields, which is compared against that year's actual shares.

    score_j(t) = b_j + <w_j, f(t)>          f(t) = the sky at t, nothing else
    p(.|t)     = softmax over the 251 fields
    loss       = cross-entropy against the observed share vector s(.,t)

Because the observed target is a DISTRIBUTION rather than a label, cross-entropy against it is
exactly a multinomial logistic regression in which every (year, field) pair is an example carrying
weight s_j(t) — so the fit is convex and has one optimum, no seeds and no restarts. Minimising this
cross-entropy IS minimising KL(actual || predicted), since the entropy of the actual pie is a
constant the model cannot touch.

The model is memoryless: f(t) contains only the sky at t. It never sees a share.

Split: the LAST 20% of years are the test, everything earlier is train — future predictability, and
the model is never shown a year from the test span.

Reported against three baselines: the uniform pie, the train-mean pie (climatology — the single
best constant answer), and, for reference only, carry-forward (which HAS memory and is therefore
not a fair comparison, but shows what memory is worth).

  python3 analysis/arxivtopics/competition/pie_softmax.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from scipy.stats import spearmanr
import arxiv_fit as af

names, Yv, labels, future = af.load_lunar()
TH, _ = af.sky_lunar(labels + future)
tv = af.META["topic_valid"]
years = np.array([int(y) for y in labels]); J, n = Yv.shape
alive = tv.sum(0)
usable = np.where(alive >= 20)[0]
t0, t1 = usable[0], usable[-1]
T = np.arange(t0, t1 + 1)
S = np.clip(Yv[:, T], 0, None) * tv[:, T]
S = S / np.maximum(S.sum(0, keepdims=True), 1e-12)          # each year's pie, sums to 1
cut = int(len(T) * 0.8)
TR, TE = T[:cut], T[cut:]
print(f"years {years[T[0]]}..{years[T[-1]]} · train {years[TR[0]]}-{years[TR[-1]]} ({len(TR)}y) · "
      f"TEST {years[TE[0]]}-{years[TE[-1]]} ({len(TE)}y, the last 20%)", flush=True)

PAIRS = [(i, k) for i in range(7) for k in range(i+1, 7)]
def feats(idx, harmonics=1):
    C = [np.ones(len(idx))]
    for h in range(1, harmonics+1):
        for i in range(7):
            C += [np.cos(h*TH[idx, i]), np.sin(h*TH[idx, i])]
    for i, k in PAIRS:
        d = TH[idx, i] - TH[idx, k]
        C += [np.cos(d), np.sin(d)]
    return np.stack(C, 1)

def ce(P, Sx):
    """Cross-entropy of the predicted pie against the actual, averaged over years (nats)."""
    return float(np.mean(-(Sx * np.log(np.maximum(P, 1e-12))).sum(0)))
def kl(P, Sx):
    H = -(Sx * np.log(np.maximum(Sx, 1e-12))).sum(0)
    return float(np.mean(-(Sx*np.log(np.maximum(P,1e-12))).sum(0) - H))
def rank_rho(P, Sx):
    return float(np.mean([spearmanr(P[:, c], Sx[:, c]).statistic for c in range(Sx.shape[1])]))
def topk(P, Sx, k=10):
    hits = [len(set(np.argsort(-P[:, c])[:k]) & set(np.argsort(-Sx[:, c])[:k]))/k for c in range(Sx.shape[1])]
    return float(np.mean(hits))

Str, Ste = S[:, :cut], S[:, cut:]
BASE = {}
BASE["uniform pie"] = np.repeat((np.ones(J)/J)[:, None], len(TE), 1)
mean_pie = Str.mean(1); mean_pie = mean_pie/mean_pie.sum()
BASE["train-mean pie (climatology)"] = np.repeat(mean_pie[:, None], len(TE), 1)
lastpie = Str[:, -1]/max(Str[:, -1].sum(), 1e-12)
BASE["carry-forward (HAS MEMORY)"] = np.repeat(lastpie[:, None], len(TE), 1)

def fit_softmax(harm, lam):
    """Convex fit of B (J x D) by full-batch gradient descent on the weighted cross-entropy."""
    Ftr, Fte = feats(TR, harm), feats(TE, harm)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    mu[0], sd[0] = 0.0, 1.0
    Ftr = (Ftr-mu)/sd; Fte = (Fte-mu)/sd
    D = Ftr.shape[1]
    Bm = np.zeros((J, D))
    Bm[:, 0] = np.log(np.maximum(mean_pie, 1e-9))            # start at climatology
    lr = 0.5
    for it in range(4000):
        Z = Bm @ Ftr.T
        Z -= Z.max(0, keepdims=True)
        P = np.exp(Z); P /= P.sum(0, keepdims=True)
        G = (P - Str) @ Ftr / len(TR)
        G[:, 1:] += lam * Bm[:, 1:]
        Bm -= lr * G
    Z = Bm @ Fte.T; Z -= Z.max(0, keepdims=True)
    P = np.exp(Z); P /= P.sum(0, keepdims=True)
    Ztr = Bm @ Ftr.T; Ztr -= Ztr.max(0, keepdims=True)
    Ptr = np.exp(Ztr); Ptr /= Ptr.sum(0, keepdims=True)
    return P, Ptr

print(f"\n— baselines on the held-out {len(TE)} years:", flush=True)
print(f"  {'model':<40}{'cross-ent':>11}{'KL':>9}{'rank rho':>10}{'top-10':>9}", flush=True)
res = {}
for nm, P in BASE.items():
    res[nm] = dict(ce=round(ce(P,Ste),4), kl=round(kl(P,Ste),4), rho=round(rank_rho(P,Ste),4), top10=round(topk(P,Ste),3))
    print(f"  {nm:<40}{res[nm]['ce']:>11.4f}{res[nm]['kl']:>9.4f}{res[nm]['rho']:>10.4f}{res[nm]['top10']:>9.3f}", flush=True)

print(f"\n— the sky softmax (memoryless: the date is the only input):", flush=True)
best = None
for harm in (1, 2, 3):
    for lam in (1e-4, 1e-3, 1e-2, 1e-1):
        P, Ptr = fit_softmax(harm, lam)
        c_tr = ce(Ptr, Str); c_te = ce(P, Ste)
        tag = f"sky softmax (harmonics {harm}, ridge {lam:g})"
        res[tag] = dict(ce=round(c_te,4), kl=round(kl(P,Ste),4), rho=round(rank_rho(P,Ste),4),
                        top10=round(topk(P,Ste),3), train_ce=round(c_tr,4))
        print(f"  {tag:<40}{c_te:>11.4f}{kl(P,Ste):>9.4f}{rank_rho(P,Ste):>10.4f}{topk(P,Ste):>9.3f}"
              f"   (train CE {c_tr:.4f})", flush=True)
        if best is None or c_tr < best[0]: best = (c_tr, tag, c_te, kl(P,Ste), rank_rho(P,Ste), topk(P,Ste))
clim = res["train-mean pie (climatology)"]["ce"]
climR = res["train-mean pie (climatology)"]["rho"]
c_tr, tag, c_te, k_te, r_te, t_te = best
print(f"\n— SELECTED BY TRAIN CROSS-ENTROPY, judged on the held-out years:", flush=True)
print(f"  chosen: {tag}", flush=True)
print(f"    train CE {c_tr:.4f}  ->  TEST CE {c_te:.4f}   (KL {k_te:.4f} · rank rho {r_te:.4f} · top-10 {t_te:.3f})", flush=True)
print(f"  climatology, the best constant pie:  TEST CE {clim:.4f}   (rank rho {climR:.4f})", flush=True)
d = clim - c_te
print(f"  the sky is worth {d:+.4f} nats — {'BETTER' if d>0 else 'WORSE'} than knowing only the average pie", flush=True)
bt = min((v['ce'], k) for k,v in res.items() if k.startswith('sky'))
print(f"\n  (for contrast: the sky model that happens to score best ON TEST is {bt[1]} at {bt[0]:.4f}."
      f" Choosing it would mean selecting on the answer, which is why the line above uses train CE.)", flush=True)
res["_selected_by_train"] = {"model": tag, "train_ce": round(c_tr,4), "test_ce": round(c_te,4),
                             "climatology_test_ce": clim, "sky_gain_nats": round(d,4)}
json.dump(res, open(os.path.expanduser("~/.artaquest-dev/artacomp/piecomp/softmax.json"), "w"), indent=1)
