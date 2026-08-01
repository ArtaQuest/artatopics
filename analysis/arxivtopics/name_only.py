#!/usr/bin/env python3
"""CAN A FIELD BE FORECAST FROM ITS NAME ALONE? (operator: "try embedding with SoTA LLMs")

The strongest form of the generalisation question. A held-out field's embedding is NOT inferred from
its history — it is read straight off an LLM embedding of its NAME, so the model meets the field with
no data about it whatsoever:

    "Artificial Intelligence" ──LLM──▶ e_j ──decoder──▶ 7 phases ──▶ 30-year forecast

Trained on the 80% of fields the model may see; the held-out 20% supply only their names.
Compared against the two honest alternatives on the identical split and walls.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import torch as T, torch.nn as nn, importlib.util as u
_m = u.spec_from_file_location("cm", "analysis/arxivtopics/comp_metric.py"); cm = u.module_from_spec(_m); _m.loader.exec_module(cm)
_r = u.spec_from_file_location("rh", "analysis/arxivtopics/rolling_holdout.py"); rh = u.module_from_spec(_r); _r.loader.exec_module(rh)
DEV, NB, BI = rh.DEV, rh.NB, rh.BI
Z = np.load("analysis/arxivtopics/llm_embeddings.npz", allow_pickle=True)
EMB = Z["E"].astype(np.float32); MODEL = str(Z["model"])
print(f"name embeddings: {EMB.shape} from {MODEL}", flush=True)

def fit_name(wall, seed=7, steps=6000, lr=3e-3, hid=128):
    """Decoder from the FIXED name-embedding to the receiver's parameters. Only the decoder learns."""
    T.manual_seed(seed); np.random.seed(seed)
    TH = TH_ALL[:, BI]; ne = TH.shape[0]; hz = min(wall + HORIZON, ne)
    tb = lambda a: T.tensor(np.asarray(a, np.float32), device=DEV)
    cth, sth = tb(np.cos(TH).T), tb(np.sin(TH).T)
    dec = nn.Sequential(nn.Linear(EMB.shape[1], hid), nn.SiLU(), nn.Linear(hid, NB*3+1)).to(DEV)
    with T.no_grad():
        dec[-1].weight.mul_(0.05); dec[-1].bias.zero_(); dec[-1].bias[NB*2:NB*3] = -2.0
    E = tb(EMB)
    tr = rh.TRAIN
    Ysq, W, m, _vm = rh._prep(wall, tr, 0.75, 5)
    Yt, Wt, ma = tb(Ysq), tb(W), tb(m)
    opt = T.optim.Adam(dec.parameters(), lr=lr)
    def fwd(rows):
        o = dec(E[rows]); pv = o[:, :NB*2].reshape(-1, NB, 2)
        p = T.atan2(pv[:,:,0], pv[:,:,1]); a = nn.functional.softplus(o[:, NB*2:NB*3]); b = o[:, NB*3]
        C = b[:,None] + (a*T.cos(p)) @ cth + (a*T.sin(p)) @ sth
        return T.clamp(C, min=1e-4)**2 + 1e-8
    trT = T.tensor(tr, device=DEV)
    best, stall, st = np.inf, 0, None
    for it in range(steps):
        sig = T.sqrt(fwd(trT) + 1e-8)
        per = ((sig[:, :wall] - Yt[:, :wall]).abs() * Wt).sum(1)
        d = (sig[:, wall:hz] - ma) / T.clamp(ma, min=1e-3)
        loss = (per + 0.03*(d**2).mean(1)).sum()/len(tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            lv = loss.item()
            if lv < best-1e-7: best, stall, st = lv, 0, {k:v.detach().clone() for k,v in dec.state_dict().items()}
            else:
                stall += 1
                if stall >= 10: break
    dec.load_state_dict(st)
    with T.no_grad():                      # held-out fields: NAME ONLY, no history used
        return np.clip(fwd(T.tensor(rh.HELD, device=DEV)).cpu().numpy(), 0, None)

if __name__ == "__main__":
    rh.HELD, rh.TRAIN = cm.HELD, cm.TRAIN
    per_seed = []
    for sd in (7, 11, 23):
        per_seed.append(float(np.mean([cm.score(fit_name(w, seed=sd), wall=w)["auc"] for w in cm.WALLS])))
        print(f"  seed {sd}: {per_seed[-1]:+.4f}", flush=True)
    mu, se = float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)/np.sqrt(3))
    print(f"\n  NAME ONLY (no history for the unseen field)  {mu:+.4f} ± {se:.4f}", flush=True)
    print(f"  history-inferred embedding                   +0.6069 ± 0.0070", flush=True)
    print(f"  fit-alone on the field's own history         +0.6200 ± 0.0006", flush=True)
    json.dump({"auc": round(mu,4), "se": round(se,4), "per_seed": per_seed, "model": MODEL},
              open("analysis/arxivtopics/name_only.json","w"), indent=1)
    print("NAMEDONE", flush=True)
