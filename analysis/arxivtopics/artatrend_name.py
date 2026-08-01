#!/usr/bin/env python3
"""ARTATREND (operator 2026-07-27, final architecture):

    name ──LLM──▶ decoder ──▶ 7 phases   ──┐
                                            ├─▶ rotate this year's sky ──▶ step ──▶ next year's value
    current year's value ───────────────────┘                    (autoregressive, rolled 30x)

This is the design the evidence points to. Forecasting a field from its NAME ALONE scored −1.02,
because the name must then also supply the field's magnitude and a name cannot know that. Here the
name is asked only for the SHAPE (the seven tunings — the thing name-embeddings are demonstrably good
at: 81% of nearest neighbours share the parent field), while the LEVEL is carried by the
autoregressive state, seeded from the current value. Name for shape, input for scale.

    √y_{t+1} = √y_t + tanh(MLP[ √y_t , aᵢcos(θᵢ(t)−pᵢ) , aᵢsin(θᵢ(t)−pᵢ) ]) · max_step
    (p, a)   = decoder(LLM embedding of the field's name)

A held-out field is therefore predictable with NOTHING but its name and its current value — which is
exactly what a free-text box on the page can supply.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import torch as T, torch.nn as nn, importlib.util as u
_m = u.spec_from_file_location("cm", "analysis/arxivtopics/comp_metric.py"); cm = u.module_from_spec(_m); _m.loader.exec_module(cm)
DEV = "mps" if T.backends.mps.is_available() else "cpu"
BODS = CHAMPION_BODIES; BI = [BODIES_ALL.index(b) for b in BODS]; NB = len(BODS)
Z = np.load("analysis/arxivtopics/llm_embeddings.npz", allow_pickle=True)
EMB = Z["E"].astype(np.float32)

class NameTrend(nn.Module):
    def __init__(self, d, hid=128, seed=7):
        super().__init__()
        T.manual_seed(seed)
        self.dec = nn.Sequential(nn.Linear(d, hid), nn.SiLU(), nn.Linear(hid, NB*3))   # phases + amps
        self.net = nn.Sequential(nn.Linear(1+2*NB, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        with T.no_grad():
            self.dec[-1].weight.mul_(0.05); self.dec[-1].bias.zero_()
            self.dec[-1].bias[NB*2:NB*3] = -2.0
            self.net[-1].weight.mul_(0.01); self.net[-1].bias.zero_()
        self.max_step = 0.15
    def shape_from_name(self, e):
        o = self.dec(e); pv = o[:, :NB*2].reshape(-1, NB, 2)
        return T.atan2(pv[:,:,0], pv[:,:,1]), nn.functional.softplus(o[:, NB*2:NB*3])
    def sky(self, e, cth, sth):
        p, a = self.shape_from_name(e)
        cp, sp = T.cos(p), T.sin(p)
        cphi = cp[:,:,None]*cth[None] + sp[:,:,None]*sth[None]
        sphi = cp[:,:,None]*sth[None] - sp[:,:,None]*cth[None]
        return T.cat([cphi*a[:,:,None], sphi*a[:,:,None]], 1)
    def step(self, y, s):
        return T.clamp(y + T.tanh(self.net(T.cat([y[:,None], s], 1)).squeeze(-1))*self.max_step, min=0.0)
    def roll(self, y0, S, k):
        out, y = [], y0
        for h in range(k):
            y = self.step(y, S[:,:,h]); out.append(y)
        return T.stack(out, 1)

def fit(wall, seed=7, steps=1200, lr=3e-3, k=HORIZON):
    T.manual_seed(seed); np.random.seed(seed)
    TH = TH_ALL[:, BI]
    tb = lambda a: T.tensor(np.asarray(a, np.float32), device=DEV)
    cth, sth = tb(np.cos(TH).T), tb(np.sin(TH).T)
    tr = cm.TRAIN
    Ysq = tb(np.sqrt(Y[tr])); tvm = train_mask(wall)[tr]
    wy = tb(np.clip(N[:wall], 0, None)**0.75)
    m = NameTrend(EMB.shape[1], seed=seed).to(DEV)
    E = tb(EMB)
    opt = T.optim.Adam(m.parameters(), lr=lr)
    origins = [o for o in range(20, wall-k) if tvm[:, o-1].mean() > 0.5]
    rs = np.random.RandomState(seed); best, stall = np.inf, 0
    st = {kk: v.detach().clone() for kk, v in m.state_dict().items()}
    for it in range(steps):
        S = m.sky(E[T.tensor(tr, device=DEV)], cth, sth)
        loss = 0.0
        for o in rs.choice(origins, size=min(5, len(origins)), replace=False):
            r = m.roll(Ysq[:, o-1], S[:,:,o-1:o-1+k], k)
            w = tb(tvm[:, o:o+k].astype(np.float32)) * wy[None, o:o+k]
            loss = loss + ((r - Ysq[:, o:o+k]).abs()*w).sum(1).div(w.sum(1).clamp(min=1e-9)).mean()
        loss = loss/5
        if not T.isfinite(loss): opt.zero_grad(); continue
        opt.zero_grad(); loss.backward(); T.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if it % 50 == 49:
            lv = loss.item()
            if np.isfinite(lv) and lv < best-1e-7: best, stall, st = lv, 0, {kk:v.detach().clone() for kk,v in m.state_dict().items()}
            else:
                stall += 1
                if stall >= 10: break
    m.load_state_dict(st)
    with T.no_grad():                      # HELD-OUT: phases from the NAME, level from its current value
        S = m.sky(E[T.tensor(cm.HELD, device=DEV)], cth, sth)
        y0 = tb(np.sqrt(Y[cm.HELD, wall-1]))
        return (m.roll(y0, S[:,:,wall-1:wall-1+HORIZON], HORIZON)**2).cpu().numpy()

def sc(yh, wall):
    tvw = TV[cm.HELD, :wall].astype(float)
    mu = (Y[cm.HELD, :wall]*tvw).sum(1)/np.maximum(tvw.sum(1), 1.0)
    hi = min(wall+HORIZON, n); yt = Y[cm.HELD, wall:hi]; yp = yh[:, :hi-wall]
    return float(np.mean([1 - ((yt[:,h]-yp[:,h])**2).sum()/max(((yt[:,h]-mu)**2).sum(),1e-9) for h in range(hi-wall)]))

if __name__ == "__main__":
    print("═══ ArtaTrend: name→phases, autoregressive from the current value ═══", flush=True)
    ps = []
    for sd in (7, 11, 23):
        ps.append(float(np.mean([sc(fit(w, seed=sd), w) for w in cm.WALLS])))
        print(f"  seed {sd}: {ps[-1]:+.4f}", flush=True)
    mu, se = float(np.mean(ps)), float(np.std(ps, ddof=1)/np.sqrt(3))
    print(f"\n  NAME→PHASES + autoregressive   {mu:+.4f} ± {se:.4f}", flush=True)
    print(f"  name only, no current value    -1.0183", flush=True)
    print(f"  fit-alone on own history       +0.6200", flush=True)
    print(f"  persistence (carry forward)    +0.7344  ← the bar an AR model must clear", flush=True)
    json.dump({"auc": round(mu,4), "se": round(se,4), "per_seed": ps}, open("analysis/arxivtopics/artatrend_name.json","w"), indent=1)
    print("NTDONE", flush=True)
