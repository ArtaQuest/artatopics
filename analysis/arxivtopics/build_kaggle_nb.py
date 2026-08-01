#!/usr/bin/env python3
"""Build a SELF-CONTAINED Kaggle GPU notebook for the big-model sweep (operator 2026-07-26:
"move to kaggle and run a bigger models on gpu").

Self-contained on purpose: the four data files (the published citations rail + the sidereal ephemeris)
are gzip+base64'd straight into the notebook (436 KB), so the kernel needs NO attached dataset and NO
internet. That matches the platform's own reproducibility rule — a notebook that needs the network to
reproduce is not reproducible — and it means the run can be repeated by anyone with the file alone.

WHAT THE GPU IS FOR. The blocker is not model size, it is VARIANCE: the embedding model's held-out AUC
moved 0.278..0.562 on the training seed alone, and after stabilising (deterministic init + EMA) the
5-wall spread is still 0.0235 against a ~0.013 difference we need to resolve. A GPU buys MANY SEEDS,
which is the only thing that actually shrinks a standard error. So the sweep is capacity x seeds:
every configuration is run at 8 seeds and reported with a proper SE, and the capacity ladder finally
goes as wide as the architecture deserves (dim up to 512, decoder up to 4 layers).

  python3 analysis/arxivtopics/build_kaggle_nb.py        → writes kaggle_bigmodel.ipynb
"""
import base64, gzip, json, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "rail_citations_received_yearly.csv": "analysis/citations/rail_citations_received_yearly.csv",
    "rail_works_yearly.csv": "analysis/citations/rail_works_yearly.csv",
    "rail_cited_by_sum_yearly.csv": "analysis/citations/rail_cited_by_sum_yearly.csv",
    "_ephemeris_yearly.csv": "analysis/arxivtopics/_ephemeris_yearly.csv",
}

blobs = {n: base64.b64encode(gzip.compress(open(os.path.join(REPO, p), "rb").read(), 9)).decode()
         for n, p in FILES.items()}

CELL_DATA = f'''# ── the data, embedded (gzip+base64) — no attached dataset, no internet ──────────────────
import base64, gzip, os, json, time
BLOBS = {json.dumps(blobs)}
os.makedirs("data", exist_ok=True)
for name, b in BLOBS.items():
    open(os.path.join("data", name), "wb").write(gzip.decompress(base64.b64decode(b)))
print("materialised:", {{n: os.path.getsize("data/" + n) for n in BLOBS}})
'''

CELL_HARNESS = '''
# ── harness: data, target, walls, metric (identical to the local one) ────────────────────
import numpy as np, pandas as pd, torch as T, torch.nn as nn
DEV = "cuda" if T.cuda.is_available() else "cpu"
print("device:", DEV, T.cuda.get_device_name(0) if DEV == "cuda" else "")

_c = pd.read_csv("data/rail_citations_received_yearly.csv")
YEARS = [int(x) for x in _c.columns if x[0].isdigit()]
_yc = [str(y) for y in YEARS]
NAMES = list(_c.subfield)
CITES = _c[_yc].to_numpy(float)
N = CITES.sum(0)
Y = 100.0 * CITES / np.maximum(N[None, :], 1.0)      # share of the year's citations
Tn, n = Y.shape
HORIZON = 30

_z = CITES > 0
def train_mask(wall):
    m = np.ones((Tn, wall), bool)
    for i in range(wall - 2, -1, -1): m[:, i] = _z[:, i] & m[:, i + 1]
    return m
TV = np.ones_like(_z, bool)
for _i in range(n - 2, -1, -1): TV[:, _i] = _z[:, _i] & TV[:, _i + 1]

BODIES_ALL = ["sun","mercury","venus","mars","jupiter","saturn","uranus","neptune","pluto","node","chiron"]
BODS = ["mars","jupiter","saturn","uranus","neptune","pluto","node"]
BI = [BODIES_ALL.index(b) for b in BODS]; NB = len(BODS)
_E = pd.read_csv("data/_ephemeris_yearly.csv").set_index("Time"); _E.index = _E.index.astype(str)
_EY = [str(y) for y in range(YEARS[0], 2056)]
TH_ALL = np.stack([np.deg2rad(_E[f"{b}_lon"].loc[_EY].to_numpy(float)) for b in BODIES_ALL], 1)

# competition protocol: 20% of fields held out entirely, five 30-year origins
SPLIT_SEED = 0
_perm = np.random.RandomState(SPLIT_SEED).permutation(Tn)
N_HELD = int(round(0.20 * Tn))
HELD = np.sort(_perm[:N_HELD]); TRAIN = np.sort(_perm[N_HELD:])
WALLS = [n - 60, n - 52, n - 45, n - 37, n - 30]
print(f"{len(HELD)} held-out fields, {len(TRAIN)} training fields; walls "
      + ", ".join(str(YEARS[w]) for w in WALLS))

def score(yh, wall, held=HELD):
    tvw = TV[held, :wall].astype(float)
    mu = (Y[held, :wall] * tvw).sum(1) / np.maximum(tvw.sum(1), 1.0)
    hi = min(wall + HORIZON, n)
    yt, yp = Y[held, wall:hi], yh[:, wall:hi]
    curve = [1 - ((yt[:, h] - yp[:, h]) ** 2).sum() / max(((yt[:, h] - mu) ** 2).sum(), 1e-9)
             for h in range(hi - wall)]
    return float(np.mean(curve))
'''

CELL_MODEL = '''
# ── the model: topic embedding → shared decoder → 7 phases → rotated sky → share ─────────
LAM_H, ANCHOR_K, WEXP = 0.03, 5, 0.75

def prep(wall, rows):
    Ysq = np.sqrt(Y[rows]); tv = train_mask(wall)[rows].astype(np.float32)
    wy = np.clip(N[:wall], 0, None) ** WEXP
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    Wa = np.zeros_like(W); Wa[:, wall - ANCHOR_K:] = (tv * wy[None])[:, wall - ANCHOR_K:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    return Ysq, W, (Ysq[:, :wall] * Wa).sum(1)[:, None]

def det_features(rows, wall):
    tv = train_mask(wall)[rows].astype(float); ys = np.sqrt(Y[rows, :wall])
    w = tv / np.maximum(tv.sum(1, keepdims=True), 1e-9)
    lvl = (ys * w).sum(1); t = np.arange(wall)[None, :] / wall
    tbar = (t * w).sum(1, keepdims=True)
    trend = ((t - tbar) * (ys - lvl[:, None]) * w).sum(1) / np.maximum(((t - tbar) ** 2 * w).sum(1), 1e-9)
    var = np.sqrt(np.maximum(((ys - lvl[:, None]) ** 2 * w).sum(1), 0)); age = tv.sum(1) / wall
    F = np.stack([lvl, trend, var, age, np.log1p(lvl * 1e3)], 1)
    return ((F - F.mean(0)) / np.maximum(F.std(0), 1e-9)).astype(np.float32)

class Shared(nn.Module):
    def __init__(self, dim, depth, width, nrows, dropout):
        super().__init__()
        self.emb = nn.Parameter(T.zeros(nrows, dim))
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
        layers, i = [], dim
        for _ in range(depth): layers += [nn.Linear(i, width), nn.SiLU()]; i = width
        layers += [nn.Linear(i, NB * 3 + 1)]
        self.dec = nn.Sequential(*layers)
        with T.no_grad():
            self.dec[-1].weight.mul_(0.05); self.dec[-1].bias.zero_()
            self.dec[-1].bias[NB * 2:NB * 3] = -2.0
    def forward(self, e, cth, sth):
        if self.drop is not None and self.training: e = self.drop(e)
        o = self.dec(e); pv = o[:, :NB * 2].reshape(-1, NB, 2)
        p = T.atan2(pv[:, :, 0], pv[:, :, 1])
        a = nn.functional.softplus(o[:, NB * 2:NB * 3]); b = o[:, NB * 3]
        C = b[:, None] + (a * T.cos(p)) @ cth + (a * T.sin(p)) @ sth
        return T.clamp(C, min=1e-4) ** 2 + 1e-8

def _fit_loop(model, opt, params, cth, sth, Yt, Wt, ma, wall, hz, steps, emb=None, ema_from=0.5):
    best, stall, state, ema = np.inf, 0, None, None
    for it in range(steps):
        model.train()
        sig = T.sqrt(model(emb if emb is not None else model.emb, cth, sth) + 1e-8)
        per = ((sig[:, :wall] - Yt[:, :wall]).abs() * Wt).sum(1)
        d = (sig[:, wall:hz] - ma) / T.clamp(ma, min=1e-3)
        loss = (per + LAM_H * (d ** 2).mean(1)).sum() / sig.shape[0]
        opt.zero_grad(); loss.backward(); opt.step()
        if emb is None and it > steps * ema_from:
            with T.no_grad():
                cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema = cur if ema is None else {k: 0.999 * ema[k] + 0.001 * cur[k] for k in ema}
        if it % 200 == 199:
            lv = loss.item()
            if lv < best - 1e-7:
                best, stall = lv, 0
                state = [x.detach().clone() for x in params] if emb is not None else \\
                        {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stall += 1
                if stall >= 10: break
    with T.no_grad():
        if emb is not None:
            for x, sv in zip(params, state): x.copy_(sv)
        else:
            model.load_state_dict(ema if ema is not None else state)
    return best

def run_entry(wall, dim=64, depth=2, width=64, dropout=0.15, lr=5e-3, steps=16000, seed=7):
    """Train shared on TRAIN, freeze, infer embeddings for the unseen HELD fields."""
    T.manual_seed(seed); np.random.seed(seed)
    TH = TH_ALL[:, BI]; ne = TH.shape[0]; hz = min(wall + HORIZON, ne)
    tb = lambda a: T.tensor(np.asarray(a, np.float32), device=DEV)
    cth, sth = tb(np.cos(TH).T), tb(np.sin(TH).T)
    P = np.random.RandomState(0).randn(5, dim).astype(np.float32) * 0.3   # fixed → deterministic init
    Ysq, W, m = prep(wall, TRAIN)
    model = Shared(dim, depth, width, len(TRAIN), dropout).to(DEV)
    with T.no_grad(): model.emb.copy_(tb(det_features(TRAIN, wall) @ P))
    opt = T.optim.Adam(model.parameters(), lr=lr)
    _fit_loop(model, opt, list(model.parameters()), cth, sth, tb(Ysq), tb(W), tb(m), wall, hz, steps)
    for prm in model.dec.parameters(): prm.requires_grad_(False)
    model.eval()
    Ysq_h, W_h, m_h = prep(wall, HELD)
    eh = T.tensor(det_features(HELD, wall) @ P, device=DEV, requires_grad=True)
    opt2 = T.optim.Adam([eh], lr=lr)
    _fit_loop(model, opt2, [eh], cth, sth, tb(Ysq_h), tb(W_h), tb(m_h), wall, hz, steps, emb=eh)
    with T.no_grad():
        return np.clip(model(eh, cth, sth).cpu().numpy(), 0, None)
'''

CELL_SWEEP = '''
# ── THE SWEEP: capacity x MANY SEEDS, reported with a standard error ─────────────────────
# A seed is the unit that matters: the local run could not resolve a ~0.013 difference because the
# seed spread was larger than it. Eight seeds per configuration is the point of the GPU.
SEEDS = [7, 11, 23, 3, 42, 101, 202, 303]
GRID = [
    dict(dim=64,  depth=2, width=64),    # the stabilised local best, for reference
    dict(dim=128, depth=2, width=128),
    dict(dim=256, depth=3, width=256),
    dict(dim=512, depth=3, width=512),
    dict(dim=256, depth=4, width=256),
]
rows = []
for cfg in GRID:
    t0 = time.time(); per_seed = []
    for sd in SEEDS:
        per_seed.append(float(np.mean([score(run_entry(w, seed=sd, **cfg), w) for w in WALLS])))
    mu = float(np.mean(per_seed)); se = float(np.std(per_seed, ddof=1) / np.sqrt(len(per_seed)))
    rows.append({**cfg, "auc": round(mu, 4), "se": round(se, 4),
                 "spread": round(max(per_seed) - min(per_seed), 4),
                 "per_seed": [round(v, 4) for v in per_seed], "mins": round((time.time() - t0) / 60, 1)})
    print(f"dim{cfg['dim']:<4d} depth{cfg['depth']} → AUC {mu:+.4f} ± {se:.4f} (SE) · "
          f"spread {rows[-1]['spread']:.4f} · {rows[-1]['mins']:.0f} min", flush=True)

print("\\n── LEAGUE (mean AUC over 5 walls, +/- SE over 8 seeds) ──")
for r in sorted(rows, key=lambda r: -r["auc"]):
    print(f"  {r['auc']:+.4f} ± {r['se']:.4f}   dim {r['dim']:<4d} depth {r['depth']}")
CONTROL = 0.6200   # fit-alone per-field receiver, measured locally (SE 0.0006)
best = max(rows, key=lambda r: r["auc"])
d = best["auc"] - CONTROL; sed = (best["se"] ** 2 + 0.0006 ** 2) ** 0.5
print(f"\\n  best vs fit-alone control ({CONTROL:+.4f}): {d:+.4f} ± {sed:.4f} = {d/sed:+.1f} SE")
print(f"  → {'the embedding model WINS' if d > 2*sed else 'the control holds / indistinguishable'}")
json.dump(rows, open("kaggle_bigmodel_results.json", "w"), indent=1)
print("\\nSWEEPDONE")
'''

nb = {"cells": [], "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3",
                                               "name": "python3"},
                                "language_info": {"name": "python", "version": "3.11"},
                                "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
md = ("# Bigger models on GPU — unseen-field generalisation\\n\\n"
      "Self-contained: the published citations rail (OpenAlex CC0, DOI 10.5281/zenodo.21537062) and the "
      "sidereal ephemeris are embedded, so this runs with **no attached dataset and no internet**.\\n\\n"
      "**Task.** 20% of the 251 research fields are held out *entirely*; a shared decoder is trained on "
      "the rest, frozen, and each unseen field's embedding is inferred from its own history before the "
      "wall. Scored on the 30-year forecast at five origins.\\n\\n"
      "**Why a GPU.** Not size — *variance*. The held-out AUC moved 0.278–0.562 on the training seed "
      "alone; every configuration here is run at 8 seeds and reported with a standard error.")
for src, kind in ((md, "markdown"), (CELL_DATA, "code"), (CELL_HARNESS, "code"),
                  (CELL_MODEL, "code"), (CELL_SWEEP, "code")):
    nb["cells"].append({"cell_type": kind, "metadata": {},
                        **({"outputs": [], "execution_count": None} if kind == "code" else {}),
                        "source": src})

out = os.path.join(HERE, "kaggle_bigmodel.ipynb")
json.dump(nb, open(out, "w"))
print(f"wrote {out} ({os.path.getsize(out)/1024:.0f} KB, {len(nb['cells'])} cells)")
