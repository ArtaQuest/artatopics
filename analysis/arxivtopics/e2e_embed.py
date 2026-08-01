#!/usr/bin/env python3
"""END-TO-END EMBEDDING MODEL (operator 2026-07-26): ONE model for all 251 fields.

    topic id ──▶ EMBEDDING e_j ──▶ decoder ──▶ 7 phases p_jᵢ (+ 7 amplitudes, 1 anchor)
                                                   │
                        that year's sky θᵢ(t) ─────┴──▶ rotate: φᵢ = θᵢ(t) − p_jᵢ
                                                          │
                                                          ▼
                                             astro predictor ──▶ ŷ_j(t)  (share that year)

Everything is trained jointly by backprop. This differs from the deployed v10 in ONE structural way:
v10 gives every field its own free (a, p, b); here a SHARED decoder generates them from a learned
per-field embedding, so fields are forced through a common bottleneck and share structure. With a wide
embedding and a flexible decoder it can represent v10 exactly; with a narrow one it is a low-rank,
amortised version that may generalise better over a 30-year extrapolation — or may lose the per-field
freedom that made v10 work. That is the question.

ADVERSARIAL DESIGN SEARCH: every embedding/decoder/training choice is attacked one axis at a time by
greedy coordinate ascent on WALL_INNER, with diagnostics that catch the two ways this architecture
fails quietly — the embedding collapsing (all fields decoded to the same phases) and the decoder
ignoring the embedding (phases constant, model reduced to one global receiver).

DISCIPLINE: every choice on WALL_INNER (fit ≤1965, judged 1966-95). Finalists fitted at WALL_OUTER
once, 3 seeds. CONSTRAINT: at prediction time the model sees ONLY the sky — the embedding is a learned
parameter, the citation record enters solely through the training loss/weights/anchor.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 analysis/arxivtopics/e2e_embed.py
"""
import os, sys, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comp_harness import *
import torch as T
import torch.nn as nn

DEV = "mps" if T.backends.mps.is_available() else "cpu"
BODS = CHAMPION_BODIES
NB = len(BODS)
BI = [BODIES_ALL.index(b) for b in BODS]

DEFAULT = dict(dim=16, decoder="linear", width=64, depth=1, head="physical", emb_init="level",
               emb_norm="none", dropout=0.0, wd=0.0, lr=2e-2, steps=9000, kpow=2.0,
               wexp=0.75, lam_h=0.03, anchor_k=5, hidden=32)


class E2E(nn.Module):
    """ONE model: embedding → decoder → (phases, amplitudes, anchor) → astro predictor → share."""

    def __init__(self, cfg, vmean):
        super().__init__()
        d = cfg["dim"]
        self.cfg = cfg
        E0 = np.random.randn(Tn, d).astype(np.float32) * 0.1
        if cfg["emb_init"] == "level":                    # seed one coordinate with each field's level
            E0[:, 0] = np.log(np.expm1(np.clip(vmean, 1e-3, None)))
        self.emb = nn.Parameter(T.tensor(E0))
        self.norm = nn.LayerNorm(d) if cfg["emb_norm"] == "layernorm" else None
        self.drop = nn.Dropout(cfg["dropout"]) if cfg["dropout"] > 0 else None
        out = NB * 2 + NB + 1                             # phase 2-vectors, amplitudes, anchor
        if cfg["decoder"] == "linear":
            self.dec = nn.Linear(d, out)
        else:
            layers, i = [], d
            for _ in range(cfg["depth"]):
                layers += [nn.Linear(i, cfg["width"]), nn.SiLU()]; i = cfg["width"]
            layers += [nn.Linear(i, out)]
            self.dec = nn.Sequential(*layers)
        with T.no_grad():                                 # start near the deployed model's init
            last = self.dec if isinstance(self.dec, nn.Linear) else self.dec[-1]
            last.weight.mul_(0.05)
            last.bias.zero_()
            last.bias[NB * 2:NB * 3] = -2.0               # softplus(-2) ≈ small amplitudes
            last.bias[NB * 2 + BODS.index("pluto")] = 0.0
        if cfg["head"] == "mlp":                          # learned astro predictor on the rotated sky
            self.mlp = nn.Sequential(nn.Linear(2 * NB + 1, cfg["hidden"]), nn.SiLU(),
                                     nn.Linear(cfg["hidden"], 1))

    def params_of(self):
        e = self.emb
        if self.norm is not None: e = self.norm(e)
        if self.drop is not None: e = self.drop(e)
        o = self.dec(e)
        pv = o[:, :NB * 2].reshape(Tn, NB, 2)
        p = T.atan2(pv[:, :, 0], pv[:, :, 1])             # 7 phases per field
        a = nn.functional.softplus(o[:, NB * 2:NB * 3])   # 7 amplitudes
        b = o[:, NB * 3]                                  # anchor
        return p, a, b

    def forward(self, cth, sth):
        """cth/sth: (NB, ne) cos/sin of that year's sky. Returns (Tn, ne) predicted share."""
        p, a, b = self.params_of()
        if self.cfg["head"] == "physical":
            C = b[:, None] + (a * T.cos(p)) @ cth + (a * T.sin(p)) @ sth
            return T.clamp(C, min=1e-4) ** self.cfg["kpow"] + 1e-8
        # MLP head: feed the ROTATED sky (cos/sin of θ−p, scaled by the amplitudes) per field-year
        cp, sp = T.cos(p), T.sin(p)
        cphi = cp[:, :, None] * cth[None] + sp[:, :, None] * sth[None]      # (Tn,NB,ne) cos(θ−p)
        sphi = cp[:, :, None] * sth[None] - sp[:, :, None] * cth[None]      # (Tn,NB,ne) sin(θ−p)
        z = T.cat([cphi * a[:, :, None], sphi * a[:, :, None],
                   b[:, None, None].expand(-1, 1, cth.shape[1])], 1)        # (Tn, 2NB+1, ne)
        out = self.mlp(z.permute(0, 2, 1)).squeeze(-1)                       # (Tn, ne)
        return T.clamp(out, min=1e-4) ** self.cfg["kpow"] + 1e-8


def fit(wall, cfg=None, seed=7, diag=False):
    cfg = {**DEFAULT, **(cfg or {})}
    T.manual_seed(seed); np.random.seed(seed)
    TH = TH_ALL[:, BI]; ne = TH.shape[0]
    Ysq = np.sqrt(Y)
    tb = lambda a: T.tensor(np.asarray(a, np.float32), device=DEV)
    cth, sth = tb(np.cos(TH).T), tb(np.sin(TH).T)
    tv = train_mask(wall).astype(np.float32)
    wy = np.clip(N[:wall], 0, None) ** cfg["wexp"]
    W = tv * wy[None]; W = W / np.maximum(W.sum(1, keepdims=True), 1e-9); Wt = tb(W)
    Wa = np.zeros_like(W); Wa[:, wall - cfg["anchor_k"]:] = (tv * wy[None])[:, wall - cfg["anchor_k"]:]
    bad = Wa.sum(1) <= 0; Wa[bad] = (tv * wy[None])[bad]
    Wa = Wa / np.maximum(Wa.sum(1, keepdims=True), 1e-9)
    m_anchor = tb(((Ysq[:, :wall] * Wa).sum(1))[:, None])
    vmean = (Ysq[:, :wall] * tv).sum(1) / np.maximum(tv.sum(1), 1.0)
    model = E2E(cfg, vmean).to(DEV)
    Yt = tb(Ysq)
    opt = T.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    hz = min(wall + HORIZON, ne)
    best, stall, state = np.inf, 0, None
    for it in range(cfg["steps"]):
        model.train()
        sig = T.sqrt(model(cth, sth) + 1e-8)
        per = ((sig[:, :wall] - Yt[:, :wall]).abs() * Wt).sum(1)
        d = (sig[:, wall:hz] - m_anchor) / T.clamp(m_anchor, min=1e-3)
        loss = (per + cfg["lam_h"] * (d ** 2).mean(1)).sum() / Tn
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 199:
            lv = loss.item()
            if lv < best - 1e-7: best, stall, state = lv, 0, copy.deepcopy(model.state_dict())
            else:
                stall += 1
                if stall >= 10: break
    model.load_state_dict(state); model.eval()
    with T.no_grad():
        yh = np.clip(model(cth, sth).cpu().numpy(), 0, None)
        if diag:
            p, a, b = model.params_of()
            pd = np.rad2deg(p.cpu().numpy()) % 360.0
            emb = model.emb.detach().cpu().numpy()
            sv = np.linalg.svd(emb - emb.mean(0), compute_uv=False)
            return yh, {"phase_spread_deg": float(np.mean(np.std(pd, 0))),
                        "emb_eff_rank": float((sv.sum() ** 2) / max((sv ** 2).sum(), 1e-12)),
                        "amp_spread": float(np.std(a.cpu().numpy()))}
    return yh


def inner(cfg=None, seed=7):
    return evaluate(fit(WALL_INNER, cfg, seed), WALL_INNER)["auc"]


if __name__ == "__main__":
    print("═══ BASELINES ═══", flush=True)
    print(f"  deployed v10 (free per-field params)     inner +0.8929 · outer +0.8174", flush=True)
    a0 = inner()
    yh, dg = fit(WALL_INNER, diag=True)
    print(f"  E2E default (dim16 linear physical)      inner {a0:+.4f} · {dg}", flush=True)

    # ── ADVERSARIAL COORDINATE ASCENT on the inner wall ──────────────────────────────────────
    AXES = [
        ("dim",       [4, 8, 32, 64, 128]),
        ("emb_init",  ["random"]),
        ("emb_norm",  ["layernorm"]),
        ("decoder",   ["mlp"]),
        ("head",      ["mlp"]),
        ("wd",        [1e-5, 1e-4, 1e-3]),
        ("dropout",   [0.05, 0.15]),
        ("lr",        [5e-3, 1e-2, 5e-2]),
        ("steps",     [16000]),
    ]
    cur, best = dict(DEFAULT), a0
    print("\n═══ ADVERSARIAL SEARCH — each design choice attacked on WALL_INNER ═══", flush=True)
    for axis, alts in AXES:
        won = None
        for v in alts:
            cfg = {**cur, axis: v}
            if axis == "decoder" and v == "mlp": cfg["depth"] = 2
            try: a = inner(cfg)
            except Exception as e:
                print(f"   {axis}={v}: FAILED {type(e).__name__}", flush=True); continue
            flag = ""
            if a > best + 1e-4: won, best, flag = v, a, "  ← kept"
            print(f"   {axis:9s}= {str(v):10s} inner {a:+.4f}{flag}", flush=True)
        if won is not None:
            cur[axis] = won
            if axis == "decoder" and won == "mlp": cur["depth"] = 2
    print(f"\n  BEST E2E CONFIG (inner {best:+.4f}): "
          f"{ {k: cur[k] for k in ('dim','decoder','depth','head','emb_init','emb_norm','dropout','wd','lr','steps')} }", flush=True)
    yh, dg = fit(WALL_INNER, cur, diag=True)
    print(f"  diagnostics: {dg}", flush=True)
    if dg["phase_spread_deg"] < 5:
        print("  ⚠ EMBEDDING COLLAPSE: fields decode to nearly the same phases", flush=True)

    print("\n═══ OUTER WALL — best E2E vs the deployed model, 3 seeds, fitted once ═══", flush=True)
    e_out = [evaluate(fit(WALL_OUTER, cur, seed=s), WALL_OUTER) for s in (7, 11, 23)]
    med = float(np.median([r["auc"] for r in e_out]))
    print(f"  E2E embedding model   OUTER AUC {med:+.4f} "
          f"[{min(r['auc'] for r in e_out):+.4f}..{max(r['auc'] for r in e_out):+.4f}] · "
          f"skill {np.median([r['skill'] for r in e_out]):+.4f} · "
          f"{np.median([r['pct'] for r in e_out]):.1f}%>0", flush=True)
    print(f"  deployed v10          OUTER AUC +0.8174 (3-seed median)", flush=True)
    print(f"  → {'E2E WINS' if med > 0.8174 else 'deployed v10 holds'}", flush=True)
    json.dump({"best_cfg": {k: str(v) for k, v in cur.items()}, "inner": best, "outer_med": med,
               "diag": dg}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "e2e_embed_result.json"), "w"), indent=1)
    print("E2EDONE", flush=True)
