#!/usr/bin/env python3
"""LLM EMBEDDINGS OF THE FIELD NAMES (operator 2026-07-26: "try embedding with SoTA LLMs").

The idea is stronger than a better initialisation. Until now an unseen field's embedding had to be
INFERRED from its own citation history; a language-model embedding of its NAME is available before the
field has any history at all. If that works, the model can forecast a research field it has never seen
from nothing but the words "Artificial Intelligence" and the positions of the planets.

    text_j = "<subfield>. A research subfield of <field>, in <domain>."
    e_j    = LLM(text_j)                       → 251 × d, computed ONCE, offline thereafter

Written to llm_embeddings.npz so every downstream run (local and the Kaggle GPU kernel) reads a fixed
matrix — no network at fit time, and the published notebook stays reproducible with the file alone.

MODEL CHOICE: an open embedding model, run locally, no API key and no provider account — the project's
vault has no embedding-provider key, and a checked-in matrix is more reproducible than a paid endpoint
that may change under us. Candidates are tried in order of quality-for-size and the first that loads
wins; the one actually used is recorded in the npz, because "which model" is part of the result.

  python3 analysis/arxivtopics/llm_embed.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "llm_embeddings.npz")

CANDIDATES = [
    "Qwen/Qwen3-Embedding-0.6B",     # 2025 SoTA for its size on MTEB
    "BAAI/bge-large-en-v1.5",        # strong, well-established
    "thenlper/gte-large",            # compact fallback
    "sentence-transformers/all-MiniLM-L6-v2",   # last resort, tiny
]


def build_texts():
    d = pd.read_csv(os.path.join(REPO, "analysis/citations/rail_citations_received_yearly.csv"))
    txt = [f"{r.subfield}. A research subfield of {r.field}, in {r.domain}."
           for _, r in d.iterrows()]
    return list(d.subfield), txt


def embed(texts):
    import torch as T
    from transformers import AutoTokenizer, AutoModel
    dev = "mps" if T.backends.mps.is_available() else "cpu"
    last_err = None
    for name in CANDIDATES:
        try:
            print(f"  loading {name} …", flush=True)
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name).to(dev).eval()
        except Exception as e:
            print(f"    unavailable ({type(e).__name__}: {str(e)[:90]})", flush=True)
            last_err = e
            continue
        vecs = []
        with T.no_grad():
            for i in range(0, len(texts), 16):
                b = tok(texts[i:i + 16], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
                out = mdl(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                if "Qwen" in name:                      # last-token pooling, per the model card
                    idx = b["attention_mask"].sum(1) - 1
                    v = out[T.arange(out.shape[0]), idx]
                else:                                   # mean pooling
                    v = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
                vecs.append(T.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(vecs, 0).astype(np.float32), name
    raise RuntimeError(f"no embedding model could be loaded: {last_err}")


if __name__ == "__main__":
    names, texts = build_texts()
    print(f"═══ embedding {len(texts)} field names with a SoTA open model ═══", flush=True)
    print(f"  e.g. {texts[0]!r}", flush=True)
    E, model = embed(texts)
    E = (E - E.mean(0)) / np.maximum(E.std(0), 1e-6)          # standardise per dimension
    np.savez_compressed(OUT, E=E, names=np.array(names), model=model)
    print(f"  model used: {model} · matrix {E.shape} → {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)", flush=True)

    # sanity: does the geometry actually encode the taxonomy? nearest neighbours should be sensible.
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    S = En @ En.T; np.fill_diagonal(S, -9)
    print("\n  nearest neighbour by name-embedding (a sanity check on the geometry):", flush=True)
    for i in [names.index(x) for x in names[:1]] + list(np.random.RandomState(0).choice(len(names), 5, replace=False)):
        print(f"    {names[i][:38]:40s} → {names[int(S[i].argmax())][:38]}", flush=True)
    d = pd.read_csv(os.path.join(REPO, "analysis/citations/rail_citations_received_yearly.csv"))
    same = float(np.mean([d.field.iloc[i] == d.field.iloc[int(S[i].argmax())] for i in range(len(names))]))
    print(f"  nearest neighbour shares the parent field for {same*100:.0f}% of subfields "
          f"(chance ≈ {100/d.field.nunique():.0f}%)", flush=True)
    print("EMBEDDONE", flush=True)
