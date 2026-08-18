#!/usr/bin/env python3
"""Deploy docs/ to the Hugging Face Space artaquest/artatopics (static SDK).

The Space is a MIRROR of the GitHub Pages site — same generated files, no hand-edits — so the
publish order is always: build_docs.py + build_ensemble_page.py → git push (Pages) → this script.
Static SDK is deliberate: the pages carry their own Pyodide lab, so the reader's own browser does
the verification. A server-side Gradio app would move that trust back onto us, and a free Gradio
Space sleeps after inactivity while a static one never does.

Token: $HF_TOKEN, else ~/.artaquest-dev/hf_token_pro. NEVER hardcode or commit it.

  python3 analysis/arxivtopics/competition/deploy_hf_space.py [--dry-run]
"""
import os, sys, pathlib
from huggingface_hub import HfApi

REPO_ID = "artaquest/artatopics"
HERE = pathlib.Path(__file__).resolve().parent
DOCS = HERE.parents[2] / "docs"
assert (DOCS / "index.html").is_file(), f"docs/ not found at {DOCS}"

tok = os.environ.get("HF_TOKEN")
if not tok:
    p = pathlib.Path.home() / ".artaquest-dev/hf_token_pro"
    tok = p.read_text().strip().strip('"') if p.is_file() else None
assert tok, "no HF token: set $HF_TOKEN or ~/.artaquest-dev/hf_token_pro"

README = """---
title: ArtaTopics
emoji: 🃏
colorFrom: yellow
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Can the sky forecast what science studies?
---

# ArtaTopics — planetary positions vs. the citation record

Can the sky forecast what science studies? This Space publishes the whole campaign, including
the parts that did not work.

- **The models** are fitted on 1700–1995 and graded on 1996–2025 they never saw, against
  do-nothing baselines. Nothing is graded on a window it was selected in.
- **The competition** ([`astro-ensemble-251`](https://artaquest.com/competitions/astro-ensemble-251))
  pitted four GPU model families — deep phasor, shared-basis receiver, random sky-feature swarm,
  residual boosting — against each other and against two baselines. Dataset:
  [astro-ensemble-251 on Kaggle](https://www.kaggle.com/datasets/artafather/astro-ensemble-251).
- **The honest headline:** every single-family GPU model loses to "carry today forward", and the
  best selected stack (−2.09) still trails a plain damped linear trend (−2.04). The sky's measured
  contribution is small, and what there is behaves like a slow calendar rather than a rhythm.
- **Verify it yourself:** the ensemble page loads Pyodide + numpy in *your* browser, rebuilds the
  deployed model from the raw citation shares, and re-derives its score from the held-out truth.
  Nothing on these pages asks to be taken on faith.

No causal claims are made anywhere in this work.

Mirror of <https://artaquest.github.io/artatopics/> · source and full history:
<https://github.com/ArtaQuest/artatopics> · data CC0 (OpenAlex), code MIT.
"""

api = HfApi(token=tok)
files = sorted(p for p in DOCS.rglob("*") if p.is_file())
total = sum(p.stat().st_size for p in files)
print(f"{REPO_ID}: {len(files)} files, {total // 1024}KB")
for p in files: print(f"   {p.relative_to(DOCS)}")
if "--dry-run" in sys.argv:
    print("dry run — nothing uploaded"); sys.exit(0)

api.create_repo(REPO_ID, repo_type="space", space_sdk="static", exist_ok=True, private=False)
(DOCS / "README.md").write_text(README)
try:
    api.upload_folder(repo_id=REPO_ID, repo_type="space", folder_path=str(DOCS),
                      commit_message="Publish the artatopics results + the ensemble stack page")
finally:
    (DOCS / "README.md").unlink(missing_ok=True)   # README is Space metadata, not a Pages artifact
print(f"live: https://huggingface.co/spaces/{REPO_ID}")
