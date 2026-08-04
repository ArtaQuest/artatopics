#!/usr/bin/env python3
"""adstopics — reproduce everything from the raw series cache + taxonomy snapshot.

  python3 analysis/adstopics/reproduce.py            # atlas only (the paper's map)
  python3 analysis/adstopics/reproduce.py --full     # the whole 71-arm tournament, then the atlas

Protocol switches (env): AQ_ATLAS_KERNEL=sinc|cos|gauss|vonmises, AQ_ATLAS_FIXEDF=<f|unset>,
AQ_ATLAS_INTERCEPT=0|1, AQ_ATLAS_CLIP=0|1. Defaults reproduce the published atlas exactly.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
LADDER = ["experiments.py", "mechanistic.py", "link_experiments.py", "combo_experiments.py",
          "spec_experiments.py", "compare_f.py", "links_prune.py", "model_v3.py"]

def run(script, *args, env=None):
    e = dict(os.environ, **(env or {}))
    print(f"== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, script), *args], cwd=REPO, env=e, check=True)

if __name__ == "__main__":
    if "--full" in sys.argv:
        for sc in LADDER:
            run(sc, "400")
        run("compare_f.py", "400", env={"AQ_LINKS13": "1"})
    run("direction.py", "selftest")
    run("direction.py", "headline", "all")
    run("atlas_run.py")
