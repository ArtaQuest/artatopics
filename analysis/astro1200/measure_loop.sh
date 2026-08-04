#!/bin/bash
# Astro-1200 popularity measurement loop — resumable, no deploys.
set -u
cd /Users/arash/Studio/artaquest
remaining() { python3 -c "import json,os;c=json.load(open('analysis/astro1200/candidates.json'));d=json.load(open('analysis/astro1200/popularity.json')) if os.path.exists('analysis/astro1200/popularity.json') else {};print(sum(1 for t in c if t not in d))"; }
while true; do
  R=$(remaining)
  echo "[astro1200-pop $(date +%H:%M)] remaining=$R"
  [ "$R" -eq 0 ] && { echo "[astro1200-pop] MEASUREMENT COMPLETE"; break; }
  python3 analysis/astro1200/measure_pop.py --limit 120 >> /tmp/astro1200_pop.out 2>&1
  sleep 2
done
