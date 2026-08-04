#!/bin/bash
# TOPICS track collector (monthly model) WITH deploy-every-batch. Collects the science topics in _topics.json,
# and after EACH batch rebuilds the combined atlas + research.json, commits, and pushes to prod through the lock
# (operator: "add to topics page after each 10 new and push to prod"). Checkpointed/resumable.
set -u
cd /Users/arash/Studio/artaquest
export AQ_POOL=analysis/_topics.json
export AQ_REG=analysis/_topics_weekly.json
export AQ_MDIR=analysis/data_monthly          # shared monthly cache so the build's load_y finds topic series too
BATCH=100
remaining() { python3 -c "import json,os;w=json.load(open('$AQ_POOL'));d=json.load(open('$AQ_REG')) if os.path.exists('$AQ_REG') else {};print(sum(1 for k in w if k not in d))"; }
fitted()    { python3 -c "import json,os;d=json.load(open('$AQ_REG')) if os.path.exists('$AQ_REG') else {};print(sum(1 for v in d.values() if v.get('res')=='weekly'))"; }
while true; do
  echo "[sci $(date +%H:%M)] collecting batch of $BATCH ..."
  python3 analysis/collect_weekly.py --limit "$BATCH" >> /tmp/collect_sci.out 2>&1
  # merge skills + topics (axis-tagged, collisions suffixed so pages stay separated) → combined atlas
  AQ_SKILLS=analysis/_fields_weekly.json AQ_TOPICS="$AQ_REG" AQ_OUT=analysis/_allfit.json python3 analysis/merge_atlas.py
  AQ_REG=analysis/_allfit.json AQ_DATA=analysis/data_monthly python3 analysis/build_disciplines.py >> /tmp/build_sci.out 2>&1
  AQ_REG=analysis/_allfit.json python3 analysis/export_research.py >> /tmp/build_sci.out 2>&1
  echo "[sci $(date +%H:%M)] fitted=$(fitted) remaining=$(remaining) — committing + deploying"
  PATHS="artaquest-web/src/data/research.json artaquest-web/src/data/field-daily wp-content/plugins/aquest/data/aq-disciplines-add.json wp-content/plugins/aquest/data/aq-disc-trends.json analysis/_topics_weekly.json analysis/_topics.json analysis/_fields_weekly.json"
  git add $PATHS 2>/dev/null
  git commit -q -m "Science topics: +$(fitted) on the Topics page (monthly atlas)" -- $PATHS 2>/dev/null
  # deploy themes (SPA incl research.json) THEN plugin data — through the lock; skips/retries if ArtaDev holds it
  { bash tools/isolated-deploy.sh && tools/ticket-agent/aq-deploy studio push --path . --options plugins --remote-site https://artaquest.org; } >> /tmp/deploy_sci.out 2>&1 &
  [ "$(remaining)" -eq 0 ] && { echo "[sci] ALL science topics collected."; break; }
  sleep 5
done
