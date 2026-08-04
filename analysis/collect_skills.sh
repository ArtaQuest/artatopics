#!/bin/bash
# SKILLS track collector (monthly FFT model) WITH deploy-every-batch. Collects the 311 skill fields in _skills.json
# (each derived from one of the 436 ISCO occupations) into the axis=house registry _fields_weekly.json, and after
# EACH batch rebuilds the combined atlas + research.json (skills + whatever topics exist), commits, and pushes to
# prod through the lock. Checkpointed/resumable. (Replaces the old professions collection — ticket: careers→skills.)
set -u
cd /Users/arash/Studio/artaquest
export AQ_POOL=analysis/_skills.json
export AQ_REG=analysis/_fields_weekly.json          # the axis=house registry (now SKILLS, was professions)
export AQ_MDIR=analysis/data_monthly                # shared monthly cache
BATCH=100
remaining() { python3 -c "import json,os;w=json.load(open('$AQ_POOL'));d=json.load(open('$AQ_REG')) if os.path.exists('$AQ_REG') else {};print(sum(1 for k in w if k not in d))"; }
fitted()    { python3 -c "import json,os;d=json.load(open('$AQ_REG')) if os.path.exists('$AQ_REG') else {};print(sum(1 for v in d.values() if v.get('res')=='weekly'))"; }
while true; do
  echo "[skills $(date +%H:%M)] collecting batch of $BATCH ..."
  python3 analysis/collect_weekly.py --limit "$BATCH" >> /tmp/collect_skills.out 2>&1
  # merge skills + topics (axis-tagged, collisions suffixed so pages stay separated) → combined atlas
  AQ_SKILLS=analysis/_fields_weekly.json AQ_TOPICS=analysis/_topics_weekly.json AQ_OUT=analysis/_allfit.json python3 analysis/merge_atlas.py
  AQ_REG=analysis/_allfit.json AQ_DATA=analysis/data_monthly python3 analysis/build_disciplines.py >> /tmp/build_skills.out 2>&1
  AQ_REG=analysis/_allfit.json python3 analysis/export_research.py >> /tmp/build_skills.out 2>&1
  echo "[skills $(date +%H:%M)] fitted=$(fitted) remaining=$(remaining) — committing + deploying"
  PATHS="artaquest-web/src/data/research.json artaquest-web/src/data/field-daily wp-content/plugins/aquest/data/aq-disciplines-add.json wp-content/plugins/aquest/data/aq-disc-trends.json analysis/_fields_weekly.json analysis/_skills.json analysis/_isco_to_skill.json"
  git add $PATHS 2>/dev/null
  git commit -q -m "Skills: +$(fitted) on the Skills page (occupation→skill, monthly FFT atlas)" -- $PATHS 2>/dev/null
  # deploy theme (SPA incl research.json + chunks) THEN plugin data — through the lock; skips/retries if ArtaDev holds it
  { bash tools/isolated-deploy.sh && tools/ticket-agent/aq-deploy studio push --path . --options plugins --remote-site https://artaquest.org; } >> /tmp/deploy_skills.out 2>&1 &
  [ "$(remaining)" -eq 0 ] && { echo "[skills] ALL skills collected."; break; }
  sleep 5
done
