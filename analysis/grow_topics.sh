#!/bin/bash
# Topic-centric field growth — the ONLY collection loop (the multi-level funnel + word pools were scrapped).
#   topics → 4 nouns + 4 adjectives (select_fields) → most-common keywords → DIRECT daily collect+fit → publish
# Walks analysis/_fields.json most-common-first, collecting a batch of real daily Trends fits each pass, then —
# once enough fields are in — regenerates the atlas + plugin data and deploys (themes via the isolated worktree,
# plugin data under the shared lock). Checkpointed/resumable; safe to kill and restart.
#   bash analysis/grow_topics.sh        (runs until every selected field is collected)
set -u
cd /Users/arash/Studio/artaquest
MIN_PUBLISH=1         # update prod after EVERY batch as fields accrue (operator: "update prod after each update")
BATCH=5               # collect 5 datasets, then regenerate + push to prod (operator: update prod after every 5 new analyses)
fitted() { python3 -c "import json,os;d=json.load(open('analysis/_fields_weekly.json')) if os.path.exists('analysis/_fields_weekly.json') else {};print(sum(1 for v in d.values() if v.get('res')=='weekly'))"; }
remaining() { python3 -c "import json,os;w=json.load(open('analysis/_fields.json'));d=json.load(open('analysis/_fields_weekly.json')) if os.path.exists('analysis/_fields_weekly.json') else {};print(sum(1 for k in w if k not in d))"; }

while true; do
  python3 analysis/drain_queue.py >> /tmp/collect_topics.out 2>&1   # apply operator Studio "Houses" add/remove requests
  echo "[grow_topics $(date +%H:%M)] collecting batch of $BATCH (most-common first)…"
  python3 analysis/collect_weekly.py --limit "$BATCH" >> /tmp/collect_topics.out 2>&1
  F=$(fitted); R=$(remaining)
  echo "[grow_topics $(date +%H:%M)] fitted=$F  remaining=$R"
  if [ "$F" -ge "$MIN_PUBLISH" ]; then
    python3 analysis/export_research.py >> /tmp/export_topics.out 2>&1
    python3 analysis/build_disciplines.py >> /tmp/export_topics.out 2>&1
    PATHS="artaquest-web/src/data/research.json artaquest-web/src/data/field-daily wp-content/plugins/aquest/data/aq-disciplines-add.json wp-content/plugins/aquest/data/aq-disc-trends.json analysis/_fields_weekly.json"
    git add $PATHS 2>/dev/null
    git commit -q -m "Topic atlas: $F fields collected (topic-centric, by commonness)" -- $PATHS 2>/dev/null
    # themes (isolated worktree build) THEN plugin data — SEQUENTIAL (two concurrent studio pushes race WP.com's
    # server-side sync lock); the pair is backgrounded so the collect loop keeps going, but themes finishes before plugins.
    { bash tools/isolated-deploy.sh && tools/ticket-agent/aq-deploy studio push --path . --options plugins --remote-site https://artaquest.org; } >> /tmp/td_topics.out 2>&1 &
  fi
  [ "$R" -eq 0 ] && { echo "[grow_topics] ALL selected fields collected."; break; }
  sleep 5
done
