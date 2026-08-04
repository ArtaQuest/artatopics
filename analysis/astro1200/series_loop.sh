#!/bin/bash
# Astro four-property series fetch loop — round-robins the study properties until every selected
# term's monthly series (own property) is cached. Resumable. No deploys.
set -u
cd /Users/arash/Studio/artaquest
remaining() { python3 - <<'PY'
import json, os
import importlib.util as u
s=u.spec_from_file_location("tf","analysis/trends_fit.py"); tf=u.module_from_spec(s); s.loader.exec_module(tf)
tot=0
for p in ("images","news","shop","youtube"):
    d=json.load(open(f"analysis/astro1200/{p}_chart_terms.json"))
    tot+=sum(1 for t,r in d.items() if r.get("selected") and not os.path.exists(f"analysis/astro1200/series_{p}/{tf.slug(t)}.csv"))
print(tot)
PY
}
while true; do
  R=$(remaining)
  echo "[series $(date +%H:%M)] remaining=$R"
  [ "$R" -eq 0 ] && { echo "[series] ALL SERIES FETCHED"; break; }
  for P in images news shop youtube; do
    python3 analysis/astro1200/fetch_series.py "$P" --limit 40 >> /tmp/astro1200_series.out 2>&1
  done
  sleep 2
done
