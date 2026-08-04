#!/bin/bash
set -u
cd /Users/arash/Studio/artaquest
remaining() { python3 -c "
import json,os
import importlib.util as u
s=u.spec_from_file_location('tf','analysis/trends_fit.py'); tf=u.module_from_spec(s); s.loader.exec_module(tf)
v=json.load(open('analysis/adstopics/vocabulary.json'))
print(sum(1 for t in v if not os.path.exists(f'analysis/adstopics/series/{tf.slug(t)}.csv')))"; }
while true; do
  R=$(remaining)
  echo "[adstopics $(date +%H:%M)] remaining=$R"
  [ "$R" -eq 0 ] && { echo "[adstopics] ALL SERIES FETCHED"; break; }
  python3 analysis/adstopics/fetch_series.py --limit 120 >> /tmp/adstopics_fetch.out 2>&1
  sleep 2
done
