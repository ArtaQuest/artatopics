#!/bin/bash
# Astro-1200 daily-chart download loop — restarts the (resumable) downloader until every day
# 2008-01-01..2025-12-31 has its CSV. No deploys.
set -u
cd /Users/arash/Studio/artaquest
expected() { python3 -c "import datetime as dt;print((dt.date(2025,12,31)-dt.date(2008,1,1)).days+1)"; }
have() { ls analysis/astro1200/news_daily_*.csv 2>/dev/null | wc -l | tr -d ' '; }
E=$(expected)
while true; do
  H=$(have)
  echo "[astro1200-daily $(date +%H:%M)] $H/$E days"
  [ "$H" -ge "$E" ] && { echo "[astro1200-daily] DOWNLOAD COMPLETE"; break; }
  python3 analysis/astro1200/news_topcharts.py >> /tmp/astro1200_daily.out 2>&1
  sleep 5
done
