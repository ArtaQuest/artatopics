#!/bin/bash
# Astro-1200 five-property lunation-chart download loop — round-robins web/images/news/shop/youtube
# (each pass lets every property advance; each downloader is itself resumable). No deploys.
set -u
cd /Users/arash/Studio/artaquest
count() { ls analysis/astro1200/${1}_lunar_*.csv 2>/dev/null | wc -l | tr -d ' '; }
while true; do
  T=0
  for P in web images news shop youtube; do
    python3 analysis/astro1200/topcharts.py "$P" >> /tmp/astro1200_charts.out 2>&1
    C=$(count "$P"); T=$((T+C))
    echo "[charts $(date +%H:%M)] $P=$C"
  done
  echo "[charts $(date +%H:%M)] TOTAL=$T/1110"
  [ "$T" -ge 1110 ] && { echo "[charts] ALL FIVE PROPERTIES COMPLETE"; break; }
  sleep 5
done
