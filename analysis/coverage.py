#!/usr/bin/env python3
"""Paper gate for the ISCO-08 occupations atlas: has EVERY one of the 436 occupations been analysed?
Exit 0 when every field in _fields.json has been processed (fit or skipped), so the paper can be written
and submitted to the Journal of Seasonality. No selection, no noun/adj — all 436 are the final data.

  python3 analysis/coverage.py        # exit 0 when all processed, else 1
"""
import json, os, sys

fields = json.load(open("analysis/_fields.json"))
reg = json.load(open("analysis/_fields_weekly.json")) if os.path.exists("analysis/_fields_weekly.json") else {}

processed = sum(1 for k in fields if k in reg)
fit = sum(1 for v in reg.values() if v.get("res") == "weekly")
skip = sum(1 for v in reg.values() if v.get("res") == "skip")
done = processed >= len(fields)

print(f"\nISCO-08 occupations analysis: {processed}/{len(fields)} processed  ({fit} fitted, {skip} skipped)")
print("READY: all occupations analysed — write + submit the paper\n" if done else "not yet — collection continuing\n")
sys.exit(0 if done else 1)
