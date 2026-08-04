#!/usr/bin/env python3
"""Canonical YEARLY citations matrix (operator 2026-07-24) — the published yearly rail's content
(GET /aq/v1/citations/yearly = the quarterly rail summed to years) cropped to the emerging-topic-safe
core: the (subset, start-year) that MAXIMISES usable non-zero data (topics × years) with NO zeros.

SOURCE: quarterly rail summed to years (year-only-dated citations fold in exactly at yearly grain).
PER-TOPIC CROP (operator 2026-07-24): independent (non-cross-sectional) models keep ALL 251 topics;
each is fit only over its own CONSISTENTLY-NON-ZERO suffix (leading pre-existence years masked out at
fit time), so an emerging field is fit from when it emerged — no shared start year, no dropped topics.

  python3 analysis/citations/build_yearly.py [raw-quarterly-csv]
"""
import sys
import numpy as np, pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else \
    "/private/tmp/claude-501/-Users-arash-Studio-artaquest/8c3c064a-abad-4f0f-8dd6-5a1588c5e7b6/scratchpad/citations-quarterly-new.csv"
OUT = "analysis/citations/citations_received_yearly.csv"

raw = pd.read_csv(RAW, low_memory=False)
raw = raw[pd.to_numeric(raw.subfield_id, errors="coerce").notna()].copy()
raw["subfield_id"] = raw.subfield_id.astype(float).astype(int)
# same 251-subfield modern universe as the quarterly matrix
uni = pd.read_csv("analysis/citations/citations_received_quarterly.csv").subfield_id.tolist()

LO, HI = 1700, 2025                                    # 2026 is a partial year at the snapshot
r = raw[raw.subfield_id.isin(uni) & raw.year.between(LO, HI)]
years0 = [str(y) for y in range(LO, HI + 1)]
# YEARLY = sum over ALL quarters (incl. quarter 0, the year-only bucket — exact at yearly grain)
piv = r.groupby(["subfield_id", "year"]).citations_received.sum().unstack(fill_value=0) \
       .reindex(columns=[int(y) for y in years0], fill_value=0).reindex(index=uni, fill_value=0)
piv.columns = years0
# FULL 251-topic yearly matrix, ALL years (operator 2026-07-24: independent per-topic models keep
# every topic; the "exclude the start until consistently non-zero" crop is PER-TOPIC and applied at
# fit time as a mask — each topic fit only over its own non-zero suffix, so no global crop here).
meta = raw.drop_duplicates("subfield_id").set_index("subfield_id")[["subfield", "field", "domain"]]
out = meta.loc[uni].reset_index().join(piv.loc[uni, years0].reset_index(drop=True))
out.to_csv(OUT, index=False)
V = out[years0].to_numpy(float)
print(f"yearly matrix: {len(out)} topics x {len(years0)} years ({years0[0]}..{years0[-1]}) · "
      f"per-topic non-zero-suffix crop applied at fit time")
