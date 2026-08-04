#!/usr/bin/env python3
"""Build the canonical citations matrix from the platform's public rail (operator 2026-07-23).

SOURCE: GET https://artaquest.org/wp-json/aq/v1/citations/quarterly — OpenAlex snapshot 2026-06-26
(CC0); citations_received = the EVENT measure (each reference edge dated by the CITING work's quarter).

PROCESSING (audited 2026-07-23):
  1. Drop subfield_id='none' (the unclassified bucket — not a topic).
  2. DATE-PRECISION FIX: quarter=0 rows are YEAR-only-dated citations (~30% overall; 60% in 1900,
     12% in 2024, and field-systematic — clinical fields are dated more precisely than chemistry).
     Dropping them biased the shares; instead each field-year's q0 total is spread across
     that year's quarters PRO-RATA to the field-year's own dated split (even split only when
     dated support is thin). Verified AUC-neutral and fairer
     (fields beating baseline 76.9% -> 80.9%).
  3. Window 1900Q1..2026Q1 — every quarter back to 1900 has >=133 active subfields; pre-1900 the
     record degenerates (1880: 52, 1850: ~2). 2026Q1 is ~17% under-ingested at the snapshot edge,
     but its SHARES are only mildly distorted (median |rel diff| 5.5% vs 4.2% for a normal
     quarter-to-quarter step) so it is kept, with the caveat published.
  4. Universe: the 251 subfields present in >=95% of 1980-2025 quarters (the modern taxonomy),
     extended back to 1900 (fields born mid-stream are honest zeros before birth). This set covers
     100% of classified quarter-dated citations in every era.

  python3 analysis/citations/build_matrix.py [path-to-raw-csv]
"""
import sys
import numpy as np, pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else \
    "/private/tmp/claude-501/-Users-arash-Studio-artaquest/8c3c064a-abad-4f0f-8dd6-5a1588c5e7b6/scratchpad/citations-quarterly.csv"
OUT = "analysis/citations/citations_received_quarterly.csv"
LO_Y, HI = 1700, 2026            # window 1700Q1 .. 2026Q1 (operator 2026-07-24: since 1700)

raw = pd.read_csv(RAW, low_memory=False)
raw = raw[pd.to_numeric(raw.subfield_id, errors="coerce").notna()].copy()   # drop 'none'
raw["subfield_id"] = raw.subfield_id.astype(float).astype(int)

# Universe: >=95% presence over 1980-2025 quarter-dated
qd = raw[(raw.quarter >= 1) & (raw.quarter <= 4)].copy()
mod = qd[qd.year.between(1980, 2025)]
mod_q = mod.year.astype(int).astype(str) + "Q" + mod.quarter.astype(int).astype(str)
pres = mod.assign(q=mod_q).groupby("subfield_id").q.nunique()
nq = mod_q.nunique()
keep_ids = sorted(pres[pres >= nq * 0.95].index.tolist())
assert len(keep_ids) == 251, f"universe changed: {len(keep_ids)}"

r = raw[raw.subfield_id.isin(keep_ids) & raw.year.between(LO_Y, HI)].copy()
quarters = [f"{yq//4}Q{yq%4+1}" for yq in range(LO_Y * 4, HI * 4 + 1)]

# quarter-dated counts
cur = r[(r.quarter >= 1) & (r.quarter <= 4)].copy()
cur["q"] = cur.year.astype(int).astype(str) + "Q" + cur.quarter.astype(int).astype(str)
piv = cur.pivot_table(index="subfield_id", columns="q", values="citations_received",
                      aggfunc="sum", fill_value=0).reindex(columns=quarters, fill_value=0) \
                      .reindex(index=keep_ids, fill_value=0).astype(float)

# DATE-PRECISION FIX: spread each field-year's q0 total across its four quarters PRO-RATA to that
# same subfield-year's dated quarterly split (parameter-free; preserves each field's own within-year
# pattern — a uniform split would flatten the seasonal band the quarterly model scores on). Gate on
# adequate dated support (>=3 nonzero quarters); with thin support fall back to an even split of the
# year (never borrow another field's shape — that could fabricate seasonality).
q0 = r[r.quarter == 0].groupby(["subfield_id", "year"]).citations_received.sum()
for (sid, yr), v in q0.items():
    cols = [f"{yr}Q{qq}" for qq in range(1, 5) if f"{yr}Q{qq}" in piv.columns]
    if not cols: continue
    dated = np.array([piv.at[sid, c] for c in cols], float)
    if (dated > 0).sum() >= 3:
        w = dated / dated.sum()
    else:
        w = np.full(len(cols), 1.0 / len(cols))
    for c, wi in zip(cols, w):
        piv.at[sid, c] += v * wi

meta = raw.drop_duplicates("subfield_id").set_index("subfield_id")[["subfield", "field", "domain"]]
out = meta.loc[keep_ids].reset_index().join(piv.reset_index(drop=True))
out.to_csv(OUT, index=False)

V = out[quarters].to_numpy(float)
S = 100 * V / np.maximum(V.sum(0, keepdims=True), 1)
print(f"matrix: {len(out)} subfields x {len(quarters)} quarters ({quarters[0]}..{quarters[-1]})")
print(f"share sums: {S.sum(0).min():.4f}..{S.sum(0).max():.4f} · q0 mass folded in: "
      f"{int(q0.sum()):,} of {int(V.sum()):,} total")
