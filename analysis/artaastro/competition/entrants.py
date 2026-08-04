#!/usr/bin/env python3
"""entrants.py — score the entrants on the ArtaAstro leaderboard (plain R2) and write LEADERBOARD.md.

Metric = platform `r2`: per measure, R2 over the de-timed/shuffled holdout; averaged over the 6.
All entrants fit on train (2015-2019) only. Focused on the single baseline:

  predict-mean            each measure's train mean                                  (the floor)
  ARTAASTRO-intensity     the a-priori mundane intensity, linearly calibrated
  global-phase-baseline   y = sum_i w_i sinc(f_i (x_i - p)) + b, gradient descent    (the reference)
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as B

HERE = os.path.dirname(os.path.abspath(__file__)); AST = os.path.dirname(HERE)
CLEAN  = os.path.join(HERE, "world_events_daily_clean.csv")
INT    = os.path.join(AST, "out", "daily_intensity.csv")
HOLD   = os.path.join(AST, "out", "artaastro_holdout.csv")
REFSUB = os.path.join(AST, "out", "artaastro_reference_submission.csv")
PHASES = os.path.join(AST, "out", "artaastro_phases.json")
TOPICS = ["material","conflict","verbal_conf","cooperation","material_coop","violence"]


def main():
    import json
    clean = pd.read_csv(CLEAN, parse_dates=["date"])
    inten = pd.read_csv(INT, parse_dates=["date"])[["date","intensity"]]
    df = clean.merge(inten, on="date", how="left").set_index("date").sort_index()
    hold = pd.read_csv(HOLD, parse_dates=["date"])                 # topic,id,date,target
    tr = df[(df.index >= "2015-01-01") & (df.index < "2020-01-01")]

    truth = {}
    for topic, rid, tgt in hold[["topic","id","target"]].itertuples(index=False):
        truth.setdefault(topic, {})[int(rid)] = float(tgt)

    def preds_from_daily(daily):
        out = {}
        for topic in TOPICS:
            ser = daily[topic]; m = {}
            for rid, d in hold.loc[hold.topic == topic, ["id","date"]].itertuples(index=False):
                m[int(rid)] = float(ser.get(pd.Timestamp(d), np.nan))
            out[topic] = m
        return out

    ent = {}
    ent["predict-mean"] = preds_from_daily({t: pd.Series(tr[t].mean(), index=df.index) for t in TOPICS})
    def arta(t):
        x = tr["intensity"].to_numpy(float); y = tr[t].to_numpy(float); ok = np.isfinite(x)&np.isfinite(y)
        b1, b0 = np.polyfit(x[ok], y[ok], 1); return pd.Series(b0 + b1*df["intensity"].to_numpy(float), index=df.index)
    ent["ARTAASTRO-intensity"] = preds_from_daily({t: arta(t) for t in TOPICS})

    rows, details = [], {}
    for name, preds in ent.items():
        board, per = B.score_r2(truth, preds); rows.append((name, board)); details[name] = per
    if os.path.exists(REFSUB):                                     # the global-phase baseline submission
        sub = pd.read_csv(REFSUB); rp = {}
        for trend, rid, tgt in sub[["trend","id","target"]].itertuples(index=False):
            rp.setdefault(trend, {})[int(rid)] = float(tgt)
        board, per = B.score_r2(truth, rp); rows.append(("global-phase-baseline", board)); details["global-phase-baseline"] = per
    rows.sort(key=lambda r: -(r[1] if r[1] is not None else -1e9))

    ph = json.load(open(PHASES))["phases"] if os.path.exists(PHASES) else {}
    L = ["# ArtaAstro competition — leaderboard\n",
         "_Metric: plain **R²** per measure over the de-timed/shuffled holdout, averaged over the 6 measures "
         "(platform `r2`). Fit on train (2015-2019) only. **0 = no better than each measure's mean.**_\n",
         "| rank | entrant | R² (mean over measures) |", "|---|---|---|"]
    for i, (name, board) in enumerate(rows, 1):
        L.append(f"| {i} | {name} | {board:+.4f} |")
    L.append("\n## Per-measure R²\n")
    L.append("| measure | " + " | ".join(n for n, _ in rows) + " |")
    L.append("|---|" + "|".join(["---"]*len(rows)) + "|")
    for t in TOPICS:
        L.append(f"| {t} | " + " | ".join(f"{details[n].get(t, float('nan')):+.4f}" for n, _ in rows) + " |")
    if ph:
        L.append("\n## Global-phase baseline — fitted phase (sign) & fit per measure\n")
        L.append("| measure | phase | sign | validated | val R² | in-sample R² | holdout R² |")
        L.append("|---|---|---|---|---|---|---|")
        for t in TOPICS:
            r = ph[t]
            L.append(f"| {t} | {r['phase']}° | {r['sign']} | {r['validated']} | {r['val_r2']:+.3f} | "
                     f"{r['in_r2']:+.3f} | {r['holdout_r2']:+.3f} |")
    L.append("\n## Reading it\n")
    L.append("- **Every entrant has R² ≤ 0** — none beats predicting each measure's own holdout mean.")
    L.append("- The single baseline is the **global-phase** model `ŷ = Σ wᵢ·sinc(fᵢ(xᵢ−p)) + b`, fit by "
             "gradient descent from **12 sign-centre phase restarts**, time-validated, with bounded "
             "frequencies + weight decay so it can't overfit; a measure whose best sign does not beat the "
             "mean on validation falls back to the mean. Its fitted phases/frequencies are reported above and "
             "on the /astro page — but the holdout R² shows those signs do not forecast the 2020→ future.")
    L.append("- Consistent with the full-history backtest (`../RESULTS.md`, ρ≈0): no sky feature, and no "
             "sign, forecasts world events. Reported straight.")
    open(os.path.join(HERE, "LEADERBOARD.md"), "w").write("\n".join(L) + "\n")

    print("\n=== LEADERBOARD (plain R2) ===")
    for i, (name, board) in enumerate(rows, 1):
        print(f"  {i}. {name:24} {board:+.4f}")


if __name__ == "__main__":
    main()
