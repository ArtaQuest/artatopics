# High or low — the balanced benchmark: reproduction bundle (v2)

    python3 reproduce.py <adstopics-dataset-balanced-v2.zip>    # scoreboard -> results.json, vs expected.json (4 dp)
    python3 reproduce.py <dataset> --fig1 --fig2 --fig3         # the manuscript's three figures (with bootstrap band)
    python3 reproduce.py <dataset> --tau-sens                   # the ladder at the adjacent threshold (robustness)
    python3 reproduce.py <dataset> --selftest                   # planted-rule protocol sanity

CPU-canonical (seed 7, deterministic). Regenerates and WRITES (results.json): the ceilings
(0.9635 / 0.9270), six references (incl. memory majority + AR pooled logistic), the phase
encoder-decoder (pure + memory), the direct pooled model, and the champion per-topic selected-5
ensemble (0.9183 horizon-AUC, 95% CI [0.9118, 0.9248], topic bootstrap B=2000) — from the
registered dataset zip alone. Also regenerates the dataset's shipped balanced_results.csv
(ONE scoreboard, ONE code path), prints the per-arm generalisation table (only the seasonal
copy survives the wall: +6.1pp; the other swaps are validation noise), and machine-compares
the champion's predictions cell-by-cell against the dataset's platform_predictions.json
(0 of 72,504 differ).
