# Will search go higher? — the rate-change benchmark: reproduction bundle

    python3 reproduce.py <adstopics-ratechange-dataset.zip>     # scoreboard -> results.json vs expected.json (4 dp)
    python3 reproduce.py <dataset> --fig1 --fig2 --fig3
    python3 reproduce.py <dataset> --selftest

CPU-canonical (seed 7, deterministic). Regenerates the ceilings (O3 0.8322 / O2 0.5947), the
references, the direct pooled model, and the CHAMPION — a refined seasonal climatology of the
rate-change direction (test AUC 0.5999, 95% CI [0.5948, 0.6052]) — from the registered dataset zip
alone, WRITES them to results.json (machine-compared to expected.json), and machine-compares the
champion's predictions to the dataset's platform_predictions.json (0 of 72,504 cells differ). The
full ladder incl. the encoder-decoder (which loses here: 0.5503/0.5777) is in the dataset's
src/ratechange.py and its shipped ratechange_results.csv. Flat months (34.8%) are ties, excluded.
