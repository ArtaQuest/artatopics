# AstroAttention — reproduction bundle (Journal of Seasonality)

    python3 reproduce.py <path-to-unzipped-adstopics-dataset>

Rebuilds the direction-classification protocol (210 clean months 2008-01..2025-06; the last 24
months are the untouched test) on the gated population and writes results.json + accuracy.png.
Declared numbers in expected.json; tolerance ±0.01 AUC (seed/device stochasticity of the
attention arms; the memory references reproduce exactly). Runtime ~15-25 min on CPU.
Research source ladder in src/ (the full experiment history; not needed to reproduce).
