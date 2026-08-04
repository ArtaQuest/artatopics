# Four chart-derived topic sets against the sky — reproduction bundle

    pip install -r requirements.txt
    python3 reproduce.py astro4-datasets.zip                 # the registered dataset artifact
    python3 reproduce.py astro4-datasets.zip --quick 120     # deterministic subset (~3 min)
    python3 reproduce.py astro4-datasets.zip --from-cache atlas_signs.csv   # stats+figures in ~10 s

Fits every topic's own-modality monthly Trends series with the platform's canonical sinc model
(topic500_reference_solution.py, verbatim; mode="atlas") and scores BOTH season read-outs — the
fitted phase's sign AND the raw monthly-mean peak's sign — against the frozen lunar-month chart
classes shipped in the dataset, per modality and pooled, with fixed-seed cluster permutations,
rotation nulls, offset circular statistics and cross-modality consistency. Writes results.json
(declared values in expected.json), atlas_signs.csv (per-topic; bundled copy = canonical MPS run)
and fig1..fig4. ~25 min GPU / ~50 min CPU; deterministic per device (~±0.5 pt cross-device).
