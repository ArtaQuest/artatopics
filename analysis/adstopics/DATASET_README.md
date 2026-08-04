# adstopics — the advertising-taxonomy YouTube-search seasonality dataset

- verticals.csv — the Google Ads content taxonomy snapshot (2,558 category paths)
- vocabulary.json — 3,085 lowercase topics extracted by splitting every path on "/", "&", ","
- blacklist.json — topics excluded for cache-key collisions (see the paper's data-quality note)
- series/<slug>.csv — each topic's worldwide monthly YouTube-search interest (Google Trends,
  columns Time,v; 270 months 2004-01..2026-06; the study uses the clean window 2008-01..2025-06
  with the most recent 12 months excluded entirely)
- phases.csv — the twelve celestial phases per clean-window month (synodic moon elongation +
  eleven sidereal body longitudes, degrees; moon validated against Meeus)
Audit findings are summarised in the paper's data section (the census script in the model bundle's src/ is part of the archival research ladder).
- atlas_topics.csv — the final full-vocabulary Gaussian atlas (per-topic sign, lamps, R², season-led flag)
- attention_atlas.csv — the anchored AstroAttention classification (ensemble phase, sign, confidence)
