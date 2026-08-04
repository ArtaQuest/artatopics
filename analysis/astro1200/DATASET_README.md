# astro4-datasets — four chart-derived topic sets (ImageTopics / NewsTopics / ShopTopics / YouTubeTopics)

- <Set>.csv — topic, class (FROZEN chart-derived season), n_lunations, avg_score, month (2008-01..
  2025-06), value (own-modality monthly interest, own max=100), 11 sidereal body phases (deg).
- preregistration_<prop>.json — every selected topic's full per-season chart record and class,
  exactly as frozen (committed before any series was fetched).
- lunations_<prop>.json + new_moons.json — the lunar calendar (new moon → new moon) and each
  lunation's sidereal season.
- datasets_manifest.json — window, counts, skips.
Provenance: 1,115 raw Google Trends Explore lunar-month "Top queries" CSVs (5 properties incl. the
pre-excluded web set), archived in the platform repository.
