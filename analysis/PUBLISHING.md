# ArtaQuest Research — the publishing pipeline

A small, Nature-style open journal at **https://artaquest.org/research/**. Two articles ship today:

1. **The Topics Seasonality Model** — `/research/?article=seasonality`
2. **Google Trends Recency Bias** — `/research/?article=recency`

Every article is: peer-discussable, openly licensed (CC BY 4.0), fully reproducible (one-click Colab on
hosted data), citable in five formats, listed on the author's profile, and built to be indexed by Google
Scholar.

## What's automated (code in `analysis/`)

| Step | Command | Output |
|------|---------|--------|
| Fit topics (weekly, top-100) | `python3 analysis/weekly_fit.py` | `_registry.json` |
| Recency sweep | `python3 analysis/recency_experiment.py` | `_recency.json` |
| Reproducibility bundle | `python3 analysis/build_pub.py` | `_pub/` data files → uploads/research/ |
| Colab notebooks | `python3 analysis/build_notebooks.py` | `*.ipynb` (hosted as a gist) |
| **SPA atlas + article index** | `python3 analysis/export_research.py` | `artaquest-web/src/data/research.json`, `articles.json` |
| **PDFs + landing pages + feed** | `python3 analysis/build_papers.py` | `_pub/papers/` |
| DOIs (needs token, see below) | `python3 analysis/zenodo_deposit.py --publish` | `_dois.json` |

Then: `cd artaquest-web && npm run build`, re-host `_pub/papers/` to `/srv/htdocs/papers/`, and
`tools/ticket-agent/aq-deploy studio push --options themes`. The papers directory and the reproducibility
bundle are static (tar-over-ssh); the SPA ships in the theme.

## Scholar-indexable surface (all live, verified)

- Crawlable HTML landing page per article with **Highwire `citation_*` meta** (title, authors,
  institution, date, journal, volume, issue, pdf_url, abstract, keywords, DOI when minted).
- **Text-extractable PDF** (not a scan), title as the largest text on page 1, reachable from the
  landing page via `citation_pdf_url`.
- **JSON-LD `ScholarlyArticle`** + OpenGraph/Twitter cards.
- **Sitemap** `https://artaquest.org/papers/sitemap.xml` (HTML + PDF), referenced from `robots.txt`.
- **RSS feed** `https://artaquest.org/papers/feed.xml`.
- Five citation formats (APA, Nature, BibTeX, RIS, CSL-JSON) on the page + downloadable `.bib/.ris/.csl.json`.
- Nature-style end-matter: Data/Code availability, Author contributions, Competing interests, Funding.
- Co-authorship supported: add entries to `AUTHORS` in `export_research.py` **and** `build_papers.py`
  (name, affiliation, optional ORCID). Articles appear on each author's `/u/<id>` profile via the `uid`.

## Operator handoff — the steps that need you (external accounts)

These cannot be done from the repo; they need your credentials and are one-time:

1. **DOI (Zenodo).** Make a token (`deposit:write` + `deposit:actions`) at
   <https://zenodo.org/account/settings/applications/tokens/new>, then:
   ```
   export ZENODO_TOKEN=...
   python3 analysis/zenodo_deposit.py --publish --sandbox   # rehearse first
   python3 analysis/zenodo_deposit.py --publish             # real, irreversible DOIs
   python3 analysis/export_research.py && python3 analysis/build_papers.py
   # then rebuild SPA + re-host papers + deploy — DOIs flow into every citation automatically
   ```
2. **Google Search Console.** Verify `artaquest.org`, submit `/papers/sitemap.xml`. Scholar discovers
   work by crawling; this speeds it up. Indexing then takes Scholar **days to weeks** on its own schedule —
   nothing more to do but wait.
3. **Google Scholar profile** (`user=Lkif9tkAAAAJ`). Once Scholar has crawled the pages, the works appear
   as suggestions to add; or add them manually (title + the landing-page URL). A DOI makes this reliable.

## Adding a third article

Write the fit/experiment, add its metadata block to `articles` in `export_research.py`, its sections to
`build_papers.py:main()`, an entry in `PAPER`/`PTITLE`/`KEYWORDS`/`ANUM`, and a discussion thread id.
Re-run the table above. Everything else (cards, profile listing, feed, sitemap, citations) is generated.
