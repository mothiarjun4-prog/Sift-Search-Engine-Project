# Mini Search Engine

A search engine built from scratch: crawler → inverted index → BM25/PageRank ranking → query API.

## Project Structure

```
search-engine/
├── crawler/
│   ├── crawler.py        # BFS web crawler with robots.txt + politeness
│   ├── storage.py        # SQLite storage for crawled pages
│   └── config.py         # Crawl settings (seeds, limits, delay)
├── index/                # (Week 2) inverted index builder
├── rank/                 # (Week 3) TF-IDF, BM25, PageRank
├── query/                # (Week 4) query parser + Flask/FastAPI API
├── data/                 # crawled_pages.db lives here
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Week 1: Run the crawler

Edit `crawler/config.py` to set your seed URLs and crawl limit, then:

```bash
python -m crawler.crawler
```

This will:
- Start from your seed URLs and do a BFS crawl
- Respect `robots.txt` for each domain
- Wait `CRAWL_DELAY` seconds between requests to the same domain (politeness)
- Normalize and dedupe URLs so it doesn't loop forever
- Store page HTML, title, and outbound links in `data/crawled_pages.db`

Check progress:
```bash
python -m crawler.storage stats
```

## Week 2: Build the inverted index

Once you've crawled some pages (Week 1), build the index over them:

```bash
python -m index.build_index
```

This will:
- Read every page from `data/crawled_pages.db`
- Strip HTML down to visible text (dropping nav/script/style noise)
- Tokenize, remove stopwords, and stem each word (Porter stemmer — no
  downloaded corpora needed, so this runs fully offline)
- Build the inverted index (term → list of docs containing it, with term
  frequency and token positions) into `data/index.db`

Sanity-check retrieval directly (plain boolean AND, no ranking yet):

```bash
python -m index.lookup web crawler
```

Check index stats any time:
```bash
python -m index.inverted_index
```

## Week 4: Query engine + web UI

Once the index is built (Weeks 1-3), start the web search UI:

```bash
python -m query.app
```

Then open **http://127.0.0.1:5000** in a browser.

Features:
- **Plain queries** — `web crawler` — ranked with BM25 (default) or TF-IDF, toggleable in the UI
- **Exact phrase search** — `"web crawler"` (with quotes) — only matches
  documents where those words appear consecutively, using the token
  positions stored back in Week 2. Falls back to normal ranking (with a
  notice) if no document contains the exact phrase.
- **Typo tolerance** — misspelled query words are corrected to the closest
  real word in the index (via Levenshtein edit distance) if no exact match
  exists, with a "did you mean" notice shown in the UI.
- **Snippets** — each result shows a real excerpt from the page around
  where your query words actually appear, not just the title.

Known limitation: phrase search matches on the sequence of CONTENT words
only, since stopwords were never indexed (Week 2). `"web crawler"` and
`"a web, crawler,"` match identically — this is a deliberate tradeoff for
a smaller/cleaner index, not a bug.

## Roadmap

- [x] Week 1: Crawler
- [x] Week 2: Inverted index (tokenize, stem, build term → doc postings)
- [x] Week 3: Ranking (TF-IDF → BM25, tuned against real data)
- [x] Week 4: Query engine (phrases, typo tolerance) + web UI

## Notes on scope/ethics

- Crawling is capped to a small seed list and page limit by default — raise `MAX_PAGES` deliberately.
- Robots.txt rules are respected; don't remove that check.
- Be a good citizen: keep `CRAWL_DELAY` reasonable (1+ second) so you don't hammer any one site.
