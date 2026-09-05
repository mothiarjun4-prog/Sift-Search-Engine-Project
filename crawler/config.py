"""
Crawl configuration.

Edit SEED_URLS and MAX_PAGES to scope your crawl. Start small (50-200 pages)
to make sure everything works before scaling up to thousands.
"""

# Where the crawl starts. A diverse spread across unrelated topics gives
# genuine general-purpose coverage without needing tens of thousands of
# pages — BFS reaches many more distinct subject areas per page crawled
# when it starts from several different corners of Wikipedia, rather than
# needing many hops outward from just one or two seeds.
SEED_URLS = [
    "https://en.wikipedia.org/wiki/Web_search_engine",
    "https://en.wikipedia.org/wiki/Information_retrieval",
    "https://en.wikipedia.org/wiki/Spring_(framework)",       # covers software-framework queries directly
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Climate_change",
    "https://en.wikipedia.org/wiki/World_War_II",
    "https://en.wikipedia.org/wiki/Solar_System",
    "https://en.wikipedia.org/wiki/Association_football",
    "https://en.wikipedia.org/wiki/Renaissance",
    "https://en.wikipedia.org/wiki/Ancient_Rome",
    "https://en.wikipedia.org/wiki/Music",
    "https://en.wikipedia.org/wiki/Human_body",
    "https://en.wikipedia.org/wiki/Economics",
    "https://en.wikipedia.org/wiki/Democracy",
]

# Hard cap on total pages crawled. 2,000 pages at CRAWL_DELAY=1.0s is
# roughly a 33-minute crawl (single domain, so requests serialize
# regardless of how many distinct topics they cover). Raise further only if
# you're prepared for the proportionally longer runtime.
MAX_PAGES = 2000

# If set, only crawl URLs whose domain is in this list. Leave empty to allow
# any domain reachable via links (crawl will grow much faster and more
# unpredictably — good for a "mini Google" feel, harder to keep coherent).
ALLOWED_DOMAINS = ["en.wikipedia.org"]

# Seconds to wait between consecutive requests to the SAME domain. Keep this
# at 1.0+ to be a polite crawler and avoid getting rate-limited or blocked.
CRAWL_DELAY = 1.0

# HTTP request timeout in seconds.
REQUEST_TIMEOUT = 10

# User-Agent string sent with requests. Identify yourself honestly.
USER_AGENT = "MiniSearchEngineBot/0.1 (educational project; contact: you@example.com)"

# SQLite DB path for storing crawled pages.
DB_PATH = "data/crawled_pages.db"
