"""
BFS web crawler with robots.txt compliance, URL normalization/dedup, and
per-domain politeness delay.

Run: python -m crawler.crawler
"""

import time
import urllib.parse as urlparse
from collections import deque
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from crawler.config import (
    SEED_URLS,
    MAX_PAGES,
    ALLOWED_DOMAINS,
    CRAWL_DELAY,
    REQUEST_TIMEOUT,
    USER_AGENT,
    DB_PATH,
)
from crawler.storage import init_db, get_conn, save_page, page_exists


def normalize_url(url: str) -> str:
    """Strip fragments, trailing slashes, and default ports so equivalent
    URLs dedupe correctly (https://a.com/x#foo == https://a.com/x)."""
    parsed = urlparse.urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    # Drop fragment; keep query string since it can change page content.
    normalized = urlparse.urlunparse((scheme, netloc, path, "", parsed.query, ""))
    return normalized


def get_domain(url: str) -> str:
    return urlparse.urlparse(url).netloc.lower()


class RobotsCache:
    """Fetches and caches robots.txt per domain so we don't refetch it for
    every single page on that domain.

    NOTE: We deliberately do NOT use RobotFileParser.read(), which issues its
    own bare urllib request with no custom User-Agent. Many sites (including
    Wikipedia) return 403 Forbidden to that generic request, and
    RobotFileParser treats a 401/403 as "disallow everything" — silently
    blocking a crawl that robots.txt would otherwise permit. Instead we fetch
    robots.txt ourselves (with our real User-Agent, via requests) and hand
    the text to rp.parse().
    """

    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self._cache: dict[str, RobotFileParser] = {}

    def is_allowed(self, url: str) -> bool:
        domain = get_domain(url)
        if domain not in self._cache:
            robots_url = f"{urlparse.urlparse(url).scheme}://{domain}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                resp = requests.get(
                    robots_url,
                    headers={"User-Agent": self.user_agent},
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code in (401, 403):
                    # Explicitly forbidden from even reading robots.txt —
                    # be conservative and disallow this domain.
                    rp.disallow_all = True
                elif resp.status_code >= 400:
                    # No robots.txt (404) or other client error -> treat as
                    # "no restrictions" per convention.
                    rp.allow_all = True
                else:
                    rp.parse(resp.text.splitlines())
            except requests.RequestException:
                # Unreachable -> default to allowing, but log so it's visible.
                print(f"[warn] could not fetch robots.txt for {domain}, allowing by default")
                rp.allow_all = True
            self._cache[domain] = rp
        rp = self._cache[domain]
        return rp.can_fetch(self.user_agent, url)


class Crawler:
    def __init__(self):
        self.robots = RobotsCache(USER_AGENT)
        self.visited: set[str] = set()
        self.queue: deque[str] = deque()
        self.last_request_time: dict[str, float] = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _respect_delay(self, domain: str):
        last = self.last_request_time.get(domain)
        if last is not None:
            elapsed = time.time() - last
            if elapsed < CRAWL_DELAY:
                time.sleep(CRAWL_DELAY - elapsed)
        self.last_request_time[domain] = time.time()

    def _domain_allowed(self, url: str) -> bool:
        if not ALLOWED_DOMAINS:
            return True
        return get_domain(url) in ALLOWED_DOMAINS

    def _extract_links(self, base_url: str, soup: BeautifulSoup) -> list[str]:
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("javascript:", "mailto:", "#")):
                continue
            absolute = urlparse.urljoin(base_url, href)
            links.append(normalize_url(absolute))
        return links

    def crawl(self):
        init_db(DB_PATH)
        for seed in SEED_URLS:
            self.queue.append(normalize_url(seed))

        pages_crawled = 0

        with get_conn(DB_PATH) as conn:
            while self.queue and pages_crawled < MAX_PAGES:
                url = self.queue.popleft()

                if url in self.visited:
                    continue
                self.visited.add(url)

                if not self._domain_allowed(url):
                    continue

                if page_exists(conn, url):
                    continue

                if not self.robots.is_allowed(url):
                    print(f"[skip: robots.txt] {url}")
                    continue

                domain = get_domain(url)
                self._respect_delay(domain)

                try:
                    resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
                    resp.raise_for_status()
                except requests.RequestException as e:
                    print(f"[error] {url} -> {e}")
                    continue

                # requests follows redirects by default, so resp.text is the
                # FINAL page's content — but resp.url may differ from the
                # `url` we requested (e.g. Wikipedia redirect pages like
                # Web_crawling -> Web_crawler). Save under the canonical
                # (post-redirect) URL so redirect aliases don't create
                # duplicate documents in the index.
                canonical_url = normalize_url(resp.url)
                if canonical_url != url:
                    if canonical_url in self.visited or page_exists(conn, canonical_url):
                        continue
                    self.visited.add(canonical_url)

                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    continue

                soup = BeautifulSoup(resp.text, "lxml")
                title = soup.title.string.strip() if soup.title and soup.title.string else url

                outbound_links = self._extract_links(canonical_url, soup)
                save_page(conn, canonical_url, title, resp.text, outbound_links)
                pages_crawled += 1
                print(f"[{pages_crawled}/{MAX_PAGES}] {title!r} -> {url}")

                for link in outbound_links:
                    if link not in self.visited and self._domain_allowed(link):
                        self.queue.append(link)

        print(f"\nDone. Crawled {pages_crawled} pages into {DB_PATH}")


if __name__ == "__main__":
    Crawler().crawl()
