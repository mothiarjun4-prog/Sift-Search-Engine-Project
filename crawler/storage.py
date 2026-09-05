"""
Storage layer for crawled pages, backed by SQLite.

Schema:
    pages(id, url UNIQUE, title, html, crawled_at)
    links(from_page_id, to_url)   -- raw link graph, used later for PageRank
"""

import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

from crawler.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    html TEXT,
    crawled_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS links (
    from_page_id INTEGER NOT NULL,
    to_url TEXT NOT NULL,
    FOREIGN KEY (from_page_id) REFERENCES pages(id)
);

CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(url);
CREATE INDEX IF NOT EXISTS idx_links_from ON links(from_page_id);
"""


@contextmanager
def get_conn(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH):
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)


def page_exists(conn, url: str) -> bool:
    cur = conn.execute("SELECT 1 FROM pages WHERE url = ?", (url,))
    return cur.fetchone() is not None


def save_page(conn, url: str, title: str, html: str, outbound_links: list[str]) -> int:
    """Insert a crawled page and its outbound links. Returns the page id."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO pages (url, title, html, crawled_at) VALUES (?, ?, ?, ?)",
        (url, title, html, datetime.now(timezone.utc).isoformat()),
    )
    if cur.lastrowid == 0:
        # Already existed (INSERT OR IGNORE hit the UNIQUE constraint)
        page_id = conn.execute("SELECT id FROM pages WHERE url = ?", (url,)).fetchone()[0]
    else:
        page_id = cur.lastrowid

    if outbound_links:
        conn.executemany(
            "INSERT INTO links (from_page_id, to_url) VALUES (?, ?)",
            [(page_id, link) for link in outbound_links],
        )
    return page_id


def count_pages(db_path: str = DB_PATH) -> int:
    with get_conn(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]


def print_stats(db_path: str = DB_PATH):
    with get_conn(db_path) as conn:
        n_pages = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        n_links = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
        top = conn.execute(
            "SELECT url, title FROM pages ORDER BY id DESC LIMIT 5"
        ).fetchall()
    print(f"Pages crawled: {n_pages}")
    print(f"Links recorded: {n_links}")
    print("Most recently crawled:")
    for url, title in top:
        print(f"  - {title!r}  ({url})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        print_stats()
    else:
        init_db()
        print(f"Initialized DB at {DB_PATH}")
