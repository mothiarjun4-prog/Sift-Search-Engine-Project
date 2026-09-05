"""
Reads every crawled page from crawler's SQLite DB, runs it through the text
pipeline, and builds the inverted index.

Run: python -m index.build_index
"""

import hashlib
import sqlite3

from crawler.config import DB_PATH as CRAWL_DB_PATH
from index.inverted_index import init_index_db, clear_index, get_conn, save_doc, print_stats, vacuum
from index.text_processing import html_to_text, preprocess


def build_index(crawl_db_path: str = CRAWL_DB_PATH):
    init_index_db()
    clear_index()  # safe to re-run: always rebuilds from scratch

    crawl_conn = sqlite3.connect(crawl_db_path)
    pages = crawl_conn.execute("SELECT id, url, title, html FROM pages").fetchall()
    crawl_conn.close()

    if not pages:
        print(f"No pages found in {crawl_db_path} — run the Week 1 crawler first.")
        return

    # Content-hash dedup: Wikipedia (and many sites) redirect some URLs to a
    # canonical page (e.g. Web_crawling -> Web_crawler). Before the crawler
    # fix, redirects got saved under both the original and canonical URL,
    # producing two identical documents in the index — which then show up
    # as literal duplicate results at query time. This catches that even
    # for data crawled before the fix, without needing to re-crawl: any two
    # pages whose extracted text is identical are the same content
    # regardless of what URL they were saved under.
    seen_hashes: set[str] = set()
    skipped_duplicates = 0

    with get_conn() as index_conn:
        indexed = 0
        for i, (doc_id, url, title, html) in enumerate(pages, start=1):
            text = html_to_text(html)
            normalized = " ".join(text.split())  # collapse whitespace before hashing
            content_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()

            if content_hash in seen_hashes:
                print(f"[{i}/{len(pages)}] skip duplicate content -> {title!r} ({url})")
                skipped_duplicates += 1
                continue
            seen_hashes.add(content_hash)

            tokens = preprocess(text)
            save_doc(index_conn, doc_id, url, title, tokens, full_text=normalized)
            indexed += 1
            print(f"[{i}/{len(pages)}] indexed {len(tokens)} tokens -> {title!r}")

    print(f"\nDone. Indexed {indexed} pages, skipped {skipped_duplicates} duplicate(s).\n")

    print("Reclaiming unused space (VACUUM)...")
    vacuum()

    print_stats()


if __name__ == "__main__":
    build_index()
