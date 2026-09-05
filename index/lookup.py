"""
Quick manual sanity-check for the inverted index: given one or more query
words, show which documents contain them (plain boolean AND, no ranking —
that's Week 3). Useful for confirming the index actually works before
building anything fancier on top of it.

Run: python -m index.lookup web crawler
"""

import sys

from index.inverted_index import get_conn, get_postings, get_doc_meta, document_frequency
from index.text_processing import preprocess_query


def lookup(raw_query: str):
    terms = preprocess_query(raw_query)
    if not terms:
        print("Query reduced to zero terms after stopword removal — try different words.")
        return

    print(f"Query terms after preprocessing: {terms}\n")

    with get_conn() as conn:
        doc_sets = []
        for term in terms:
            postings = get_postings(conn, term)
            df = document_frequency(conn, term)
            print(f"  '{term}': appears in {df} document(s)")
            doc_sets.append({doc_id for doc_id, _, _ in postings})

        if not doc_sets:
            return

        matching = set.intersection(*doc_sets) if doc_sets else set()
        print(f"\n{len(matching)} document(s) contain ALL query terms:\n")
        for doc_id in list(matching)[:20]:
            meta = get_doc_meta(conn, doc_id)
            if meta:
                url, title, length = meta
                print(f"  - {title}  ({url})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m index.lookup <query words>")
        sys.exit(1)
    lookup(" ".join(sys.argv[1:]))
