"""
The inverted index itself: term -> list of (doc_id, term_frequency, positions).

This is the core data structure of a search engine. Instead of scanning
every document for a query term (O(num_docs)), we look the term up directly
and get back exactly the documents that contain it.

Schema:
    doc_meta(doc_id, url, title, length, snippet_text)
        length = number of indexed tokens in the doc (post-stopword/stem),
        needed later for BM25's document-length normalization.
        snippet_text = the first SNIPPET_TEXT_MAX_CHARS of extracted plain
        text (see index/build_index.py), stored once at index-build time so
        query/snippet.py never needs to re-open the raw crawled HTML (which
        is both much larger on disk and slower to re-parse on every single
        search). Capped in length rather than storing full article text so
        index.db stays a bounded, deployable size regardless of how long
        any given crawled page is -- the tradeoff is that a query term
        appearing only very late in a long article won't be found for the
        snippet (get_snippet() falls back to the start of the stored text
        in that case), which is an acceptable loss for a short UI excerpt.

    postings(term, doc_id, tf, positions)
        one row per (term, doc) pair. tf = how many times the term appears
        in that doc. positions = JSON list of token offsets, kept for
        phrase-search support ("exact phrase" queries need to check that
        positions are consecutive across terms).
"""

import sqlite3
from contextlib import contextmanager

INDEX_DB_PATH = "data/index.db"

# Cap on how much plain text we store per doc for snippet generation. Keeps
# index.db size bounded and predictable rather than scaling with the size
# of the largest crawled page. Lowered from 4000 -> 1200 specifically to
# get index.db under GitHub's 100MB per-file limit for a 2000+ page corpus
# without needing Git LFS. The actual tradeoff is narrow: get_snippet()'s
# displayed window is only ~160 characters regardless of this cap, so this
# only matters if a query term's FIRST occurrence in a page falls after
# character 1200 — in which case the snippet falls back to the article's
# opening text rather than the true match location. For Wikipedia-style
# prose, the most relevant content is usually in the opening paragraphs
# anyway, so this is a small, deliberate quality/size tradeoff, not a bug.
SNIPPET_TEXT_MAX_CHARS = 1200

SCHEMA = """
CREATE TABLE IF NOT EXISTS doc_meta (
    doc_id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    length INTEGER NOT NULL,
    snippet_text TEXT
);

CREATE TABLE IF NOT EXISTS postings (
    term TEXT NOT NULL,
    doc_id INTEGER NOT NULL,
    tf INTEGER NOT NULL,
    positions TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES doc_meta(doc_id)
);

CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);
"""


@contextmanager
def get_conn(db_path: str = INDEX_DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def encode_positions(positions: list[int]) -> str:
    """Compact position encoding: comma-joined DELTAS (gap from the
    previous position) instead of a JSON list of absolute positions.

    Two savings stack here: dropping JSON's brackets/quotes/spacing
    overhead (fixed cost per row, multiplied across ~1M+ postings rows),
    and delta encoding turning large absolute positions into small gap
    numbers for common terms scattered throughout a long document --
    exactly the rows that dominate index.db's size. Measured ~48% smaller
    on realistic test data.
    """
    if not positions:
        return ""
    deltas = [positions[0]] + [positions[i] - positions[i - 1] for i in range(1, len(positions))]
    return ",".join(str(d) for d in deltas)


def decode_positions(encoded: str) -> list[int]:
    """Inverse of encode_positions: reconstruct absolute positions from
    comma-joined deltas via a running cumulative sum."""
    if not encoded:
        return []
    deltas = [int(x) for x in encoded.split(",")]
    positions = [deltas[0]]
    for d in deltas[1:]:
        positions.append(positions[-1] + d)
    return positions


def vacuum(db_path: str = INDEX_DB_PATH):
    """Reclaim unused space left behind by DROP TABLE / rebuild cycles.
    Must run on its own connection, outside any open transaction -- SQLite
    refuses to VACUUM while a transaction is active, which is exactly why
    this is a separate function rather than folded into get_conn's usual
    commit-on-exit pattern.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("VACUUM")
    conn.close()


def init_index_db(db_path: str = INDEX_DB_PATH):
    """Create the index schema, dropping any existing tables first.

    build_index.py always fully regenerates the index from crawled_pages.db
    on every run (never incrementally updates it), so it's safe -- and
    necessary -- to drop-and-recreate here rather than CREATE TABLE IF NOT
    EXISTS, which silently does nothing if a table already exists with an
    OUTDATED schema (exactly what happened when snippet_text was added:
    old index.db files kept their old doc_meta structure, and every
    INSERT referencing the new column failed at runtime instead of at
    schema-creation time).
    """
    with get_conn(db_path) as conn:
        conn.executescript("DROP TABLE IF EXISTS postings; DROP TABLE IF EXISTS doc_meta;")
        conn.executescript(SCHEMA)


def clear_index(db_path: str = INDEX_DB_PATH):
    """Wipe the index so build_index.py can be re-run cleanly (e.g. after
    re-crawling or changing the tokenizer)."""
    with get_conn(db_path) as conn:
        conn.executescript("DELETE FROM postings; DELETE FROM doc_meta;")


def save_doc(conn, doc_id: int, url: str, title: str, tokens: list[str], full_text: str = ""):
    """Write one document's metadata and postings.

    tokens is the full preprocessed token list for the doc (in order) — we
    derive term frequency and positions from it here rather than asking the
    caller to compute them, so there's exactly one place that logic lives.

    full_text is the raw extracted (pre-stemming) page text; only the first
    SNIPPET_TEXT_MAX_CHARS of it is stored, purely so query/snippet.py has
    human-readable text to show without needing the original crawled HTML.
    """
    conn.execute(
        "INSERT OR REPLACE INTO doc_meta (doc_id, url, title, length, snippet_text) VALUES (?, ?, ?, ?, ?)",
        (doc_id, url, title, len(tokens), full_text[:SNIPPET_TEXT_MAX_CHARS]),
    )

    term_positions: dict[str, list[int]] = {}
    for position, term in enumerate(tokens):
        term_positions.setdefault(term, []).append(position)

    conn.executemany(
        "INSERT INTO postings (term, doc_id, tf, positions) VALUES (?, ?, ?, ?)",
        [
            (term, doc_id, len(positions), encode_positions(positions))
            for term, positions in term_positions.items()
        ],
    )


def get_snippet_text(conn, doc_id: int) -> str:
    """The stored (capped-length) plain text for a doc, used to build a
    search-result snippet without touching the original crawled HTML."""
    row = conn.execute(
        "SELECT snippet_text FROM doc_meta WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    return row[0] if row and row[0] else ""


def get_postings(conn, term: str) -> list[tuple[int, int, list[int]]]:
    """Return [(doc_id, tf, positions), ...] for every doc containing term."""
    rows = conn.execute(
        "SELECT doc_id, tf, positions FROM postings WHERE term = ?", (term,)
    ).fetchall()
    return [(doc_id, tf, decode_positions(positions)) for doc_id, tf, positions in rows]


def document_frequency(conn, term: str) -> int:
    """Number of documents containing term at least once. Core ingredient
    of IDF (inverse document frequency) in Week 3's ranking."""
    row = conn.execute(
        "SELECT COUNT(*) FROM postings WHERE term = ?", (term,)
    ).fetchone()
    return row[0]


def get_doc_meta(conn, doc_id: int) -> tuple[str, str, int] | None:
    row = conn.execute(
        "SELECT url, title, length FROM doc_meta WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    return row


def total_docs(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM doc_meta").fetchone()[0]


def average_doc_length(conn) -> float:
    """Needed by BM25's length-normalization term in Week 3."""
    row = conn.execute("SELECT AVG(length) FROM doc_meta").fetchone()
    return row[0] or 0.0


def vocabulary_size(conn) -> int:
    return conn.execute("SELECT COUNT(DISTINCT term) FROM postings").fetchone()[0]


def get_vocabulary(conn) -> list[str]:
    """Every distinct term in the index. Used for typo correction: if a
    query word isn't in the vocabulary at all, we search this list for the
    closest match instead of just returning zero results."""
    rows = conn.execute("SELECT DISTINCT term FROM postings").fetchall()
    return [row[0] for row in rows]


def get_term_positions_for_doc(conn, term: str, doc_id: int) -> list[int]:
    """Positions of `term` within one specific document. Used by phrase
    search to check whether several terms appear at consecutive positions
    (i.e. as a phrase) rather than just anywhere in the document."""
    row = conn.execute(
        "SELECT positions FROM postings WHERE term = ? AND doc_id = ?", (term, doc_id)
    ).fetchone()
    return decode_positions(row[0]) if row else []


def print_stats(db_path: str = INDEX_DB_PATH):
    with get_conn(db_path) as conn:
        n_docs = total_docs(conn)
        vocab = vocabulary_size(conn)
        avg_len = average_doc_length(conn)
        top_terms = conn.execute(
            """
            SELECT term, COUNT(*) as df FROM postings
            GROUP BY term ORDER BY df DESC LIMIT 10
            """
        ).fetchall()
    print(f"Documents indexed: {n_docs}")
    print(f"Vocabulary size:   {vocab}")
    print(f"Avg doc length:    {avg_len:.1f} tokens")
    print("Most common terms (by document frequency):")
    for term, df in top_terms:
        print(f"  {term:<15} appears in {df} docs")


if __name__ == "__main__":
    print_stats()
