"""
Generates a short "here's why this result matched" snippet for a search
result — the same idea as the bolded excerpt under a Google result.

Reads from the already-extracted, capped-length text stored in index.db at
build time (see index/inverted_index.py's snippet_text column), NOT from
the original crawled HTML. This matters for two reasons: it avoids
re-parsing full page HTML with BeautifulSoup on every single search result
of every single query (a real, measured source of slow query latency), and
it means a deployed instance of this app never needs the (much larger) raw
crawl database at all -- only index.db, which is what actually gets shipped.
"""

import re

from index.inverted_index import get_conn, get_snippet_text

SNIPPET_WINDOW = 160  # characters of context shown around the match


def get_snippet(doc_id: int, raw_query_words: list[str]) -> str:
    """Best-effort snippet: find the first place any raw (unstemmed) query
    word appears as a WHOLE WORD in the stored text, and return a window
    around it. Falls back to the start of the text if nothing matches
    directly (this can happen since the query words are matched against
    stemmed forms, not literal text — e.g. query 'crawling' won't literally
    appear as 'crawling' if the page only ever says 'crawler'), or if the
    only occurrence happens to fall past the stored text's length cap.

    Word-boundary matching matters here: a naive substring search for
    "spring" would also match inside "Springer", pointing the snippet at
    an unrelated word that happens to contain the query as a prefix. \\b
    in the regex ensures we only match "spring" as its own word.
    """
    with get_conn() as conn:
        text = get_snippet_text(conn, doc_id)

    if not text:
        return ""

    lower_text = text.lower()

    best_pos = None
    for word in raw_query_words:
        match = re.search(r"\b" + re.escape(word.lower()) + r"\b", lower_text)
        if match and (best_pos is None or match.start() < best_pos):
            best_pos = match.start()

    if best_pos is None:
        snippet = text[: SNIPPET_WINDOW * 2]
    else:
        start = max(0, best_pos - SNIPPET_WINDOW // 2)
        end = min(len(text), best_pos + SNIPPET_WINDOW)
        snippet = text[start:end]

    snippet = snippet.strip()
    if best_pos is not None and best_pos - SNIPPET_WINDOW // 2 > 0:
        snippet = "…" + snippet
    if len(text) > (best_pos or 0) + SNIPPET_WINDOW:
        snippet = snippet + "…"
    return snippet
