"""
The full Week 4 query pipeline, tying together everything built so far:

  raw query string
    -> parse into free terms + quoted phrases (query/parser.py)
    -> typo-correct any term not in the vocabulary (query/fuzzy.py)
    -> rank ALL terms (free + phrase words) with BM25/TF-IDF (rank/)
    -> filter results down to docs that actually contain each phrase as a
       consecutive sequence, not just the words anywhere (query/phrase.py)
    -> gracefully fall back to unfiltered ranking if the phrase filter
       would otherwise return nothing, rather than showing a dead end
"""

from index.inverted_index import get_conn
from query.parser import parse_query
from query.phrase import filter_by_phrases
from rank.bm25 import bm25_score_terms
from rank.tfidf import tfidf_score_terms


def search(raw_query: str, method: str = "bm25", top_k: int = 10, k1: float = None, b: float = None) -> dict:
    parsed = parse_query(raw_query)
    all_terms = parsed["free_terms"] + [t for phrase in parsed["phrases"] for t in phrase]

    if not all_terms:
        return {"results": [], "corrections": {}, "phrase_fallback": False}

    # Rank on ALL terms first (phrase words count toward relevance too, not
    # just as a filter) — over-fetch a larger candidate pool than top_k
    # since phrase filtering below may eliminate some of them.
    pool_size = max(top_k * 5, 50)
    if method == "tfidf":
        ranked = tfidf_score_terms(all_terms, top_k=pool_size)
    else:
        kwargs = {}
        if k1 is not None:
            kwargs["k1"] = k1
        if b is not None:
            kwargs["b"] = b
        ranked = bm25_score_terms(all_terms, top_k=pool_size, **kwargs)

    phrase_fallback = False
    if parsed["phrases"]:
        with get_conn() as conn:
            ranked_ids = {r["doc_id"] for r in ranked}
            kept_ids = filter_by_phrases(conn, ranked_ids, parsed["phrases"])

        if kept_ids:
            ranked = [r for r in ranked if r["doc_id"] in kept_ids]
        else:
            # No document satisfies the exact phrase — rather than show
            # nothing, fall back to the unfiltered relevance ranking and
            # tell the caller so the UI can be honest about it.
            phrase_fallback = True

    return {
        "results": ranked[:top_k],
        "corrections": parsed["corrections"],
        "phrase_fallback": phrase_fallback,
    }
