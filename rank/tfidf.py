"""
TF-IDF ranking: score each candidate doc by how much it emphasizes the
query terms, weighted by how distinctive each term is across the corpus.

- TF (term frequency): a doc mentioning "crawler" 10 times is probably more
  about crawlers than one mentioning it once in passing.
- IDF (inverse document frequency): a term appearing in 190/200 docs (like
  "web" in this corpus) tells you almost nothing about relevance, so it
  should count for very little. A term in 3/200 docs (like "pagerank") is
  highly distinctive, so it should count for a lot.

We use log-dampened TF (1 + log(tf)) rather than raw tf, which is standard
practice: the difference between a term appearing 1 time vs 2 times matters
a lot; the difference between 50 times and 51 times should barely matter.
"""

import math

from index.inverted_index import get_conn, total_docs
from index.text_processing import preprocess_query
from rank.scoring import gather_term_stats, candidate_doc_ids, fetch_results


def idf(n_docs: int, df: int) -> float:
    """Standard IDF. +1 in the denominator avoids division by zero if a
    query term somehow has df=0 (shouldn't happen here since such terms
    wouldn't produce any candidates, but keeps this function safe standalone)."""
    return math.log(n_docs / (df + 1))


def tf_weight(tf: int) -> float:
    """Log-dampened term frequency. tf=0 -> 0 (not log(1)=0 by coincidence,
    but because a term that doesn't appear contributes nothing)."""
    return 0.0 if tf == 0 else 1.0 + math.log(tf)


def tfidf_score_terms(query_terms: list[str], top_k: int = 10) -> list[dict]:
    """Core scoring logic, taking an already-preprocessed term list. Split
    out from tfidf_search() so query/search_engine.py can feed it terms
    that have already been through phrase-splitting and typo-correction,
    without re-running preprocess() on top of that."""
    if not query_terms:
        return []

    with get_conn() as conn:
        n_docs = total_docs(conn)
        term_stats = gather_term_stats(conn, query_terms)
        candidates = candidate_doc_ids(term_stats)

        scored = []
        for doc_id in candidates:
            score = 0.0
            for ts in term_stats:
                tf = ts.doc_tf.get(doc_id, 0)
                if tf == 0:
                    continue
                score += tf_weight(tf) * idf(n_docs, ts.df)
            scored.append((doc_id, score))

        return fetch_results(conn, scored, top_k)


def tfidf_search(raw_query: str, top_k: int = 10) -> list[dict]:
    query_terms = preprocess_query(raw_query)
    return tfidf_score_terms(query_terms, top_k)
