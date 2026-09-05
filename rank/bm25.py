"""
BM25 ranking — what Elasticsearch, Lucene, and most production search
engines actually use instead of plain TF-IDF.

BM25 fixes two specific weaknesses of TF-IDF:

1. TF-IDF's tf weight grows (roughly) without bound as term frequency
   increases. BM25's tf term saturates: going from 1 to 2 occurrences of a
   term matters a lot, but going from 20 to 21 barely moves the score. This
   models the intuition that a word appearing many times is a sign of
   relevance, but not linearly forever.

2. TF-IDF has no explicit way to penalize long documents for just naturally
   containing more words (and thus more term occurrences) than short ones.
   BM25 explicitly normalizes term frequency against document length
   relative to the corpus's average length, via the `b` parameter.

Formula, per query term t and doc d:

    score(t, d) = idf(t) * (tf(t,d) * (k1 + 1))
                  ---------------------------------------------------
                  (tf(t,d) + k1 * (1 - b + b * len(d) / avgdl))

- k1 controls how quickly tf saturates (higher = slower saturation, more
  like raw TF-IDF; 1.2-2.0 is typical, we use 1.5).
- b controls how much document length is penalized (0 = no length
  normalization at all, 1 = full normalization; 0.75 is the standard default).
"""

import math

from index.inverted_index import get_conn, total_docs, average_doc_length, get_doc_meta
from index.text_processing import preprocess_query
from rank.scoring import gather_term_stats, candidate_doc_ids, fetch_results

K1 = 1.5
# Standard BM25 defaults use b=0.75, but testing against an adversarial
# synthetic corpus (a very short, off-topic document containing both query
# terms) showed 0.75 lets length-normalization overpower genuine topical
# relevance. Sweeping b from 0.75 down to 0 showed the on-topic result
# overtakes the off-topic one somewhere around b=0.3-0.1. We land on 0.3 as
# a middle ground: still penalizes long, padded documents somewhat, without
# over-rewarding very short ones. Re-tune this against your real corpus —
# the right value is corpus-dependent, not a universal constant.
B = 0.3


def bm25_idf(n_docs: int, df: int) -> float:
    """BM25's IDF variant (Robertson-Sparck Jones), not the same formula as
    plain TF-IDF's idf(). The +1 inside the log keeps the value non-negative
    even for terms that appear in more than half the corpus — plain IDF can
    go negative there, which would let a common term actively *hurt* a
    doc's score, which is not the intended behavior."""
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1)


def bm25_score_terms(query_terms: list[str], top_k: int = 10, k1: float = K1, b: float = B) -> list[dict]:
    """Core scoring logic, taking an already-preprocessed term list. See
    tfidf_score_terms() for why this split exists."""
    if not query_terms:
        return []

    with get_conn() as conn:
        n_docs = total_docs(conn)
        avgdl = average_doc_length(conn) or 1.0  # guard against empty index
        term_stats = gather_term_stats(conn, query_terms)
        candidates = candidate_doc_ids(term_stats)

        # Cache doc lengths since we look each one up once per candidate,
        # not once per (candidate, query-term) pair.
        doc_lengths: dict[int, int] = {}
        for doc_id in candidates:
            meta = get_doc_meta(conn, doc_id)
            doc_lengths[doc_id] = meta[2] if meta else avgdl

        scored = []
        for doc_id in candidates:
            length = doc_lengths[doc_id]
            score = 0.0
            for ts in term_stats:
                tf = ts.doc_tf.get(doc_id, 0)
                if tf == 0:
                    continue
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * length / avgdl)
                score += bm25_idf(n_docs, ts.df) * (numerator / denominator)
            scored.append((doc_id, score))

        return fetch_results(conn, scored, top_k)


def bm25_search(raw_query: str, top_k: int = 10, k1: float = K1, b: float = B) -> list[dict]:
    query_terms = preprocess_query(raw_query)
    return bm25_score_terms(query_terms, top_k, k1, b)
