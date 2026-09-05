"""
Shared plumbing used by both TF-IDF and BM25: given a preprocessed query,
gather the candidate documents (union of docs containing ANY query term —
unlike Week 2's boolean AND, ranking doesn't require every term to match)
and the per-term statistics each scoring formula needs.
"""

from dataclasses import dataclass

from index.inverted_index import get_conn, get_postings, get_doc_meta, document_frequency, total_docs


@dataclass
class TermStats:
    term: str
    df: int  # document frequency: how many docs contain this term at all
    # doc_id -> term frequency in that doc
    doc_tf: dict[int, int]


def gather_term_stats(conn, query_terms: list[str]) -> list[TermStats]:
    """One TermStats per (deduplicated) query term."""
    stats = []
    seen = set()
    for term in query_terms:
        if term in seen:
            continue
        seen.add(term)
        postings = get_postings(conn, term)
        doc_tf = {doc_id: tf for doc_id, tf, _positions in postings}
        stats.append(TermStats(term=term, df=len(doc_tf), doc_tf=doc_tf))
    return stats


def candidate_doc_ids(term_stats: list[TermStats]) -> set[int]:
    """Union, not intersection: a doc matching only 2 of 3 query terms can
    still be worth showing, just ranked lower. This is the key difference
    from Week 2's lookup.py, which required ALL terms (boolean AND)."""
    ids: set[int] = set()
    for ts in term_stats:
        ids |= ts.doc_tf.keys()
    return ids


def fetch_results(conn, scored: list[tuple[int, float]], top_k: int) -> list[dict]:
    """Turn [(doc_id, score), ...] into result dicts with url/title attached,
    sorted by score descending."""
    scored.sort(key=lambda pair: pair[1], reverse=True)
    results = []
    for doc_id, score in scored[:top_k]:
        meta = get_doc_meta(conn, doc_id)
        if meta is None:
            continue
        url, title, length = meta
        results.append({"doc_id": doc_id, "score": score, "url": url, "title": title})
    return results
