"""
Checks whether a sequence of terms appears consecutively (as a phrase) in a
specific document, using the token positions stored in the inverted index.
"""

from index.inverted_index import get_term_positions_for_doc


def phrase_present_in_doc(conn, phrase_terms: list[str], doc_id: int) -> bool:
    """True if phrase_terms appear at consecutive positions in doc_id.

    Approach: take the position list of the FIRST term as candidate phrase
    start points. For each candidate start position p, check that the 2nd
    term appears at position p+1, the 3rd at p+2, and so on. If any start
    position satisfies the whole chain, the phrase is present.
    """
    if not phrase_terms:
        return True  # an empty phrase constrains nothing

    first_term_positions = get_term_positions_for_doc(conn, phrase_terms[0], doc_id)
    if not first_term_positions:
        return False

    # Precompute position sets for the remaining terms once, rather than
    # re-querying the DB inside the candidate-start loop below.
    later_position_sets = [
        set(get_term_positions_for_doc(conn, term, doc_id))
        for term in phrase_terms[1:]
    ]

    for start in first_term_positions:
        if all(
            (start + offset) in later_position_sets[offset - 1]
            for offset in range(1, len(phrase_terms))
        ):
            return True
    return False


def filter_by_phrases(conn, doc_ids: set[int], phrases: list[list[str]]) -> set[int]:
    """Keep only docs that contain EVERY given phrase. If that empties the
    result entirely, the caller should decide whether to fall back to the
    unfiltered set (see query/search_engine.py) rather than show nothing."""
    if not phrases:
        return doc_ids
    return {
        doc_id
        for doc_id in doc_ids
        if all(phrase_present_in_doc(conn, phrase, doc_id) for phrase in phrases)
    }
