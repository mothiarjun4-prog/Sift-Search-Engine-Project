"""
Typo tolerance: if a query word isn't in the index vocabulary at all, find
the closest vocabulary word by edit distance instead of just returning zero
results for a simple misspelling ("crawlr" -> "crawl").
"""

from index.inverted_index import get_conn, get_vocabulary

# Cached, process-lifetime vocabulary structure. Rebuilding this from SQL on
# every single query (as the first version of this module did) is fine at a
# few hundred documents, but becomes a real cost once the corpus — and
# therefore the vocabulary — grows into the thousands of terms, since it's a
# full-table read PLUS a full linear scan per uncorrected query word. The
# index only changes when build_index.py re-runs, not while the app is
# serving queries, so caching for the app's lifetime is safe.
_vocab_set: set[str] | None = None
_vocab_by_length: dict[int, list[str]] | None = None


def _ensure_vocab_loaded(force_refresh: bool = False):
    global _vocab_set, _vocab_by_length
    if _vocab_set is not None and not force_refresh:
        return

    with get_conn() as conn:
        terms = get_vocabulary(conn)

    _vocab_set = set(terms)

    # Bucket terms by length so correct_term() only has to scan the handful
    # of buckets within max_distance of the query word's length, instead of
    # every term in the vocabulary. Edit distance can never be smaller than
    # the length difference between two strings, so this loses no correct
    # matches — it's a lower-bound pruning trick, not an approximation.
    by_length: dict[int, list[str]] = {}
    for term in terms:
        by_length.setdefault(len(term), []).append(term)
    _vocab_by_length = by_length


def levenshtein(a: str, b: str) -> int:
    """Classic dynamic-programming edit distance: the minimum number of
    single-character insertions, deletions, or substitutions to turn `a`
    into `b`. dp[i][j] = edit distance between a[:i] and b[:j].
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    # Only need the previous row to compute the current one — O(min(n,m))
    # space instead of the full O(n*m) table, since we never look back
    # further than one row.
    prev_row = list(range(m + 1))
    for i in range(1, n + 1):
        curr_row = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,      # deletion
                curr_row[j - 1] + 1,  # insertion
                prev_row[j - 1] + cost,  # substitution (or match, cost=0)
            )
        prev_row = curr_row
    return prev_row[m]


def _max_allowed_distance(term_length: int) -> int:
    """Scale how many edits we'll tolerate based on word length. A flat
    max_distance=2 for every word is too permissive for short words — 2
    edits on a 5-letter word changes 40% of it (which is how 'tfidf' was
    getting corrected to the unrelated word 'idf' during testing) — but
    appropriately strict for longer ones. This mirrors the length-scaled
    fuzziness real search engines (e.g. Elasticsearch) use."""
    if term_length <= 3:
        return 0  # too short to safely correct at all
    elif term_length <= 5:
        return 1
    else:
        return 2


def correct_term(term: str, max_distance: int | None = None) -> str | None:
    """Return the closest vocabulary word to `term` within its allowed edit
    distance, or None if term is already valid or nothing is close enough.

    Only scans vocabulary buckets whose length is within max_distance of
    the query word's length — edit distance can never be smaller than the
    length difference, so terms outside that range can never qualify
    anyway. This keeps typo correction fast even as the vocabulary grows
    into the tens of thousands of terms.
    """
    _ensure_vocab_loaded()

    if term in _vocab_set:
        return None  # not a typo, no correction needed

    if max_distance is None:
        max_distance = _max_allowed_distance(len(term))
    if max_distance == 0:
        return None

    best_match = None
    best_distance = max_distance + 1

    for length in range(len(term) - max_distance, len(term) + max_distance + 1):
        for candidate in _vocab_by_length.get(length, []):
            distance = levenshtein(term, candidate)
            if distance < best_distance:
                best_distance = distance
                best_match = candidate

    return best_match if best_distance <= max_distance else None


def correct_terms(terms: list[str]) -> tuple[list[str], dict[str, str]]:
    """Correct every term in a list against the current index vocabulary.
    Returns (corrected_terms, corrections) where corrections maps
    original -> corrected for any word that was actually changed, so the
    caller can show the user a "did you mean" note.
    """
    corrected = []
    corrections = {}
    for term in terms:
        fix = correct_term(term)
        if fix is not None:
            corrections[term] = fix
            corrected.append(fix)
        else:
            corrected.append(term)
    return corrected, corrections
