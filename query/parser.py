"""
Parses a raw query string into:
  - phrases: quoted "exact phrase" segments, each as an ordered list of
    stemmed terms (for positional phrase matching)
  - free_terms: everything else, as a flat list of stemmed terms (for
    normal ranking, order doesn't matter)

Also applies typo correction to every term (free or inside a phrase) that
isn't in the index vocabulary.

IMPORTANT LIMITATION: because Week 2's indexing pipeline strips stopwords
before storing positions, a phrase like "to be or not to be" has almost
nothing left to match on — the stopwords were never indexed at all. Phrase
search here works on the sequence of CONTENT words only. In practice this
means "web crawler" and "the web crawler" and "a web, crawler" all match
identically, since stopwords/punctuation don't affect the position
sequence either at index time or query time. This is a real, explainable
tradeoff of prioritizing a smaller/cleaner index over exact literal phrase
matching — most lightweight search implementations make the same call.
"""

import re

from index.text_processing import preprocess
from query.fuzzy import correct_terms

_PHRASE_RE = re.compile(r'"([^"]+)"')


def parse_query(raw_query: str) -> dict:
    phrase_texts = _PHRASE_RE.findall(raw_query)
    free_text = _PHRASE_RE.sub(" ", raw_query)  # remove quoted parts, keep the rest

    phrases = [preprocess(p) for p in phrase_texts]
    phrases = [p for p in phrases if p]  # drop phrases that became empty (all-stopword)

    free_terms = preprocess(free_text)

    # Typo-correct everything together in one vocabulary lookup, then split
    # back into phrases vs free terms using the lengths we started with.
    all_terms = free_terms + [term for phrase in phrases for term in phrase]
    corrected_all, corrections = correct_terms(all_terms)

    corrected_free = corrected_all[: len(free_terms)]
    rest = corrected_all[len(free_terms):]
    corrected_phrases = []
    for phrase in phrases:
        corrected_phrases.append(rest[: len(phrase)])
        rest = rest[len(phrase):]

    return {
        "free_terms": corrected_free,
        "phrases": corrected_phrases,
        "corrections": corrections,  # original -> corrected, for "did you mean"
    }
