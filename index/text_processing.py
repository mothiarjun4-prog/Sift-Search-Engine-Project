"""
Turns raw crawled HTML into a clean list of index-ready tokens.

Pipeline: HTML -> plain text -> lowercase word tokens -> drop stopwords ->
stem. Every step is deliberately dependency-light (no downloaded NLTK
corpora) so indexing works fully offline and reproducibly.
"""

import re

from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

# A standard, hand-maintained English stopword list. Hardcoded (rather than
# nltk.corpus.stopwords) so this module needs no network access or
# nltk.download() step to run.
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_stemmer = PorterStemmer()

# Porter stemming handles INFLECTION (crawl/crawls/crawling -> crawl) but
# deliberately does not touch DERIVATION (crawl -> crawler is a different,
# related word, not a tense/plural of the same word) — this is true of
# lemmatizers too, not just Porter, since neither is designed to collapse
# derived words. Left alone, "crawler" and "crawl" end up as two disconnected
# index terms, which can hurt retrieval when a query uses one form and the
# indexed page mostly uses the other (discovered via real ranking output,
# see rank/ tests). We patch just this specific, common pattern — an agent
# noun ending in "-er" for a verb that's also meaningful in this domain —
# with an explicit mapping, rather than a blanket suffix-strip rule (which
# risks false merges like "water" -> "wat", "corner" -> "corn").
#
# Each mapping's value is computed by stemming the verb form itself, so the
# agent noun is guaranteed to collapse to the SAME index term the verb
# produces, not a hand-guessed approximation.
_AGENT_NOUN_TO_VERB = {
    "crawler": "crawl",
    "browser": "browse",
    "server": "serve",
    "parser": "parse",
    "builder": "build",
    "ranker": "rank",
    "downloader": "download",
    "indexer": "index",
    "cacher": "cache",
    "scraper": "scrape",
    "requester": "request",
    "user": "use",
}
_AGENT_NOUN_STEM_MAP = {
    agent_noun: _stemmer.stem(verb) for agent_noun, verb in _AGENT_NOUN_TO_VERB.items()
}

# Tags whose text content is not real page content (nav chrome, scripts,
# styling) and should be stripped before extracting text.
_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript"]

# MediaWiki (Wikipedia) wraps page chrome in <div>s rather than semantic
# <nav>/<footer> tags, so _NOISE_TAGS alone doesn't catch it. These class/id
# names are boilerplate repeated identically on every article: the sidebar
# ("Jump to content", "Add links", tools menu), inline [edit] links, the
# category footer, and the "Retrieved from ..." print footer. Left in, they
# get indexed identically for all 200 pages and pollute term stats (though
# proper IDF weighting would eventually discount them anyway).
_WIKI_NOISE_SELECTORS = [
    "mw-jump-link",     # "Jump to content" skip-nav link
    "mw-editsection",   # inline [edit] links next to headings
    "catlinks",         # "Categories: ..." footer block
    "printfooter",      # "Retrieved from https://..." line
    "navbox",           # bottom-of-article nav boxes (not the article itself)
]

# The actual article body in MediaWiki always lives inside this container.
# Restricting extraction to it — when present — sidesteps the sidebar,
# header, and personal-tools chrome entirely, rather than trying to
# enumerate every noise element individually.
_WIKI_CONTENT_ID = "mw-content-text"


def html_to_text(html: str) -> str:
    """Strip tags/scripts/styles and return visible page text.

    If the page looks like a MediaWiki article (has #mw-content-text), we
    scope extraction to just that container — the real article text — so
    Wikipedia's sidebar/header/footer chrome never enters the index at all.
    Otherwise we fall back to whole-page extraction with generic noise-tag
    stripping, so this still works if you point the crawler at other sites.
    """
    soup = BeautifulSoup(html, "lxml")

    content = soup.find(id=_WIKI_CONTENT_ID)
    scope = content if content is not None else soup

    for tag in scope(_NOISE_TAGS):
        tag.decompose()
    for class_name in _WIKI_NOISE_SELECTORS:
        for tag in scope.find_all(class_=class_name):
            tag.decompose()

    return scope.get_text(separator=" ")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word/number tokens. Punctuation is dropped
    entirely (a search engine matching on exact punctuation is rarely what
    you want, and it keeps the vocabulary much smaller)."""
    return _TOKEN_RE.findall(text.lower())


def preprocess(text: str) -> list[str]:
    """Full pipeline: tokenize -> drop stopwords and 1-char noise -> stem ->
    collapse known agent-noun/verb pairs.

    Stemming ('running', 'runs', 'ran' -> mostly 'run'/'ran') means a search
    for "run" also matches documents that only say "running" — this is what
    real search engines do and it's a meaningful chunk of relevance quality.
    """
    tokens = tokenize(text)
    stems = [
        _stemmer.stem(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 1
    ]
    return [_AGENT_NOUN_STEM_MAP.get(stem, stem) for stem in stems]


def preprocess_query(text: str) -> list[str]:
    """Same pipeline, used at query time so query terms are stemmed the same
    way as indexed terms (a search for "running" must reduce to the same
    stem, "run", as the indexed documents)."""
    return preprocess(text)
