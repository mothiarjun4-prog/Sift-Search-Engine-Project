"""
Minimal web UI for the search engine.

Run: python -m query.app
Then open http://127.0.0.1:5000
"""

import re
import time

from flask import Flask, render_template, request

from query.search_engine import search
from query.snippet import get_snippet

app = Flask(__name__)

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


@app.route("/")
def index():
    raw_query = request.args.get("q", "").strip()
    method = request.args.get("method", "bm25")

    if not raw_query:
        return render_template("index.html", query="", results=[], method=method)

    start = time.perf_counter()
    outcome = search(raw_query, method=method, top_k=10)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Snippets are generated against the RAW words the user typed (not the
    # stemmed index terms), since a stemmed form like "crawl" wouldn't
    # literally appear in the page text to search for.
    raw_words = _WORD_RE.findall(raw_query.lower())

    results = []
    for r in outcome["results"]:
        results.append({
            **r,
            "snippet": get_snippet(r["doc_id"], raw_words),
        })

    return render_template(
        "index.html",
        query=raw_query,
        method=method,
        results=results,
        corrections=outcome["corrections"],
        phrase_fallback=outcome["phrase_fallback"],
        elapsed_ms=round(elapsed_ms, 1),
    )


if __name__ == "__main__":
    app.run(debug=True)
