"""
Run a query through both ranking methods and print results side by side.

Run: python -m rank.search web crawler
     python -m rank.search --method bm25 pagerank algorithm
     python -m rank.search --method bm25 --b 0.3 --k1 1.2 crawling search
"""

import sys

from rank.tfidf import tfidf_search
from rank.bm25 import bm25_search, K1 as DEFAULT_K1, B as DEFAULT_B


def print_results(label: str, results: list[dict]):
    print(f"\n--- {label} ---")
    if not results:
        print("  (no matches)")
        return
    for rank, r in enumerate(results, start=1):
        print(f"  {rank}. [{r['score']:.4f}] {r['title']}  ({r['url']})")


if __name__ == "__main__":
    args = sys.argv[1:]
    method = None
    k1 = DEFAULT_K1
    b = DEFAULT_B

    # Simple manual flag parsing (argparse would be overkill for 3 flags).
    i = 0
    remaining = []
    while i < len(args):
        if args[i] == "--method":
            method = args[i + 1]
            i += 2
        elif args[i] == "--k1":
            k1 = float(args[i + 1])
            i += 2
        elif args[i] == "--b":
            b = float(args[i + 1])
            i += 2
        else:
            remaining.append(args[i])
            i += 1
    args = remaining

    if not args:
        print("Usage: python -m rank.search [--method tfidf|bm25] [--k1 N] [--b N] <query words>")
        sys.exit(1)

    query = " ".join(args)
    print(f"Query: {query!r}")
    if method in (None, "bm25"):
        print(f"(BM25 params: k1={k1}, b={b})")

    if method in (None, "tfidf"):
        print_results("TF-IDF", tfidf_search(query))
    if method in (None, "bm25"):
        print_results("BM25", bm25_search(query, k1=k1, b=b))
