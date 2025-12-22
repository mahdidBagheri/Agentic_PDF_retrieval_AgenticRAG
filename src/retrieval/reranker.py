import math
import re


def tokenize(text: str):
    """
    Simple tokenizer:
    - lowercase
    - remove punctuation
    - split on whitespace
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def keyword_score(query: str, text: str) -> float:
    """
    Measures lexical overlap between query and chunk text.
    """
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(text))

    if not query_terms:
        return 0.0

    overlap = query_terms & text_terms

    # Normalized overlap score
    return len(overlap) / math.sqrt(len(query_terms))


def rerank(query: str, results: list, alpha: float = 0.75):
    """
    Combines semantic + keyword scores.
    alpha controls semantic weight.
    """
    reranked = []

    for r in results:
        kw = keyword_score(query, r["text"])

        r["keyword_score"] = kw
        r["rerank_score"] = alpha * r["score"] + (1 - alpha) * kw

        reranked.append(r)

    return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)