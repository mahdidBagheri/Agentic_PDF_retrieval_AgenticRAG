from typing import List


def confident_enough(
    contexts: List[dict],
    min_score: float = 0.35,
    min_chunks: int = 1,
) -> bool:
    """
    Decide whether retrieved contexts are strong enough to answer.

    contexts: list of retrieved chunks
        Each chunk is expected to contain:
        - 'score' (semantic similarity)
        - or 'rerank_score' if reranking is used

    min_score:
        Minimum similarity score for top chunk

    min_chunks:
        Minimum number of retrieved chunks required
    """

    if not contexts:
        return False

    if len(contexts) < min_chunks:
        return False

    top = contexts[0]

    score = top.get("rerank_score", top.get("score", 0.0))

    return score >= min_score