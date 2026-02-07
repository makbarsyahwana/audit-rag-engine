"""Result reranking (optional). Simple score-based reranker for MVP."""

import logging

from src.models.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


def rerank(
    chunks: list[RetrievedChunk],
    query: str,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Rerank retrieved chunks by score (descending).

    For MVP this is a simple sort. Can be replaced with a cross-encoder
    reranker (e.g., Cohere Rerank, ColBERT) in later phases.

    Args:
        chunks: List of retrieved chunks to rerank.
        query: Original query (unused in score-based rerank, kept for API).
        top_k: Number of top results to return after reranking.

    Returns:
        Reranked and truncated list of chunks.
    """
    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
    result = sorted_chunks[:top_k]
    logger.info("Reranked %d chunks → top %d", len(chunks), len(result))
    return result
