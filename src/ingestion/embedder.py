"""Embedding generation for chunks and entities."""

import logging
from typing import Optional

from langchain_openai import OpenAIEmbeddings

from src.config import settings

logger = logging.getLogger(__name__)

_embeddings_client: Optional[OpenAIEmbeddings] = None


def get_embeddings_client() -> OpenAIEmbeddings:
    """Get or create the OpenAI embeddings client (singleton)."""
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            openai_api_key=settings.openai_api_key,
        )
        logger.info(
            "Embeddings client created (model=%s, dims=%d)",
            settings.embedding_model,
            settings.embedding_dimensions,
        )
    return _embeddings_client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors.
    """
    if not texts:
        return []

    client = get_embeddings_client()
    embeddings = await client.aembed_documents(texts)
    logger.info("Embedded %d texts", len(texts))
    return embeddings


async def embed_query(query: str) -> list[float]:
    """Generate an embedding for a single query string.

    Args:
        query: The query text to embed.

    Returns:
        Embedding vector.
    """
    client = get_embeddings_client()
    return await client.aembed_query(query)
