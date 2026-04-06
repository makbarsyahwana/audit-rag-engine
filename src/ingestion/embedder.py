"""Embedding generation for chunks and entities."""

import logging
from typing import Optional

from langchain_core.embeddings import Embeddings

from src.config import settings
from src.generation.provider_factory import build_embeddings

logger = logging.getLogger(__name__)

_embeddings_client: Optional[Embeddings] = None


def get_embeddings_client() -> Embeddings:
    """Get or create the embeddings client (singleton).

    Provider is determined by ``EMBEDDING_PROVIDER`` (default: openai_compatible).
    API key fallback chain: EMBEDDING_API_KEY → OPENROUTER_API_KEY → OPENAI_API_KEY.
    """
    global _embeddings_client
    if _embeddings_client is None:
        _api_key = (
            settings.embedding_api_key
            or settings.openrouter_api_key
            or settings.openai_api_key
        )
        _embeddings_client = build_embeddings(
            provider=settings.embedding_provider,
            model=settings.embedding_model,
            base_url=settings.embedding_base_url or None,
            api_key=_api_key,
            dimensions=settings.embedding_dimensions,
        )
        logger.info(
            "Embeddings client created (provider=%s, model=%s, dims=%d, base_url=%s)",
            settings.embedding_provider,
            settings.embedding_model,
            settings.embedding_dimensions,
            settings.embedding_base_url or "default",
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
