"""Fulltext (keyword) search over Neo4j Lucene index."""

import logging
import time
from typing import Optional

from src.models.retrieval import DateRangeFilter, RetrievedChunk
from src.stores.neo4j_store import neo4j_store

logger = logging.getLogger(__name__)


async def fulltext_search(
    query: str,
    engagement_id: str,
    top_k: int = 10,
    doc_types: Optional[list[str]] = None,
    date_range: Optional[DateRangeFilter] = None,
) -> tuple[list[RetrievedChunk], float]:
    """Perform fulltext keyword search over chunk content.

    Args:
        query: The user query text (used as Lucene query).
        engagement_id: Engagement scope filter.
        top_k: Number of results to return.
        doc_types: Optional doc_type filter.

    Returns:
        Tuple of (list of RetrievedChunk, latency_ms).
    """
    start = time.time()

    records = await neo4j_store.fulltext_search(
        query_text=query,
        engagement_id=engagement_id,
        top_k=top_k,
        doc_types=doc_types,
        date_start=date_range.start if date_range else None,
        date_end=date_range.end if date_range else None,
    )

    chunks = _parse_chunk_records(records)
    latency_ms = (time.time() - start) * 1000

    logger.info("Fulltext search: %d results in %.1fms", len(chunks), latency_ms)
    return chunks, latency_ms


def _parse_chunk_records(records: list[dict]) -> list[RetrievedChunk]:
    """Parse Neo4j records into RetrievedChunk models."""
    chunks = []
    for record in records:
        chunk_node = record.get("chunk", {})
        doc_node = record.get("doc") or {}
        score = record.get("score", 0.0)

        if not chunk_node:
            continue

        chunks.append(RetrievedChunk(
            chunk_id=chunk_node.get("id", ""),
            content=chunk_node.get("content", ""),
            score=score,
            document_id=chunk_node.get("document_id", ""),
            document_name=doc_node.get("filename", "") if doc_node else "",
            page_number=chunk_node.get("page_number"),
            section=chunk_node.get("section"),
            doc_type=chunk_node.get("doc_type", ""),
        ))

    return chunks
