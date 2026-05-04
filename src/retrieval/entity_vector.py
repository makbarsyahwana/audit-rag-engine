"""Entity vector search over Neo4j entity embeddings (HNSW index).

KNN over entity embeddings → expand via MENTIONED_IN → return chunks.
Matches the llm-graph-builder `entity_vector` chat mode.
"""

import logging
import time
from typing import Optional

from src.ingestion.embedder import embed_query
from src.models.retrieval import DateRangeFilter, RelatedEntity, RetrievedChunk
from src.stores.neo4j_store import neo4j_store

logger = logging.getLogger(__name__)


async def entity_vector_search(
    query: str,
    engagement_id: str,
    top_k: int = 10,
    entity_types: Optional[list[str]] = None,
    date_range: Optional[DateRangeFilter] = None,
) -> tuple[list[RetrievedChunk], float]:
    """KNN search over entity embeddings, then expand to linked chunks.

    Args:
        query: The user query text.
        engagement_id: Engagement scope filter.
        top_k: Number of entity results.
        entity_types: Optional entity type filter.

    Returns:
        Tuple of (list of RetrievedChunk, latency_ms).
    """
    start = time.time()

    query_embedding = await embed_query(query)

    records = await neo4j_store.entity_vector_search(
        query_embedding=query_embedding,
        engagement_id=engagement_id,
        top_k=top_k,
        entity_types=entity_types,
        date_start=date_range.start if date_range else None,
        date_end=date_range.end if date_range else None,
    )

    chunks = _parse_entity_vector_records(records)
    latency_ms = (time.time() - start) * 1000

    logger.info("Entity vector search: %d results in %.1fms", len(chunks), latency_ms)
    return chunks, latency_ms


def _parse_entity_vector_records(records: list[dict]) -> list[RetrievedChunk]:
    """Parse Neo4j entity vector search records into RetrievedChunk models."""
    chunks = []
    seen_chunk_ids: set[str] = set()

    for record in records:
        entity_node = record.get("entity", {})
        score = record.get("score", 0.0)
        chunk_docs = record.get("chunk_docs", [])
        related_raw = record.get("related_entities", [])

        if not entity_node:
            continue

        entity_info = RelatedEntity(
            name=entity_node.get("name", ""),
            type=entity_node.get("type", ""),
        )

        related_entities = []
        for rel in related_raw:
            rel_ent = rel.get("entity") or {}
            if rel_ent:
                related_entities.append(RelatedEntity(
                    name=rel_ent.get("name", ""),
                    type=rel_ent.get("type", ""),
                    relationship=rel.get("relationship", ""),
                ))

        for cd in chunk_docs:
            chunk_node = cd.get("chunk") or {}
            doc_node = cd.get("doc") or {}

            if not chunk_node:
                continue

            chunk_id = chunk_node.get("id", "")
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                content=chunk_node.get("content", ""),
                score=score,
                document_id=chunk_node.get("document_id", ""),
                document_name=doc_node.get("filename", "") if doc_node else "",
                page_number=chunk_node.get("page_number"),
                section=chunk_node.get("section"),
                doc_type=chunk_node.get("doc_type", ""),
                entities=[entity_info],
                related_entities=related_entities,
            ))

    return chunks
