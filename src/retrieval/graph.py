"""Graph traversal retrieval via Neo4j entity relationships."""

import logging
import time

from src.models.retrieval import RelatedEntity, RetrievedChunk
from src.stores.neo4j_store import neo4j_store

logger = logging.getLogger(__name__)


async def graph_search(
    query: str,
    engagement_id: str,
    top_k: int = 10,
    depth: int = 2,
) -> tuple[list[RetrievedChunk], float]:
    """Perform graph traversal search: find entities and their linked chunks.

    Args:
        query: The user query text (used for entity fulltext search).
        engagement_id: Engagement scope filter.
        top_k: Number of entity results.
        depth: Relationship traversal depth.

    Returns:
        Tuple of (list of RetrievedChunk, latency_ms).
    """
    start = time.time()

    records = await neo4j_store.graph_search(
        search_term=query,
        engagement_id=engagement_id,
        top_k=top_k,
        depth=depth,
    )

    chunks = _parse_graph_records(records)
    latency_ms = (time.time() - start) * 1000

    logger.info("Graph search: %d results in %.1fms", len(chunks), latency_ms)
    return chunks, latency_ms


def _parse_graph_records(records: list[dict]) -> list[RetrievedChunk]:
    """Parse Neo4j graph search records into RetrievedChunk models."""
    chunks = []
    seen_chunk_ids: set[str] = set()

    for record in records:
        entity_node = record.get("entity", {})
        chunk_docs = record.get("chunk_docs", [])
        related_raw = record.get("related_entities", [])
        score = record.get("score", 0.0)

        # Build related entities list
        related_entities = []
        for rel in related_raw:
            rel_entity = rel.get("entity") or {}
            if rel_entity:
                related_entities.append(RelatedEntity(
                    name=rel_entity.get("name", ""),
                    type=rel_entity.get("type", ""),
                    relationship=rel.get("relationship", ""),
                ))

        # Build entity info
        entity_info = RelatedEntity(
            name=entity_node.get("name", ""),
            type=entity_node.get("type", ""),
        ) if entity_node else None

        # Extract chunks from entity's MENTIONED_IN relationships
        for cd in chunk_docs:
            chunk_node = cd.get("chunk") or {}
            doc_node = cd.get("doc") or {}

            if not chunk_node:
                continue

            chunk_id = chunk_node.get("id", "")
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            entities = [entity_info] if entity_info else []

            chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                content=chunk_node.get("content", ""),
                score=score,
                document_id=chunk_node.get("document_id", ""),
                document_name=doc_node.get("filename", "") if doc_node else "",
                page_number=chunk_node.get("page_number"),
                section=chunk_node.get("section"),
                doc_type=chunk_node.get("doc_type", ""),
                entities=entities,
                related_entities=related_entities,
            ))

    return chunks
