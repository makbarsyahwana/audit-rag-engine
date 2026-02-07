"""Evidence search with control, period, and entity filters."""

import logging
import time
from typing import Optional

from src.ingestion.embedder import embed_query
from src.models.workflow import EvidenceChunk
from src.stores.neo4j_store import neo4j_store

logger = logging.getLogger(__name__)


async def evidence_search(
    query: str,
    engagement_id: str,
    top_k: int = 10,
    control_id: Optional[str] = None,
    period: Optional[str] = None,
    entity: Optional[str] = None,
    doc_types: Optional[list[str]] = None,
) -> tuple[list[EvidenceChunk], float]:
    """Search for evidence chunks with audit-specific filters.

    Performs hybrid vector+fulltext search then applies post-filters
    for control_id, period, and entity metadata.

    Args:
        query: Search query text.
        engagement_id: Engagement scope.
        top_k: Max results to return.
        control_id: Filter by control ID in entity graph.
        period: Filter by time period (matched against chunk metadata).
        entity: Filter by entity/organization name.
        doc_types: Filter by document types.

    Returns:
        Tuple of (evidence_chunks, latency_ms).
    """
    start = time.time()

    # Embed query for vector search
    query_embedding = await embed_query(query)

    # Build Cypher WHERE filters
    where_parts = ["chunk.engagement_id = $engagement_id"]
    params: dict = {
        "query_embedding": query_embedding,
        "engagement_id": engagement_id,
        "top_k": top_k * 3,  # over-fetch for post-filtering
        "query_text": query,
    }

    if doc_types:
        where_parts.append("chunk.doc_type IN $doc_types")
        params["doc_types"] = doc_types

    where_clause = " AND ".join(where_parts)

    # Hybrid: vector + optional entity expansion
    cypher = f"""
    CALL db.index.vector.queryNodes(
        'chunk_embeddings', $top_k, $query_embedding
    )
    YIELD node AS chunk, score
    WHERE {where_clause}
    OPTIONAL MATCH (chunk)-[:BELONGS_TO]->(doc:Document)
    OPTIONAL MATCH (ent:Entity)-[:MENTIONED_IN]->(chunk)
    RETURN chunk, score, doc,
           collect(DISTINCT ent) AS entities
    ORDER BY score DESC
    LIMIT $top_k
    """

    async with neo4j_store.driver.session() as session:
        result = await session.run(cypher, **params)
        records = [record.data() async for record in result]

    # Transform to EvidenceChunk and apply post-filters
    evidence_chunks: list[EvidenceChunk] = []

    for record in records:
        chunk_data = record.get("chunk", {})
        doc_data = record.get("doc", {}) or {}
        entities_data = record.get("entities", [])

        # Extract control IDs from associated entities
        chunk_control_ids = [
            e.get("id", "")
            for e in entities_data
            if e and e.get("type", "").lower() == "control"
        ]

        # Extract entity names for filtering
        entity_names = [
            e.get("name", "").lower()
            for e in entities_data
            if e
        ]

        # Post-filter: control_id
        if control_id and control_id not in chunk_control_ids:
            # Also check if control_id appears in chunk content
            content = chunk_data.get("content", "")
            if control_id.lower() not in content.lower():
                continue

        # Post-filter: entity
        if entity:
            entity_lower = entity.lower()
            content = chunk_data.get("content", "")
            if (
                entity_lower not in content.lower()
                and entity_lower not in entity_names
            ):
                continue

        # Post-filter: period
        if period:
            content = chunk_data.get("content", "")
            if period.lower() not in content.lower():
                continue

        evidence_chunks.append(
            EvidenceChunk(
                chunk_id=chunk_data.get("id", ""),
                content=chunk_data.get("content", ""),
                score=record.get("score", 0.0),
                document_id=chunk_data.get("document_id", ""),
                document_name=doc_data.get("filename", ""),
                page_number=chunk_data.get("page_number"),
                section=chunk_data.get("section"),
                doc_type=chunk_data.get("doc_type", ""),
                control_ids=chunk_control_ids,
                period=period,
                entity=entity,
            )
        )

        if len(evidence_chunks) >= top_k:
            break

    latency_ms = (time.time() - start) * 1000
    logger.info(
        "Evidence search: %d results (control=%s, period=%s, entity=%s) "
        "in %.0fms",
        len(evidence_chunks),
        control_id,
        period,
        entity,
        latency_ms,
    )
    return evidence_chunks, latency_ms
