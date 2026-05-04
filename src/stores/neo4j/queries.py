"""Neo4j retrieval query operations (vector, fulltext, graph, hybrid)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from src.config import settings

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


def _engagement_scope(engagement_id: str) -> list[str]:
    """Build the engagement scope list for retrieval queries.

    Always includes both the client's engagement_id and the global
    engagement_id so that external/public documents are searched
    alongside client-specific documents.
    """
    scope = [engagement_id]
    if engagement_id != settings.global_engagement_id:
        scope.append(settings.global_engagement_id)
    return scope


def _to_utc_zoned_iso(dt: datetime) -> str:
    """Format *dt* as ISO-8601 with explicit UTC offset for Neo4j `datetime($s)`.

    Chunk `created_at` is written with Cypher `datetime()` (zoned instant). Naive
    Python datetimes and `isoformat()` strings without offset make Neo4j parse
    `datetime($s)` as LOCAL DATETIME, which is not comparable to zoned properties.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    iso = dt.isoformat(timespec="microseconds")
    if iso.endswith("+00:00"):
        return iso[:-6] + "Z"
    if iso.endswith("-00:00"):
        return iso[:-6] + "Z"
    return iso


def _date_range_clause(
    node_alias: str,
    start: Optional[datetime],
    end: Optional[datetime],
    params: dict[str, Any],
) -> str:
    """Build Cypher WHERE fragments for a date-range filter on created_at.

    Mutates *params* to inject $dr_start / $dr_end as zoned ISO strings (UTC `Z`).
    Returns an AND-prefixed clause (empty string when no filter).
    """
    parts: list[str] = []
    if start is not None:
        params["dr_start"] = _to_utc_zoned_iso(start)
        parts.append(f"{node_alias}.created_at >= datetime($dr_start)")
    if end is not None:
        params["dr_end"] = _to_utc_zoned_iso(end)
        parts.append(f"{node_alias}.created_at <= datetime($dr_end)")
    if not parts:
        return ""
    return " AND " + " AND ".join(parts)


# ------------------------------------------------------------------
# Vector search
# ------------------------------------------------------------------


async def vector_search(
    driver: AsyncDriver,
    query_embedding: list[float],
    engagement_id: str,
    top_k: int = 10,
    doc_types: Optional[list[str]] = None,
    date_start: Optional[datetime] = None,
    date_end: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Vector similarity search over chunk embeddings."""
    params: dict[str, Any] = {
        "query_embedding": query_embedding,
        "engagement_ids": _engagement_scope(engagement_id),
        "top_k": top_k,
    }
    where_clause = "WHERE chunk.engagement_id IN $engagement_ids"
    if doc_types:
        where_clause += " AND chunk.doc_type IN $doc_types"
        params["doc_types"] = doc_types
    where_clause += _date_range_clause("chunk", date_start, date_end, params)

    query = f"""
    CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
    YIELD node AS chunk, score
    {where_clause}
    OPTIONAL MATCH (chunk)-[:BELONGS_TO]->(doc:Document)
    RETURN chunk, score, doc
    ORDER BY score DESC
    LIMIT $top_k
    """

    async with driver.session() as session:
        result = await session.run(query, **params)
        return [record.data() async for record in result]


# ------------------------------------------------------------------
# Fulltext search
# ------------------------------------------------------------------


async def fulltext_search(
    driver: AsyncDriver,
    query_text: str,
    engagement_id: str,
    top_k: int = 10,
    doc_types: Optional[list[str]] = None,
    date_start: Optional[datetime] = None,
    date_end: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Fulltext (keyword) search over chunk content."""
    params: dict[str, Any] = {
        "query_text": query_text,
        "engagement_ids": _engagement_scope(engagement_id),
        "top_k": top_k,
    }
    where_clause = "WHERE chunk.engagement_id IN $engagement_ids"
    if doc_types:
        where_clause += " AND chunk.doc_type IN $doc_types"
        params["doc_types"] = doc_types
    where_clause += _date_range_clause("chunk", date_start, date_end, params)

    query = f"""
    CALL db.index.fulltext.queryNodes('chunk_fulltext', $query_text)
    YIELD node AS chunk, score
    {where_clause}
    OPTIONAL MATCH (chunk)-[:BELONGS_TO]->(doc:Document)
    RETURN chunk, score, doc
    ORDER BY score DESC
    LIMIT $top_k
    """

    async with driver.session() as session:
        result = await session.run(query, **params)
        return [record.data() async for record in result]


# ------------------------------------------------------------------
# Graph traversal
# ------------------------------------------------------------------


async def graph_search(
    driver: AsyncDriver,
    search_term: str,
    engagement_id: str,
    depth: int = 2,
    top_k: int = 10,
    date_start: Optional[datetime] = None,
    date_end: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Graph traversal: find entities and related chunks."""
    params: dict[str, Any] = {
        "search_term": search_term,
        "engagement_ids": _engagement_scope(engagement_id),
        "top_k": top_k,
    }
    chunk_date_filter = _date_range_clause("chunk", date_start, date_end, params)
    # Only constrain OPTIONAL MATCH when filtering by chunk index time: preserve entities
    # with no MENTIONED_IN chunks (chunk is null). Unconditional "chunk IS NOT NULL"
    # would drop those entities entirely.
    if chunk_date_filter:
        chunk_where = f"WHERE chunk IS NULL OR (chunk IS NOT NULL{chunk_date_filter})"
    else:
        chunk_where = ""

    query = f"""
    CALL db.index.fulltext.queryNodes('entity_fulltext', $search_term)
    YIELD node AS entity, score
    WHERE entity.engagement_id IN $engagement_ids
    WITH entity, score
    ORDER BY score DESC
    LIMIT $top_k
    OPTIONAL MATCH (entity)-[:MENTIONED_IN]->(chunk:Chunk)
    {chunk_where}
    OPTIONAL MATCH (chunk)-[:BELONGS_TO]->(doc:Document)
    OPTIONAL MATCH (entity)-[rel:RELATES_TO]-(related:Entity)
    RETURN entity, score,
           collect(DISTINCT {{chunk: chunk, doc: doc}}) AS chunk_docs,
           collect(DISTINCT {{entity: related, relationship: type(rel)}}) AS related_entities
    """
    async with driver.session() as session:
        result = await session.run(query, **params)
        return [record.data() async for record in result]


# ------------------------------------------------------------------
# Entity vector search
# ------------------------------------------------------------------


async def entity_vector_search(
    driver: AsyncDriver,
    query_embedding: list[float],
    engagement_id: str,
    top_k: int = 10,
    entity_types: Optional[list[str]] = None,
    date_start: Optional[datetime] = None,
    date_end: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """KNN search over entity embeddings, expand to linked chunks."""
    params: dict[str, Any] = {
        "query_embedding": query_embedding,
        "engagement_ids": _engagement_scope(engagement_id),
        "top_k": top_k,
    }
    where_clause = "WHERE entity.engagement_id IN $engagement_ids"
    if entity_types:
        where_clause += " AND entity.type IN $entity_types"
        params["entity_types"] = entity_types

    chunk_date_filter = _date_range_clause("chunk", date_start, date_end, params)
    if chunk_date_filter:
        chunk_where = f"WHERE chunk IS NULL OR (chunk IS NOT NULL{chunk_date_filter})"
    else:
        chunk_where = ""

    query = f"""
    CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $query_embedding)
    YIELD node AS entity, score
    {where_clause}
    WITH entity, score
    ORDER BY score DESC
    LIMIT $top_k
    OPTIONAL MATCH (entity)-[:MENTIONED_IN]->(chunk:Chunk)
    {chunk_where}
    OPTIONAL MATCH (chunk)-[:BELONGS_TO]->(doc:Document)
    OPTIONAL MATCH (entity)-[rel:RELATES_TO]-(related:Entity)
    RETURN entity, score,
           collect(DISTINCT {{chunk: chunk, doc: doc}}) AS chunk_docs,
           collect(DISTINCT {{entity: related, relationship: type(rel)}}) AS related_entities
    """

    async with driver.session() as session:
        result = await session.run(query, **params)
        return [record.data() async for record in result]


# ------------------------------------------------------------------
# Hybrid (V + K + G)
# ------------------------------------------------------------------


async def hybrid_search(
    driver: AsyncDriver,
    query_embedding: list[float],
    query_text: str,
    engagement_id: str,
    top_k: int = 10,
    doc_types: Optional[list[str]] = None,
    date_start: Optional[datetime] = None,
    date_end: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Full hybrid retrieval: vector + fulltext + graph expansion."""
    params: dict[str, Any] = {
        "query_embedding": query_embedding,
        "engagement_ids": _engagement_scope(engagement_id),
        "top_k": top_k,
    }
    where_clause = "WHERE chunk.engagement_id IN $engagement_ids"
    if doc_types:
        where_clause += " AND chunk.doc_type IN $doc_types"
        params["doc_types"] = doc_types
    where_clause += _date_range_clause("chunk", date_start, date_end, params)

    query = f"""
    CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
    YIELD node AS chunk, score
    {where_clause}
    OPTIONAL MATCH (chunk)-[:BELONGS_TO]->(doc:Document)
    OPTIONAL MATCH (entity:Entity)-[:MENTIONED_IN]->(chunk)
    OPTIONAL MATCH (entity)-[rel:RELATES_TO]-(related:Entity)
    RETURN chunk, score, doc,
           collect(DISTINCT entity) AS entities,
           collect(DISTINCT {{entity: related, relationship: type(rel)}}) AS related_entities
    ORDER BY score DESC
    LIMIT $top_k
    """

    async with driver.session() as session:
        result = await session.run(query, **params)
        return [record.data() async for record in result]


# ------------------------------------------------------------------
# Graph + Vector + Fulltext (full hybrid)
# ------------------------------------------------------------------


async def graph_vector_fulltext_search(
    driver: AsyncDriver,
    query_embedding: list[float],
    query_text: str,
    engagement_id: str,
    top_k: int = 10,
    doc_types: Optional[list[str]] = None,
    date_start: Optional[datetime] = None,
    date_end: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Full hybrid: vector KNN + fulltext hits -> graph expansion.

    Combines vector similarity, keyword matching, and graph traversal
    into a single merged result set.
    """
    params: dict[str, Any] = {
        "query_embedding": query_embedding,
        "query_text": query_text,
        "engagement_ids": _engagement_scope(engagement_id),
        "top_k": top_k,
    }
    where_filter = ""
    if doc_types:
        where_filter += " AND chunk.doc_type IN $doc_types"
        params["doc_types"] = doc_types
    where_filter += _date_range_clause("chunk", date_start, date_end, params)
    ft_where_filter = where_filter.replace("chunk.", "ft_chunk.")

    query = f"""
    // Vector KNN hits
    CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
    YIELD node AS chunk, score
    WHERE chunk.engagement_id IN $engagement_ids {where_filter}
    WITH collect({{chunk: chunk, score: score}}) AS vector_hits

    // Fulltext hits
    CALL db.index.fulltext.queryNodes('chunk_fulltext', $query_text)
    YIELD node AS ft_chunk, score AS ft_score
    WHERE ft_chunk.engagement_id IN $engagement_ids {ft_where_filter}
    WITH vector_hits, collect({{chunk: ft_chunk, score: ft_score * 0.8}}) AS ft_hits

    // Merge and deduplicate
    WITH vector_hits + ft_hits AS all_hits
    UNWIND all_hits AS hit
    WITH hit.chunk AS chunk, max(hit.score) AS score
    ORDER BY score DESC
    LIMIT $top_k

    // Graph expansion
    OPTIONAL MATCH (chunk)-[:BELONGS_TO]->(doc:Document)
    OPTIONAL MATCH (entity:Entity)-[:MENTIONED_IN]->(chunk)
    OPTIONAL MATCH (entity)-[rel:RELATES_TO]-(related:Entity)
    RETURN chunk, score, doc,
           collect(DISTINCT entity) AS entities,
           collect(DISTINCT {{entity: related, relationship: type(rel)}}) AS related_entities
    ORDER BY score DESC
    """

    async with driver.session() as session:
        result = await session.run(query, **params)
        return [record.data() async for record in result]
