"""Neo4j write operations (chunks, documents, entities, relationships)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


async def upsert_chunk(driver: AsyncDriver, chunk: dict[str, Any]) -> None:
    """Upsert a Chunk node with embedding."""
    query = """
    MERGE (c:Chunk {id: $chunk_id})
    ON CREATE SET c.created_at = datetime()
    SET c.document_id = $document_id,
        c.engagement_id = $engagement_id,
        c.content = $content,
        c.content_preview = $content_preview,
        c.chunk_index = $chunk_index,
        c.page_number = $page_number,
        c.section = $section,
        c.doc_type = $doc_type,
        c.embedding = $embedding,
        c.updated_at = datetime()
    """
    async with driver.session() as session:
        await session.run(query, **chunk)


async def upsert_chunks_batch(driver: AsyncDriver, chunks: list[dict[str, Any]]) -> None:
    """Batch upsert Chunk nodes."""
    query = """
    UNWIND $chunks AS chunk
    MERGE (c:Chunk {id: chunk.chunk_id})
    ON CREATE SET c.created_at = datetime()
    SET c.document_id = chunk.document_id,
        c.engagement_id = chunk.engagement_id,
        c.content = chunk.content,
        c.content_preview = chunk.content_preview,
        c.chunk_index = chunk.chunk_index,
        c.page_number = chunk.page_number,
        c.section = chunk.section,
        c.doc_type = chunk.doc_type,
        c.embedding = chunk.embedding,
        c.updated_at = datetime()
    """
    async with driver.session() as session:
        await session.run(query, chunks=chunks)


async def upsert_document_node(
    driver: AsyncDriver,
    document_id: str,
    engagement_id: str,
    filename: str,
    doc_type: str,
) -> None:
    """Upsert a Document node and link chunks to it."""
    query = """
    MERGE (d:Document {id: $document_id})
    SET d.engagement_id = $engagement_id,
        d.filename = $filename,
        d.doc_type = $doc_type
    WITH d
    MATCH (c:Chunk {document_id: $document_id})
    MERGE (c)-[:BELONGS_TO]->(d)
    """
    async with driver.session() as session:
        await session.run(
            query,
            document_id=document_id,
            engagement_id=engagement_id,
            filename=filename,
            doc_type=doc_type,
        )


async def upsert_entity(driver: AsyncDriver, entity: dict[str, Any]) -> None:
    """Upsert an Entity node."""
    query = """
    MERGE (e:Entity {id: $entity_id})
    SET e.name = $name,
        e.type = $type,
        e.engagement_id = $engagement_id,
        e.description = $description,
        e.source_chunk_ids = $source_chunk_ids,
        e.source_document_ids = $source_document_ids,
        e.extraction_model = $extraction_model,
        e.schema_version = $schema_version,
        e.confidentiality_level = $confidentiality_level,
        e.embedding = $embedding
    """
    async with driver.session() as session:
        await session.run(query, **entity)


async def upsert_entities_batch(driver: AsyncDriver, entities: list[dict[str, Any]]) -> None:
    """Batch upsert Entity nodes."""
    query = """
    UNWIND $entities AS ent
    MERGE (e:Entity {id: ent.entity_id})
    SET e.name = ent.name,
        e.type = ent.type,
        e.engagement_id = ent.engagement_id,
        e.description = ent.description,
        e.source_chunk_ids = ent.source_chunk_ids,
        e.source_document_ids = ent.source_document_ids,
        e.extraction_model = ent.extraction_model,
        e.schema_version = ent.schema_version,
        e.confidentiality_level = ent.confidentiality_level,
        e.embedding = ent.embedding
    """
    async with driver.session() as session:
        await session.run(query, entities=entities)


async def upsert_relationship(driver: AsyncDriver, rel: dict[str, Any]) -> None:
    """Upsert a RELATES_TO relationship between entities."""
    query = """
    MATCH (from_e:Entity {id: $from_entity_id})
    MATCH (to_e:Entity {id: $to_entity_id})
    MERGE (from_e)-[r:RELATES_TO {id: $relationship_id}]->(to_e)
    SET r.type = $type,
        r.confidence = $confidence,
        r.source_chunk_id = $source_chunk_id,
        r.source_document_id = $source_document_id,
        r.engagement_id = $engagement_id
    """
    async with driver.session() as session:
        await session.run(query, **rel)


async def create_entity_chunk_links(
    driver: AsyncDriver,
    entity_id: str,
    chunk_ids: list[str],
    confidence: float = 1.0,
) -> None:
    """Create MENTIONED_IN relationships from entity to chunks."""
    query = """
    MATCH (e:Entity {id: $entity_id})
    UNWIND $chunk_ids AS cid
    MATCH (c:Chunk {id: cid})
    MERGE (e)-[r:MENTIONED_IN]->(c)
    SET r.confidence = $confidence
    """
    async with driver.session() as session:
        await session.run(
            query, entity_id=entity_id, chunk_ids=chunk_ids, confidence=confidence
        )


async def delete_document_data(driver: AsyncDriver, document_id: str) -> None:
    """Delete all nodes and relationships for a document."""
    queries = [
        "MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk {document_id: $doc_id}) DETACH DELETE e",
        "MATCH (c:Chunk {document_id: $doc_id}) DETACH DELETE c",
        "MATCH (d:Document {id: $doc_id}) DETACH DELETE d",
    ]
    async with driver.session() as session:
        for q in queries:
            await session.run(q, doc_id=document_id)


async def deduplicate_entities(driver: AsyncDriver, engagement_id: str) -> int:
    """Merge duplicate entities within an engagement (same name+type)."""
    query = """
    MATCH (e:Entity {engagement_id: $engagement_id})
    WITH e.name AS name, e.type AS type, collect(e) AS dupes
    WHERE size(dupes) > 1
    WITH dupes
    UNWIND dupes[1..] AS dup
    WITH dupes[0] AS keeper, dup
    MATCH (dup)-[r_in]-()
    WITH keeper, dup, collect(r_in) AS rels
    CALL {
        WITH keeper, dup
        MATCH (dup)-[:MENTIONED_IN]->(c:Chunk)
        MERGE (keeper)-[:MENTIONED_IN]->(c)
    }
    CALL {
        WITH keeper, dup
        SET keeper.source_chunk_ids = keeper.source_chunk_ids + dup.source_chunk_ids
    }
    DETACH DELETE dup
    RETURN count(dup) AS merged
    """
    async with driver.session() as session:
        result = await session.run(query, engagement_id=engagement_id)
        record = await result.single()
        return record["merged"] if record else 0
