"""Neo4j client for unified vector + graph + fulltext operations."""

import logging
from typing import Any, Optional

from neo4j import AsyncDriver, AsyncGraphDatabase

from src.config import settings

logger = logging.getLogger(__name__)


class Neo4jStore:
    """Async Neo4j driver wrapper for vector, fulltext, and graph operations."""

    _driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        """Initialize the Neo4j async driver."""
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        logger.info("Connected to Neo4j at %s", settings.neo4j_uri)

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver is not None:
            await self._driver.close()
            logger.info("Neo4j connection closed")

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        return self._driver

    # ------------------------------------------------------------------
    # Index initialization
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create vector, fulltext, and property indexes if they don't exist."""
        dims = settings.embedding_dimensions
        statements = [
            # Vector indexes (HNSW)
            (
                f"CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS "
                f"FOR (c:Chunk) ON (c.embedding) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dims}, "
                f"`vector.similarity_function`: 'cosine'}}}}"
            ),
            (
                f"CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS "
                f"FOR (e:Entity) ON (e.embedding) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dims}, "
                f"`vector.similarity_function`: 'cosine'}}}}"
            ),
            # Fulltext indexes (Lucene)
            (
                "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS "
                "FOR (c:Chunk) ON EACH [c.content]"
            ),
            (
                "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
                "FOR (e:Entity) ON EACH [e.name, e.description]"
            ),
            # Property indexes
            "CREATE INDEX chunk_engagement IF NOT EXISTS FOR (c:Chunk) ON (c.engagement_id)",
            "CREATE INDEX chunk_doc_type IF NOT EXISTS FOR (c:Chunk) ON (c.doc_type)",
            "CREATE INDEX entity_engagement IF NOT EXISTS FOR (e:Entity) ON (e.engagement_id)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX document_engagement IF NOT EXISTS FOR (d:Document) ON (d.engagement_id)",
        ]
        async with self.driver.session() as session:
            for stmt in statements:
                try:
                    await session.run(stmt)
                except Exception as exc:
                    logger.warning("Index creation skipped or failed: %s — %s", stmt[:60], exc)
        logger.info("Neo4j indexes ensured")

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    async def upsert_chunk(self, chunk: dict[str, Any]) -> None:
        """Upsert a Chunk node with embedding."""
        query = """
        MERGE (c:Chunk {id: $chunk_id})
        SET c.document_id = $document_id,
            c.engagement_id = $engagement_id,
            c.content = $content,
            c.content_preview = $content_preview,
            c.chunk_index = $chunk_index,
            c.page_number = $page_number,
            c.section = $section,
            c.doc_type = $doc_type,
            c.embedding = $embedding
        """
        async with self.driver.session() as session:
            await session.run(query, **chunk)

    async def upsert_chunks_batch(self, chunks: list[dict[str, Any]]) -> None:
        """Batch upsert Chunk nodes."""
        query = """
        UNWIND $chunks AS chunk
        MERGE (c:Chunk {id: chunk.chunk_id})
        SET c.document_id = chunk.document_id,
            c.engagement_id = chunk.engagement_id,
            c.content = chunk.content,
            c.content_preview = chunk.content_preview,
            c.chunk_index = chunk.chunk_index,
            c.page_number = chunk.page_number,
            c.section = chunk.section,
            c.doc_type = chunk.doc_type,
            c.embedding = chunk.embedding
        """
        async with self.driver.session() as session:
            await session.run(query, chunks=chunks)

    # ------------------------------------------------------------------
    # Document node operations
    # ------------------------------------------------------------------

    async def upsert_document_node(
        self, document_id: str, engagement_id: str, filename: str, doc_type: str
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
        async with self.driver.session() as session:
            await session.run(
                query,
                document_id=document_id,
                engagement_id=engagement_id,
                filename=filename,
                doc_type=doc_type,
            )

    # ------------------------------------------------------------------
    # Entity / relationship operations
    # ------------------------------------------------------------------

    async def upsert_entity(self, entity: dict[str, Any]) -> None:
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
        async with self.driver.session() as session:
            await session.run(query, **entity)

    async def upsert_entities_batch(self, entities: list[dict[str, Any]]) -> None:
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
        async with self.driver.session() as session:
            await session.run(query, entities=entities)

    async def upsert_relationship(self, rel: dict[str, Any]) -> None:
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
        async with self.driver.session() as session:
            await session.run(query, **rel)

    async def create_entity_chunk_links(
        self, entity_id: str, chunk_ids: list[str], confidence: float = 1.0
    ) -> None:
        """Create MENTIONED_IN relationships from entity to chunks."""
        query = """
        MATCH (e:Entity {id: $entity_id})
        UNWIND $chunk_ids AS cid
        MATCH (c:Chunk {id: cid})
        MERGE (e)-[r:MENTIONED_IN]->(c)
        SET r.confidence = $confidence
        """
        async with self.driver.session() as session:
            await session.run(
                query, entity_id=entity_id, chunk_ids=chunk_ids, confidence=confidence
            )

    # ------------------------------------------------------------------
    # Retrieval: Vector search
    # ------------------------------------------------------------------

    async def vector_search(
        self,
        query_embedding: list[float],
        engagement_id: str,
        top_k: int = 10,
        doc_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search over chunk embeddings."""
        where_clause = "WHERE chunk.engagement_id = $engagement_id"
        if doc_types:
            where_clause += " AND chunk.doc_type IN $doc_types"

        query = f"""
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
        YIELD node AS chunk, score
        {where_clause}
        OPTIONAL MATCH (chunk)-[:BELONGS_TO]->(doc:Document)
        RETURN chunk, score, doc
        ORDER BY score DESC
        LIMIT $top_k
        """
        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "engagement_id": engagement_id,
            "top_k": top_k,
        }
        if doc_types:
            params["doc_types"] = doc_types

        async with self.driver.session() as session:
            result = await session.run(query, **params)
            records = [record.data() async for record in result]
        return records

    # ------------------------------------------------------------------
    # Retrieval: Fulltext search
    # ------------------------------------------------------------------

    async def fulltext_search(
        self,
        query_text: str,
        engagement_id: str,
        top_k: int = 10,
        doc_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Fulltext (keyword) search over chunk content."""
        where_clause = "WHERE chunk.engagement_id = $engagement_id"
        if doc_types:
            where_clause += " AND chunk.doc_type IN $doc_types"

        query = f"""
        CALL db.index.fulltext.queryNodes('chunk_fulltext', $query_text)
        YIELD node AS chunk, score
        {where_clause}
        OPTIONAL MATCH (chunk)-[:BELONGS_TO]->(doc:Document)
        RETURN chunk, score, doc
        ORDER BY score DESC
        LIMIT $top_k
        """
        params: dict[str, Any] = {
            "query_text": query_text,
            "engagement_id": engagement_id,
            "top_k": top_k,
        }
        if doc_types:
            params["doc_types"] = doc_types

        async with self.driver.session() as session:
            result = await session.run(query, **params)
            records = [record.data() async for record in result]
        return records

    # ------------------------------------------------------------------
    # Retrieval: Graph traversal
    # ------------------------------------------------------------------

    async def graph_search(
        self,
        search_term: str,
        engagement_id: str,
        depth: int = 2,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Graph traversal: find entities and related chunks."""
        query = """
        CALL db.index.fulltext.queryNodes('entity_fulltext', $search_term)
        YIELD node AS entity, score
        WHERE entity.engagement_id = $engagement_id
        WITH entity, score
        ORDER BY score DESC
        LIMIT $top_k
        OPTIONAL MATCH (entity)-[:MENTIONED_IN]->(chunk:Chunk)
        OPTIONAL MATCH (chunk)-[:BELONGS_TO]->(doc:Document)
        OPTIONAL MATCH (entity)-[rel:RELATES_TO]-(related:Entity)
        RETURN entity, score,
               collect(DISTINCT {chunk: chunk, doc: doc}) AS chunk_docs,
               collect(DISTINCT {entity: related, relationship: type(rel)}) AS related_entities
        """
        async with self.driver.session() as session:
            result = await session.run(
                query,
                search_term=search_term,
                engagement_id=engagement_id,
                top_k=top_k,
            )
            records = [record.data() async for record in result]
        return records

    # ------------------------------------------------------------------
    # Retrieval: Entity vector search (KNN over entity embeddings)
    # ------------------------------------------------------------------

    async def entity_vector_search(
        self,
        query_embedding: list[float],
        engagement_id: str,
        top_k: int = 10,
        entity_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """KNN search over entity embeddings, expand to linked chunks."""
        where_clause = "WHERE entity.engagement_id = $engagement_id"
        if entity_types:
            where_clause += " AND entity.type IN $entity_types"

        query = f"""
        CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $query_embedding)
        YIELD node AS entity, score
        {where_clause}
        WITH entity, score
        ORDER BY score DESC
        LIMIT $top_k
        OPTIONAL MATCH (entity)-[:MENTIONED_IN]->(chunk:Chunk)
        OPTIONAL MATCH (chunk)-[:BELONGS_TO]->(doc:Document)
        OPTIONAL MATCH (entity)-[rel:RELATES_TO]-(related:Entity)
        RETURN entity, score,
               collect(DISTINCT {{chunk: chunk, doc: doc}}) AS chunk_docs,
               collect(DISTINCT {{entity: related, relationship: type(rel)}}) AS related_entities
        """
        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "engagement_id": engagement_id,
            "top_k": top_k,
        }
        if entity_types:
            params["entity_types"] = entity_types

        async with self.driver.session() as session:
            result = await session.run(query, **params)
            records = [record.data() async for record in result]
        return records

    # ------------------------------------------------------------------
    # Retrieval: Hybrid (V + K + G)
    # ------------------------------------------------------------------

    async def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        engagement_id: str,
        top_k: int = 10,
        doc_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Full hybrid retrieval: vector + fulltext + graph expansion."""
        where_clause = "WHERE chunk.engagement_id = $engagement_id"
        if doc_types:
            where_clause += " AND chunk.doc_type IN $doc_types"

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
        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "engagement_id": engagement_id,
            "top_k": top_k,
        }
        if doc_types:
            params["doc_types"] = doc_types

        async with self.driver.session() as session:
            result = await session.run(query, **params)
            records = [record.data() async for record in result]
        return records

    # ------------------------------------------------------------------
    # Retrieval: Graph + Vector + Fulltext (full hybrid)
    # ------------------------------------------------------------------

    async def graph_vector_fulltext_search(
        self,
        query_embedding: list[float],
        query_text: str,
        engagement_id: str,
        top_k: int = 10,
        doc_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Full hybrid: vector KNN ∪ fulltext hits → graph expansion.

        Combines vector similarity, keyword matching, and graph traversal
        into a single merged result set — matching llm-graph-builder's
        graph_vector_fulltext mode.
        """
        where_filter = "AND chunk.doc_type IN $doc_types" if doc_types else ""

        query = f"""
        // Vector KNN hits
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
        YIELD node AS chunk, score
        WHERE chunk.engagement_id = $engagement_id {where_filter}
        WITH collect({{chunk: chunk, score: score}}) AS vector_hits

        // Fulltext hits
        CALL db.index.fulltext.queryNodes('chunk_fulltext', $query_text)
        YIELD node AS ft_chunk, score AS ft_score
        WHERE ft_chunk.engagement_id = $engagement_id {where_filter}
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
        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "query_text": query_text,
            "engagement_id": engagement_id,
            "top_k": top_k,
        }
        if doc_types:
            params["doc_types"] = doc_types

        async with self.driver.session() as session:
            result = await session.run(query, **params)
            records = [record.data() async for record in result]
        return records

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def delete_document_data(self, document_id: str) -> None:
        """Delete all nodes and relationships for a document."""
        queries = [
            "MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk {document_id: $doc_id}) DETACH DELETE e",
            "MATCH (c:Chunk {document_id: $doc_id}) DETACH DELETE c",
            "MATCH (d:Document {id: $doc_id}) DETACH DELETE d",
        ]
        async with self.driver.session() as session:
            for q in queries:
                await session.run(q, doc_id=document_id)

    async def deduplicate_entities(self, engagement_id: str) -> int:
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
        async with self.driver.session() as session:
            result = await session.run(query, engagement_id=engagement_id)
            record = await result.single()
            return record["merged"] if record else 0


neo4j_store = Neo4jStore()
