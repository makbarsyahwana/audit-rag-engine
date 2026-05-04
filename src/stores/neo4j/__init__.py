"""Neo4j store — facade over schema, write, and query modules.

All external call sites continue to use ``neo4j_store.<method>(...)``;
internally, each method delegates to the appropriate sub-module.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from neo4j import AsyncDriver, AsyncGraphDatabase

from src.config import settings
from src.stores.neo4j import queries, schema, writes

logger = logging.getLogger(__name__)


class Neo4jStore:
    """Async Neo4j driver wrapper — thin facade over schema, writes, and queries."""

    _driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        logger.info("Connected to Neo4j at %s", settings.neo4j_uri)

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            logger.info("Neo4j connection closed")

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        return self._driver

    # -- Schema ----------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        await schema.ensure_indexes(self.driver)

    # -- Writes ----------------------------------------------------------------

    async def upsert_chunk(self, chunk: dict[str, Any]) -> None:
        await writes.upsert_chunk(self.driver, chunk)

    async def upsert_chunks_batch(self, chunks: list[dict[str, Any]]) -> None:
        await writes.upsert_chunks_batch(self.driver, chunks)

    async def upsert_document_node(
        self, document_id: str, engagement_id: str, filename: str, doc_type: str
    ) -> None:
        await writes.upsert_document_node(self.driver, document_id, engagement_id, filename, doc_type)

    async def upsert_entity(self, entity: dict[str, Any]) -> None:
        await writes.upsert_entity(self.driver, entity)

    async def upsert_entities_batch(self, entities: list[dict[str, Any]]) -> None:
        await writes.upsert_entities_batch(self.driver, entities)

    async def upsert_relationship(self, rel: dict[str, Any]) -> None:
        await writes.upsert_relationship(self.driver, rel)

    async def create_entity_chunk_links(
        self, entity_id: str, chunk_ids: list[str], confidence: float = 1.0
    ) -> None:
        await writes.create_entity_chunk_links(self.driver, entity_id, chunk_ids, confidence)

    async def delete_document_data(self, document_id: str) -> None:
        await writes.delete_document_data(self.driver, document_id)

    async def deduplicate_entities(self, engagement_id: str) -> int:
        return await writes.deduplicate_entities(self.driver, engagement_id)

    # -- Queries ---------------------------------------------------------------

    async def vector_search(
        self,
        query_embedding: list[float],
        engagement_id: str,
        top_k: int = 10,
        doc_types: Optional[list[str]] = None,
        date_start: Optional[datetime] = None,
        date_end: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        return await queries.vector_search(
            self.driver, query_embedding, engagement_id, top_k, doc_types, date_start, date_end
        )

    async def fulltext_search(
        self,
        query_text: str,
        engagement_id: str,
        top_k: int = 10,
        doc_types: Optional[list[str]] = None,
        date_start: Optional[datetime] = None,
        date_end: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        return await queries.fulltext_search(
            self.driver, query_text, engagement_id, top_k, doc_types, date_start, date_end
        )

    async def graph_search(
        self,
        search_term: str,
        engagement_id: str,
        depth: int = 2,
        top_k: int = 10,
        date_start: Optional[datetime] = None,
        date_end: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        return await queries.graph_search(
            self.driver, search_term, engagement_id, depth, top_k, date_start, date_end
        )

    async def entity_vector_search(
        self,
        query_embedding: list[float],
        engagement_id: str,
        top_k: int = 10,
        entity_types: Optional[list[str]] = None,
        date_start: Optional[datetime] = None,
        date_end: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        return await queries.entity_vector_search(
            self.driver, query_embedding, engagement_id, top_k, entity_types, date_start, date_end
        )

    async def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        engagement_id: str,
        top_k: int = 10,
        doc_types: Optional[list[str]] = None,
        date_start: Optional[datetime] = None,
        date_end: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        return await queries.hybrid_search(
            self.driver, query_embedding, query_text, engagement_id, top_k, doc_types, date_start, date_end
        )

    async def graph_vector_fulltext_search(
        self,
        query_embedding: list[float],
        query_text: str,
        engagement_id: str,
        top_k: int = 10,
        doc_types: Optional[list[str]] = None,
        date_start: Optional[datetime] = None,
        date_end: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        return await queries.graph_vector_fulltext_search(
            self.driver, query_embedding, query_text, engagement_id, top_k, doc_types, date_start, date_end
        )


neo4j_store = Neo4jStore()
