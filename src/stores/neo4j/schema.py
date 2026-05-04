"""Neo4j index and schema management (DDL)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config import settings

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


async def ensure_indexes(driver: AsyncDriver) -> None:
    """Create vector, fulltext, and property indexes if they don't exist."""
    dims = settings.embedding_dimensions
    statements = [
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
        (
            "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS "
            "FOR (c:Chunk) ON EACH [c.content]"
        ),
        (
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name, e.description]"
        ),
        "CREATE INDEX chunk_engagement IF NOT EXISTS FOR (c:Chunk) ON (c.engagement_id)",
        "CREATE INDEX chunk_doc_type IF NOT EXISTS FOR (c:Chunk) ON (c.doc_type)",
        "CREATE INDEX entity_engagement IF NOT EXISTS FOR (e:Entity) ON (e.engagement_id)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        "CREATE INDEX document_engagement IF NOT EXISTS FOR (d:Document) ON (d.engagement_id)",
    ]
    async with driver.session() as session:
        for stmt in statements:
            try:
                await session.run(stmt)
            except Exception as exc:
                logger.warning("Index creation skipped or failed: %s — %s", stmt[:60], exc)
    logger.info("Neo4j indexes ensured")
