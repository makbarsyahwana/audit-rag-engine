"""Entity deduplication within engagement scope."""

import logging

from src.stores.neo4j_store import neo4j_store

logger = logging.getLogger(__name__)


async def deduplicate_entities(engagement_id: str) -> int:
    """Merge duplicate entities (same name+type) within an engagement.

    Uses Neo4j to find entities with identical name and type,
    keeps the first one, merges relationships, and deletes duplicates.

    Args:
        engagement_id: Scope deduplication to this engagement.

    Returns:
        Number of duplicate entities merged.
    """
    merged = await neo4j_store.deduplicate_entities(engagement_id)
    logger.info(
        "Deduplicated entities for engagement %s: %d merged",
        engagement_id,
        merged,
    )
    return merged
