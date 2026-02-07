"""Initialize Neo4j indexes (vector, fulltext, property) for the audit RAG engine.

Usage:
    python -m scripts.init_indexes

See DATA_REQUIREMENTS.md §5.2 for index definitions.
"""

import asyncio
import sys

sys.path.insert(0, ".")

from src.stores.neo4j_store import neo4j_store


async def main() -> None:
    """Connect to Neo4j and create all required indexes."""
    print("Connecting to Neo4j...")
    await neo4j_store.connect()
    print("Creating indexes...")
    await neo4j_store.ensure_indexes()
    print("Done. All indexes created/verified.")
    await neo4j_store.close()


if __name__ == "__main__":
    asyncio.run(main())
