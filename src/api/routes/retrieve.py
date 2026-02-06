from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def retrieve_auto():
    """Auto-routed retrieval (query router selects mode)."""
    # TODO: Phase 1
    return {"status": "not_implemented"}


@router.post("/vector")
async def retrieve_vector():
    """Vector-only retrieval via Neo4j HNSW index."""
    # TODO: Phase 1
    return {"status": "not_implemented"}


@router.post("/fulltext")
async def retrieve_fulltext():
    """Fulltext-only retrieval via Neo4j Lucene index."""
    # TODO: Phase 1
    return {"status": "not_implemented"}


@router.post("/graph")
async def retrieve_graph():
    """Graph-only retrieval via Neo4j traversal."""
    # TODO: Phase 1
    return {"status": "not_implemented"}


@router.post("/hybrid")
async def retrieve_hybrid():
    """Full hybrid retrieval (V+K+G) via Neo4j HybridCypherRetriever."""
    # TODO: Phase 1
    return {"status": "not_implemented"}
