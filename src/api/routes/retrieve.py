import logging

from fastapi import APIRouter

from src.config import settings
from src.models.retrieval import (
    RetrievalMode,
    RetrieveRequest,
    RetrieveResponse,
)
from src.retrieval.fulltext import fulltext_search
from src.retrieval.graph import graph_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.router import classify_query
from src.retrieval.vector import vector_search

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=RetrieveResponse)
async def retrieve_auto(request: RetrieveRequest):
    """Auto-routed retrieval (query router selects mode)."""
    mode = request.mode
    if mode == RetrievalMode.AUTO:
        mode = classify_query(request.query)

    # Delegate to the appropriate mode handler
    doc_types = request.filters.doc_types or None

    if mode == RetrievalMode.VECTOR:
        chunks, latency = await vector_search(
            query=request.query,
            engagement_id=request.engagement_id,
            top_k=request.top_k,
            doc_types=doc_types,
        )
    elif mode == RetrievalMode.FULLTEXT:
        chunks, latency = await fulltext_search(
            query=request.query,
            engagement_id=request.engagement_id,
            top_k=request.top_k,
            doc_types=doc_types,
        )
    elif mode == RetrievalMode.GRAPH:
        chunks, latency = await graph_search(
            query=request.query,
            engagement_id=request.engagement_id,
            top_k=request.top_k,
            depth=request.graph_expansion.depth,
        )
    else:
        chunks, latency = await hybrid_search(
            query=request.query,
            engagement_id=request.engagement_id,
            top_k=request.top_k,
            doc_types=doc_types,
        )

    # Optional reranking
    if settings.rerank_enabled:
        chunks = rerank(chunks, request.query, request.top_k)

    return RetrieveResponse(
        query=request.query,
        mode=mode,
        chunks=chunks,
        total_results=len(chunks),
        latency_ms=latency,
    )


@router.post("/vector", response_model=RetrieveResponse)
async def retrieve_vector(request: RetrieveRequest):
    """Vector-only retrieval via Neo4j HNSW index."""
    doc_types = request.filters.doc_types or None
    chunks, latency = await vector_search(
        query=request.query,
        engagement_id=request.engagement_id,
        top_k=request.top_k,
        doc_types=doc_types,
    )

    return RetrieveResponse(
        query=request.query,
        mode=RetrievalMode.VECTOR,
        chunks=chunks,
        total_results=len(chunks),
        latency_ms=latency,
    )


@router.post("/fulltext", response_model=RetrieveResponse)
async def retrieve_fulltext(request: RetrieveRequest):
    """Fulltext-only retrieval via Neo4j Lucene index."""
    doc_types = request.filters.doc_types or None
    chunks, latency = await fulltext_search(
        query=request.query,
        engagement_id=request.engagement_id,
        top_k=request.top_k,
        doc_types=doc_types,
    )

    return RetrieveResponse(
        query=request.query,
        mode=RetrievalMode.FULLTEXT,
        chunks=chunks,
        total_results=len(chunks),
        latency_ms=latency,
    )


@router.post("/graph", response_model=RetrieveResponse)
async def retrieve_graph(request: RetrieveRequest):
    """Graph-only retrieval via Neo4j traversal."""
    chunks, latency = await graph_search(
        query=request.query,
        engagement_id=request.engagement_id,
        top_k=request.top_k,
        depth=request.graph_expansion.depth,
    )

    return RetrieveResponse(
        query=request.query,
        mode=RetrievalMode.GRAPH,
        chunks=chunks,
        total_results=len(chunks),
        latency_ms=latency,
    )


@router.post("/hybrid", response_model=RetrieveResponse)
async def retrieve_hybrid(request: RetrieveRequest):
    """Full hybrid retrieval (V+K+G) via Neo4j unified query."""
    doc_types = request.filters.doc_types or None
    chunks, latency = await hybrid_search(
        query=request.query,
        engagement_id=request.engagement_id,
        top_k=request.top_k,
        doc_types=doc_types,
    )

    if settings.rerank_enabled:
        chunks = rerank(chunks, request.query, request.top_k)

    return RetrieveResponse(
        query=request.query,
        mode=RetrievalMode.HYBRID,
        chunks=chunks,
        total_results=len(chunks),
        latency_ms=latency,
    )
