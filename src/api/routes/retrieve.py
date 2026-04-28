import logging

from fastapi import APIRouter, HTTPException, Request

from src.config import settings
from src.models.retrieval import (
    RetrievalMode,
    RetrieveRequest,
    RetrieveResponse,
)
from src.retrieval.entity_vector import entity_vector_search
from src.retrieval.fulltext import fulltext_search
from src.retrieval.graph import graph_search
from src.retrieval.graph_vector_fulltext import graph_vector_fulltext_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.router import classify_query
from src.retrieval.vector import vector_search
from src.security.prompt_guard import scan_query
from src.security.service_auth import parse_identity, verify_engagement_access

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=RetrieveResponse, response_model_by_alias=True)
async def retrieve_auto(request: RetrieveRequest, req: Request):
    """Auto-routed retrieval (query router selects mode)."""
    # Security checks (ASI01 + ASI03)
    identity = parse_identity(req)
    verify_engagement_access(identity, request.engagement_id)
    scan = scan_query(request.query)
    if scan.blocked:
        raise HTTPException(status_code=400, detail=scan.message)

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
    elif mode == RetrievalMode.ENTITY_VECTOR:
        chunks, latency = await entity_vector_search(
            query=request.query,
            engagement_id=request.engagement_id,
            top_k=request.top_k,
            entity_types=request.filters.entity_types or None,
        )
    elif mode == RetrievalMode.GRAPH_VECTOR_FULLTEXT:
        chunks, latency = await graph_vector_fulltext_search(
            query=request.query,
            engagement_id=request.engagement_id,
            top_k=request.top_k,
            doc_types=doc_types,
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


@router.post("/vector", response_model=RetrieveResponse, response_model_by_alias=True)
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


@router.post("/fulltext", response_model=RetrieveResponse, response_model_by_alias=True)
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


@router.post("/graph", response_model=RetrieveResponse, response_model_by_alias=True)
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


@router.post("/entity-vector", response_model=RetrieveResponse, response_model_by_alias=True)
async def retrieve_entity_vector(request: RetrieveRequest):
    """Entity vector retrieval via Neo4j entity embeddings KNN."""
    chunks, latency = await entity_vector_search(
        query=request.query,
        engagement_id=request.engagement_id,
        top_k=request.top_k,
        entity_types=request.filters.entity_types or None,
    )

    return RetrieveResponse(
        query=request.query,
        mode=RetrievalMode.ENTITY_VECTOR,
        chunks=chunks,
        total_results=len(chunks),
        latency_ms=latency,
    )


@router.post(
    "/graph-vector-fulltext",
    response_model=RetrieveResponse,
    response_model_by_alias=True,
)
async def retrieve_graph_vector_fulltext(request: RetrieveRequest):
    """Full hybrid retrieval: vector + fulltext + graph expansion."""
    doc_types = request.filters.doc_types or None
    chunks, latency = await graph_vector_fulltext_search(
        query=request.query,
        engagement_id=request.engagement_id,
        top_k=request.top_k,
        doc_types=doc_types,
    )

    if settings.rerank_enabled:
        chunks = rerank(chunks, request.query, request.top_k)

    return RetrieveResponse(
        query=request.query,
        mode=RetrievalMode.GRAPH_VECTOR_FULLTEXT,
        chunks=chunks,
        total_results=len(chunks),
        latency_ms=latency,
    )


@router.post("/hybrid", response_model=RetrieveResponse, response_model_by_alias=True)
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
