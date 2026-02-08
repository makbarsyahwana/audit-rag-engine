import logging
import time

from fastapi import APIRouter

from src.config import settings
from src.generation.citations import check_abstention, extract_citations
from src.generation.llm import invoke_llm
from src.generation.prompts.qa import build_qa_messages
from src.models.retrieval import (
    GenerateRequest,
    GenerateResponse,
    RetrievalMode,
)
from src.retrieval.entity_vector import entity_vector_search
from src.retrieval.fulltext import fulltext_search
from src.retrieval.graph import graph_search
from src.retrieval.graph_vector_fulltext import graph_vector_fulltext_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.router import classify_query
from src.retrieval.vector import vector_search

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest):
    """Generate answer with citations from retrieved context.

    Pipeline:
    1. Route query → select retrieval mode
    2. Retrieve relevant chunks
    3. Build prompt with context
    4. Invoke LLM
    5. Extract citations and confidence
    6. Check for abstention
    """
    start = time.time()

    # 1. Route query
    mode = request.mode
    if mode == RetrievalMode.AUTO:
        mode = classify_query(request.query)

    # 2. Retrieve
    doc_types = request.filters.doc_types or None

    if mode == RetrievalMode.VECTOR:
        chunks, _ = await vector_search(
            query=request.query,
            engagement_id=request.engagement_id,
            top_k=request.top_k,
            doc_types=doc_types,
        )
    elif mode == RetrievalMode.FULLTEXT:
        chunks, _ = await fulltext_search(
            query=request.query,
            engagement_id=request.engagement_id,
            top_k=request.top_k,
            doc_types=doc_types,
        )
    elif mode == RetrievalMode.GRAPH:
        chunks, _ = await graph_search(
            query=request.query,
            engagement_id=request.engagement_id,
            top_k=request.top_k,
            depth=request.graph_expansion.depth,
        )
    elif mode == RetrievalMode.ENTITY_VECTOR:
        chunks, _ = await entity_vector_search(
            query=request.query,
            engagement_id=request.engagement_id,
            top_k=request.top_k,
            entity_types=request.filters.entity_types or None,
        )
    elif mode == RetrievalMode.GRAPH_VECTOR_FULLTEXT:
        chunks, _ = await graph_vector_fulltext_search(
            query=request.query,
            engagement_id=request.engagement_id,
            top_k=request.top_k,
            doc_types=doc_types,
        )
    else:
        chunks, _ = await hybrid_search(
            query=request.query,
            engagement_id=request.engagement_id,
            top_k=request.top_k,
            doc_types=doc_types,
        )

    # Optional reranking
    if settings.rerank_enabled:
        chunks = rerank(chunks, request.query, request.top_k)

    # 3. Handle no results
    if not chunks:
        latency_ms = (time.time() - start) * 1000
        return GenerateResponse(
            answer="I don't have sufficient evidence in the provided documents "
                   "to answer this question.",
            citations=[],
            confidence=0.0,
            explanation="No relevant chunks found for the given query and engagement.",
            retrieval_mode=mode,
            chunks_retrieved=0,
            latency_ms=latency_ms,
            abstained=True,
        )

    # 4. Build prompt and invoke LLM (include entity context)
    chunk_dicts = []
    for c in chunks:
        d = {
            "chunk_id": c.chunk_id,
            "content": c.content,
            "document_name": c.document_name,
            "page_number": c.page_number,
        }
        if c.entities:
            d["entities"] = [
                f"{e.name} ({e.type})" for e in c.entities if e.name
            ]
        if c.related_entities:
            d["related_entities"] = [
                f"{e.name} ({e.type}) [{e.relationship}]"
                for e in c.related_entities if e.name
            ]
        chunk_dicts.append(d)
    messages = build_qa_messages(request.query, chunk_dicts)
    llm_response = await invoke_llm(messages)

    # 5. Extract citations and confidence
    citations, confidence, cleaned_answer = extract_citations(llm_response, chunks)

    # 6. Check abstention
    abstained = check_abstention(llm_response)

    latency_ms = (time.time() - start) * 1000

    logger.info(
        "Generation complete: citations=%d, confidence=%.2f, abstained=%s, "
        "latency=%.0fms",
        len(citations),
        confidence,
        abstained,
        latency_ms,
    )

    return GenerateResponse(
        answer=cleaned_answer,
        citations=citations,
        confidence=confidence,
        explanation=f"Retrieved {len(chunks)} chunks via {mode.value} mode.",
        retrieval_mode=mode,
        chunks_retrieved=len(chunks),
        latency_ms=latency_ms,
        abstained=abstained,
    )
