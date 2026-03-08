import logging
import time

from fastapi import APIRouter, HTTPException, Request

from src.config import settings
from src.generation.citations import check_abstention, extract_citations
from src.generation.llm import invoke_llm
from src.generation.prompts.prompt_router import get_qa_builder
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
from src.security.behavioral_monitor import behavioral_monitor
from src.security.circuit_breaker import llm_circuit_breaker
from src.security.kill_switch import kill_switch
from src.security.output_guard import check_output
from src.security.prompt_guard import scan_query
from src.security.service_auth import parse_identity, verify_engagement_access
from src.security.token_budget import token_budget

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest, req: Request):
    """Generate answer with citations from retrieved context.

    Pipeline:
    0. Security checks (ASI01/03/08/09)
    1. Route query → select retrieval mode
    2. Retrieve relevant chunks
    3. Build prompt with context
    4. Invoke LLM
    5. Extract citations and confidence
    6. Check for abstention
    7. Output guardrails
    """
    start = time.time()

    # 0a. Verify engagement access (ASI03)
    identity = parse_identity(req)
    verify_engagement_access(identity, request.engagement_id)

    # 0b. Kill switch check (ASI09)
    ks_level = await kill_switch.get_level()
    if not kill_switch.is_generation_allowed(ks_level):
        reason = await kill_switch.get_reason()
        raise HTTPException(
            status_code=503,
            detail=f"Generation disabled (kill switch: {ks_level.value}). {reason}",
        )

    # 0c. Prompt injection scan (ASI01)
    scan = scan_query(request.query)
    if scan.blocked:
        raise HTTPException(status_code=400, detail=scan.message)

    # 0d. Token budget check (ASI02)
    if not await token_budget.check_budget(request.engagement_id):
        raise HTTPException(
            status_code=429,
            detail="Daily token budget exceeded for this engagement",
        )

    # 0e. Circuit breaker check (ASI08)
    if not llm_circuit_breaker.is_allowed():
        fallback = llm_circuit_breaker.get_fallback_response()
        return GenerateResponse(
            answer=fallback,
            citations=[],
            confidence=0.0,
            explanation="Circuit breaker open — LLM temporarily unavailable.",
            retrieval_mode=RetrievalMode.AUTO,
            chunks_retrieved=0,
            latency_ms=(time.time() - start) * 1000,
            abstained=True,
        )

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
    build_messages = get_qa_builder(request.app_mode)
    messages = build_messages(request.query, chunk_dicts)
    try:
        llm_response = await invoke_llm(messages)
        llm_circuit_breaker.record_success()
    except Exception as exc:
        llm_circuit_breaker.record_failure()
        logger.error("LLM invocation failed: %s", exc)
        raise HTTPException(
            status_code=502, detail="LLM generation failed"
        ) from exc

    # 5. Extract citations and confidence
    citations, confidence, cleaned_answer = extract_citations(llm_response, chunks)

    # 6. Check abstention
    abstained = check_abstention(llm_response)

    # 7. Output guardrails (ASI01 + ASI06)
    guard = check_output(
        response_text=cleaned_answer,
        confidence=confidence,
        engagement_id=request.engagement_id,
    )
    if not guard.passed:
        logger.warning("Output blocked: %s", guard.blocked_reason)
        cleaned_answer = (
            "This response was blocked by security guardrails. "
            f"Reason: {guard.blocked_reason}"
        )
        confidence = 0.0
        abstained = True

    latency_ms = (time.time() - start) * 1000

    # Record usage and behavior (ASI02 + ASI10)
    await token_budget.record_usage(request.engagement_id)
    behavioral_monitor.record(
        engagement_id=request.engagement_id,
        response_length=len(cleaned_answer),
        confidence=confidence,
        citation_count=len(citations),
        is_abstention=abstained,
    )

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
