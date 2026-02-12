"""Phase 2 workflow API routes."""

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from src.generation.citations import extract_citations
from src.generation.llm import invoke_llm
from src.generation.prompts.finding import build_finding_messages
from src.generation.prompts.workpaper import build_workpaper_messages
from src.models.workflow import (
    EvidenceSearchRequest,
    EvidenceSearchResponse,
    FindingDraftRequest,
    FindingDraftResponse,
    MissingEvidenceRequest,
    MissingEvidenceResponse,
    TraceabilityMatrixRequest,
    TraceabilityMatrixResponse,
    WorkpaperDraftRequest,
    WorkpaperDraftResponse,
)
from src.retrieval.evidence import evidence_search
from src.security.behavioral_monitor import behavioral_monitor
from src.security.circuit_breaker import llm_circuit_breaker
from src.security.kill_switch import kill_switch
from src.security.output_guard import check_output
from src.security.prompt_guard import scan_query
from src.security.service_auth import parse_identity, verify_engagement_access
from src.workflows.evidence_detection import detect_missing_evidence
from src.workflows.traceability import generate_traceability_matrix

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Evidence search
# ---------------------------------------------------------------------------

@router.post("/evidence/search", response_model=EvidenceSearchResponse)
async def search_evidence(request: EvidenceSearchRequest, req: Request):
    """Search for evidence with control, period, and entity filters."""
    # Security: identity + access (ASI03) + prompt guard (ASI01)
    identity = parse_identity(req)
    verify_engagement_access(identity, request.engagement_id)
    scan_result = scan_query(request.query)
    if scan_result.blocked:
        raise HTTPException(status_code=400, detail=f"Query blocked: {scan_result.reason}")

    chunks, latency_ms = await evidence_search(
        query=request.query,
        engagement_id=request.engagement_id,
        top_k=request.top_k,
        control_id=request.control_id,
        period=request.period,
        entity=request.entity,
        doc_types=request.doc_types or None,
    )

    filters_applied = {}
    if request.control_id:
        filters_applied["control_id"] = request.control_id
    if request.period:
        filters_applied["period"] = request.period
    if request.entity:
        filters_applied["entity"] = request.entity
    if request.doc_types:
        filters_applied["doc_types"] = request.doc_types

    return EvidenceSearchResponse(
        query=request.query,
        engagement_id=request.engagement_id,
        chunks=chunks,
        total_results=len(chunks),
        filters_applied=filters_applied,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Traceability matrix
# ---------------------------------------------------------------------------

@router.post(
    "/traceability/matrix",
    response_model=TraceabilityMatrixResponse,
)
async def build_traceability_matrix(request: TraceabilityMatrixRequest, req: Request):
    """Generate a requirement-control traceability matrix."""
    # Security: identity + access (ASI03)
    identity = parse_identity(req)
    verify_engagement_access(identity, request.engagement_id)

    return await generate_traceability_matrix(
        engagement_id=request.engagement_id,
        requirements=request.requirements,
        controls=request.controls,
        mappings=request.mappings,
        framework=request.framework,
    )


# ---------------------------------------------------------------------------
# Missing evidence detection
# ---------------------------------------------------------------------------

@router.post(
    "/evidence/gaps",
    response_model=MissingEvidenceResponse,
)
async def find_evidence_gaps(request: MissingEvidenceRequest, req: Request):
    """Detect missing or incomplete evidence for controls."""
    # Security: identity + access (ASI03)
    identity = parse_identity(req)
    verify_engagement_access(identity, request.engagement_id)

    return await detect_missing_evidence(
        engagement_id=request.engagement_id,
        controls=request.controls,
        required_periods=request.required_periods,
    )


# ---------------------------------------------------------------------------
# Workpaper draft generation
# ---------------------------------------------------------------------------

@router.post(
    "/draft/workpaper",
    response_model=WorkpaperDraftResponse,
)
async def draft_workpaper(request: WorkpaperDraftRequest, req: Request):
    """Generate a workpaper narrative draft from evidence."""
    # Security: identity + access (ASI03) + kill switch (ASI09)
    identity = parse_identity(req)
    verify_engagement_access(identity, request.engagement_id)
    level = kill_switch.get_level()
    if level != "active":
        raise HTTPException(status_code=503, detail=f"Service in {level} mode")

    start = time.time()

    # Retrieve evidence for the workpaper topic
    query = request.query or f"{request.title} {request.control_ref}"
    chunks, _ = await evidence_search(
        query=query,
        engagement_id=request.engagement_id,
        top_k=request.top_k,
        control_id=request.control_ref or None,
    )

    if not chunks:
        latency_ms = (time.time() - start) * 1000
        return WorkpaperDraftResponse(
            title=request.title,
            draft=(
                "Unable to generate workpaper draft — no relevant "
                "evidence found in the engagement documents."
            ),
            citations=[],
            confidence=0.0,
            chunks_used=0,
            latency_ms=latency_ms,
        )

    # Build prompt and invoke LLM
    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "content": c.content,
            "document_name": c.document_name,
            "page_number": c.page_number,
        }
        for c in chunks
    ]
    messages = build_workpaper_messages(
        chunks=chunk_dicts,
        title=request.title,
        objective=request.objective,
        scope=request.scope,
        control_ref=request.control_ref,
    )

    # Security: circuit breaker (ASI08)
    if not llm_circuit_breaker.is_allowed():
        raise HTTPException(status_code=503, detail="LLM service temporarily unavailable")
    try:
        llm_response = await invoke_llm(messages)
        llm_circuit_breaker.record_success()
    except Exception as exc:
        llm_circuit_breaker.record_failure()
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    # Security: output guard (ASI01/06) + behavioral monitor (ASI10)
    output_result = check_output(llm_response, request.engagement_id)
    if output_result.blocked:
        logger.warning("Output blocked in workpaper draft: %s", output_result.reason)
        llm_response = "[Output blocked by security controls. Please retry or contact admin.]"
    behavioral_monitor.record(len(llm_response), 0.0)

    # Extract citations and confidence
    from src.models.retrieval import RetrievedChunk

    retrieved = [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            content=c.content,
            score=c.score,
            document_id=c.document_id,
            document_name=c.document_name,
            page_number=c.page_number,
        )
        for c in chunks
    ]
    citations, confidence, cleaned = extract_citations(
        llm_response, retrieved
    )

    latency_ms = (time.time() - start) * 1000

    return WorkpaperDraftResponse(
        title=request.title,
        draft=cleaned,
        citations=[c.model_dump() for c in citations],
        confidence=confidence,
        chunks_used=len(chunks),
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Finding draft generation
# ---------------------------------------------------------------------------

@router.post(
    "/draft/finding",
    response_model=FindingDraftResponse,
)
async def draft_finding(request: FindingDraftRequest, req: Request):
    """Generate a finding draft from evidence."""
    # Security: identity + access (ASI03) + kill switch (ASI09)
    identity = parse_identity(req)
    verify_engagement_access(identity, request.engagement_id)
    level = kill_switch.get_level()
    if level != "active":
        raise HTTPException(status_code=503, detail=f"Service in {level} mode")

    start = time.time()

    # Retrieve evidence for the finding topic
    query = (
        request.query
        or f"{request.title} {request.control_ref} {request.observation}"
    )
    chunks, _ = await evidence_search(
        query=query,
        engagement_id=request.engagement_id,
        top_k=request.top_k,
        control_id=request.control_ref or None,
    )

    if not chunks:
        latency_ms = (time.time() - start) * 1000
        return FindingDraftResponse(
            title=request.title,
            draft=(
                "Unable to generate finding draft — no relevant "
                "evidence found in the engagement documents."
            ),
            citations=[],
            confidence=0.0,
            chunks_used=0,
            latency_ms=latency_ms,
        )

    # Build prompt and invoke LLM
    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "content": c.content,
            "document_name": c.document_name,
            "page_number": c.page_number,
        }
        for c in chunks
    ]
    messages = build_finding_messages(
        chunks=chunk_dicts,
        title=request.title,
        control_ref=request.control_ref,
        observation=request.observation,
    )

    # Security: circuit breaker (ASI08)
    if not llm_circuit_breaker.is_allowed():
        raise HTTPException(status_code=503, detail="LLM service temporarily unavailable")
    try:
        llm_response = await invoke_llm(messages)
        llm_circuit_breaker.record_success()
    except Exception as exc:
        llm_circuit_breaker.record_failure()
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    # Security: output guard (ASI01/06) + behavioral monitor (ASI10)
    output_result = check_output(llm_response, request.engagement_id)
    if output_result.blocked:
        logger.warning("Output blocked in finding draft: %s", output_result.reason)
        llm_response = "[Output blocked by security controls. Please retry or contact admin.]"
    behavioral_monitor.record(len(llm_response), 0.0)

    # Extract citations, confidence, and suggested severity
    from src.models.retrieval import RetrievedChunk

    retrieved = [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            content=c.content,
            score=c.score,
            document_id=c.document_id,
            document_name=c.document_name,
            page_number=c.page_number,
        )
        for c in chunks
    ]
    citations, confidence, cleaned = extract_citations(
        llm_response, retrieved
    )

    # Try to extract severity from the draft
    suggested_severity = _extract_severity(cleaned)

    latency_ms = (time.time() - start) * 1000

    return FindingDraftResponse(
        title=request.title,
        draft=cleaned,
        citations=[c.model_dump() for c in citations],
        confidence=confidence,
        suggested_severity=suggested_severity,
        chunks_used=len(chunks),
        latency_ms=latency_ms,
    )


def _extract_severity(draft: str) -> str | None:
    """Extract suggested severity from the LLM draft output."""
    import re

    pattern = re.compile(
        r"(?:suggested\s+)?severity[:\s]*"
        r"(CRITICAL|HIGH|MEDIUM|LOW|INFORMATIONAL)",
        re.IGNORECASE,
    )
    match = pattern.search(draft)
    return match.group(1).upper() if match else None
