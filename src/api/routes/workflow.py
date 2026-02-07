"""Phase 2 workflow API routes."""

import logging
import time

from fastapi import APIRouter

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
from src.workflows.evidence_detection import detect_missing_evidence
from src.workflows.traceability import generate_traceability_matrix

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Evidence search
# ---------------------------------------------------------------------------

@router.post("/evidence/search", response_model=EvidenceSearchResponse)
async def search_evidence(request: EvidenceSearchRequest):
    """Search for evidence with control, period, and entity filters."""
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
async def build_traceability_matrix(request: TraceabilityMatrixRequest):
    """Generate a requirement-control traceability matrix."""
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
async def find_evidence_gaps(request: MissingEvidenceRequest):
    """Detect missing or incomplete evidence for controls."""
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
async def draft_workpaper(request: WorkpaperDraftRequest):
    """Generate a workpaper narrative draft from evidence."""
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
    llm_response = await invoke_llm(messages)

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
async def draft_finding(request: FindingDraftRequest):
    """Generate a finding draft from evidence."""
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
    llm_response = await invoke_llm(messages)

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
