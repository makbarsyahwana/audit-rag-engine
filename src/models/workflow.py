"""Pydantic models for Phase 2 workflow endpoints."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field  # noqa: I001


# ---------------------------------------------------------------------------
# Evidence search
# ---------------------------------------------------------------------------

class EvidenceSearchRequest(BaseModel):
    """Request body for evidence search with audit-specific filters."""

    query: str
    engagement_id: str
    control_id: Optional[str] = None
    period: Optional[str] = None          # e.g. "Q1 2025", "2025-01/2025-03"
    entity: Optional[str] = None          # e.g. "acme-corp"
    doc_types: list[str] = Field(default_factory=list)
    top_k: int = 10


class EvidenceChunk(BaseModel):
    """A chunk returned from evidence search with control/period metadata."""

    chunk_id: str
    content: str
    score: float = 0.0
    document_id: str = ""
    document_name: str = ""
    page_number: Optional[int] = None
    section: Optional[str] = None
    doc_type: str = ""
    control_ids: list[str] = Field(default_factory=list)
    period: Optional[str] = None
    entity: Optional[str] = None


class EvidenceSearchResponse(BaseModel):
    """Response from evidence search."""

    query: str
    engagement_id: str
    chunks: list[EvidenceChunk] = Field(default_factory=list)
    total_results: int = 0
    filters_applied: dict = Field(default_factory=dict)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Traceability matrix
# ---------------------------------------------------------------------------

class ControlCoverage(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"
    UNTESTED = "untested"


class RequirementMapping(BaseModel):
    """A single row in the traceability matrix."""

    requirement_id: str
    requirement_ref: str             # e.g. "A.12.1.2"
    requirement_title: str
    framework: str                   # e.g. "ISO 27001"
    controls: list["ControlLink"] = Field(default_factory=list)
    evidence_count: int = 0
    coverage: ControlCoverage = ControlCoverage.UNTESTED


class ControlLink(BaseModel):
    """A control linked to a requirement."""

    control_id: str
    control_ref: str                 # e.g. "CHG-01"
    control_title: str
    evidence_count: int = 0
    coverage: ControlCoverage = ControlCoverage.UNTESTED


class TraceabilityMatrixRequest(BaseModel):
    """Request body for traceability matrix generation."""

    engagement_id: str
    framework: Optional[str] = None  # filter to specific framework
    requirements: list["RequirementInput"] = Field(default_factory=list)
    controls: list["ControlInput"] = Field(default_factory=list)
    mappings: list["MappingInput"] = Field(default_factory=list)


class RequirementInput(BaseModel):
    """Requirement data for matrix generation."""

    id: str
    ref: str
    title: str
    framework: str


class ControlInput(BaseModel):
    """Control data for matrix generation."""

    id: str
    ref: str
    title: str


class MappingInput(BaseModel):
    """Requirement-to-control mapping input."""

    requirement_id: str
    control_id: str
    coverage: str = "full"


class TraceabilityMatrixResponse(BaseModel):
    """Response from traceability matrix generation."""

    engagement_id: str
    framework: Optional[str] = None
    matrix: list[RequirementMapping] = Field(default_factory=list)
    summary: "MatrixSummary"
    gaps: list["CoverageGap"] = Field(default_factory=list)


class MatrixSummary(BaseModel):
    """Summary statistics for the traceability matrix."""

    total_requirements: int = 0
    total_controls: int = 0
    fully_covered: int = 0
    partially_covered: int = 0
    not_covered: int = 0
    coverage_percentage: float = 0.0


class CoverageGap(BaseModel):
    """A gap identified in the traceability matrix."""

    requirement_id: str
    requirement_ref: str
    requirement_title: str
    framework: str
    gap_type: str                    # "unmapped", "no_evidence", "partial"
    description: str


# ---------------------------------------------------------------------------
# Missing evidence detection
# ---------------------------------------------------------------------------

class MissingEvidenceRequest(BaseModel):
    """Request to detect missing evidence for an engagement."""

    engagement_id: str
    controls: list["ControlEvidenceInput"] = Field(default_factory=list)
    required_periods: list[str] = Field(default_factory=list)  # e.g. ["Q1 2025", "Q2 2025"]


class ControlEvidenceInput(BaseModel):
    """Control with its expected evidence for gap detection."""

    control_id: str
    control_ref: str
    control_title: str
    expected_evidence_types: list[str] = Field(default_factory=list)
    frequency: Optional[str] = None  # e.g. "quarterly", "annual"


class EvidenceGap(BaseModel):
    """A single missing evidence item."""

    control_id: str
    control_ref: str
    control_title: str
    gap_type: str                    # "missing", "expired", "incomplete"
    period: Optional[str] = None
    expected_type: Optional[str] = None
    description: str
    severity: str = "medium"         # high, medium, low


class MissingEvidenceResponse(BaseModel):
    """Response from missing evidence detection."""

    engagement_id: str
    total_controls: int = 0
    controls_with_gaps: int = 0
    gaps: list[EvidenceGap] = Field(default_factory=list)
    summary: str = ""
    completeness_percentage: float = 0.0


# ---------------------------------------------------------------------------
# Workpaper / finding draft generation
# ---------------------------------------------------------------------------

class WorkpaperDraftRequest(BaseModel):
    """Request body for workpaper narrative drafting."""

    engagement_id: str
    title: str
    objective: str = ""
    scope: str = ""
    control_ref: str = ""
    query: str = ""                  # optional search query for evidence retrieval
    top_k: int = 15


class WorkpaperDraftResponse(BaseModel):
    """Response from workpaper drafting."""

    title: str
    draft: str
    citations: list[dict] = Field(default_factory=list)
    confidence: float = 0.0
    chunks_used: int = 0
    latency_ms: float = 0.0


class FindingDraftRequest(BaseModel):
    """Request body for finding drafting."""

    engagement_id: str
    title: str
    control_ref: str = ""
    observation: str = ""
    query: str = ""                  # optional search query for evidence retrieval
    top_k: int = 15


class FindingDraftResponse(BaseModel):
    """Response from finding drafting."""

    title: str
    draft: str
    citations: list[dict] = Field(default_factory=list)
    confidence: float = 0.0
    suggested_severity: Optional[str] = None
    chunks_used: int = 0
    latency_ms: float = 0.0
