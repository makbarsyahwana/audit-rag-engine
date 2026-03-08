"""Retrieval request/response models."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    VECTOR = "vector"
    FULLTEXT = "fulltext"
    GRAPH = "graph"
    HYBRID = "hybrid"
    ENTITY_VECTOR = "entity_vector"
    GRAPH_VECTOR_FULLTEXT = "graph_vector_fulltext"
    AUTO = "auto"


class RetrievalFilters(BaseModel):
    """Filters applied during retrieval."""

    doc_types: list[str] = Field(default_factory=list)
    confidentiality_levels: list[str] = Field(default_factory=list)
    entity_types: list[str] = Field(default_factory=list)
    date_range: Optional[dict] = None  # {"start": "...", "end": "..."}


class GraphExpansion(BaseModel):
    """Graph expansion config for retrieval."""

    enabled: bool = True
    depth: int = 2


class RetrieveRequest(BaseModel):
    """Request body for retrieval endpoints."""

    query: str
    engagement_id: str
    mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = 10
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    graph_expansion: GraphExpansion = Field(default_factory=GraphExpansion)
    app_mode: str = "audit"  # "audit" | "legal" | "compliance"


class RelatedEntity(BaseModel):
    """An entity related to a retrieved chunk."""

    name: str
    type: str
    relationship: str = ""


class RetrievedChunk(BaseModel):
    """A single chunk returned from retrieval."""

    chunk_id: str
    content: str
    score: float = 0.0
    document_id: str = ""
    document_name: str = ""
    page_number: Optional[int] = None
    section: Optional[str] = None
    doc_type: str = ""
    entities: list[RelatedEntity] = Field(default_factory=list)
    related_entities: list[RelatedEntity] = Field(default_factory=list)


class RetrieveResponse(BaseModel):
    """Response from retrieval endpoints."""

    query: str
    mode: RetrievalMode
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    total_results: int = 0
    latency_ms: float = 0.0


class GenerateRequest(BaseModel):
    """Request body for the generation endpoint."""

    query: str
    engagement_id: str
    mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = 10
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    graph_expansion: GraphExpansion = Field(default_factory=GraphExpansion)
    app_mode: str = "audit"  # "audit" | "legal" | "compliance"


class Citation(BaseModel):
    """A citation extracted from the LLM response."""

    chunk_id: str
    document_id: str = ""
    document_name: str = ""
    excerpt: str = ""
    page: Optional[int] = None
    score: float = 0.0


class GenerateResponse(BaseModel):
    """Response from the generation endpoint."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    chunks_retrieved: int = 0
    latency_ms: float = 0.0
    abstained: bool = False
