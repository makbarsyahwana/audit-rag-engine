"""Retrieval request/response models."""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelCaseModel(BaseModel):
    """Base for response models — serialize fields as camelCase.

    `populate_by_name=True` lets callers (and our own code) still construct
    instances with the Python snake_case field names. Routes must opt in to
    camelCase serialization via `response_model_by_alias=True`.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


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

    query: str = Field(..., min_length=1)
    engagement_id: str = Field(..., min_length=1)
    mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=10, ge=1, le=100)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    graph_expansion: GraphExpansion = Field(default_factory=GraphExpansion)
    app_mode: Literal["audit", "legal", "compliance"] = "audit"


class RelatedEntity(CamelCaseModel):
    """An entity related to a retrieved chunk."""

    name: str
    type: str
    relationship: str = ""


class RetrievedChunk(CamelCaseModel):
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


class RetrieveResponse(CamelCaseModel):
    """Response from retrieval endpoints."""

    query: str
    mode: RetrievalMode
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    total_results: int = 0
    latency_ms: float = 0.0


class GenerateRequest(BaseModel):
    """Request body for the generation endpoint."""

    query: str = Field(..., min_length=1)
    engagement_id: str = Field(..., min_length=1)
    mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=10, ge=1, le=100)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    graph_expansion: GraphExpansion = Field(default_factory=GraphExpansion)
    app_mode: Literal["audit", "legal", "compliance"] = "audit"


class Citation(CamelCaseModel):
    """A citation extracted from the LLM response."""

    chunk_id: str
    document_id: str = ""
    document_name: str = ""
    excerpt: str = ""
    page_number: Optional[int] = None
    score: float = 0.0


class GenerateResponse(CamelCaseModel):
    """Response from the generation endpoint."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    chunks_retrieved: int = 0
    latency_ms: float = 0.0
    abstained: bool = False
    model: str = ""  # LLM model name used for generation (consumed by NestJS)
    prompt_tokens: int = 0
    completion_tokens: int = 0
