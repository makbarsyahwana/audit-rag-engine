"""Chunk-related Pydantic models."""

from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata for a single chunk derived from Docling."""

    page_number: Optional[int] = None
    section: Optional[str] = None
    heading: Optional[str] = None
    heading_path: list[str] = Field(default_factory=list)
    bbox: Optional[dict] = None  # {"x0", "y0", "x1", "y1", "page"}


class ChunkRecord(BaseModel):
    """Full chunk record stored in MongoDB."""

    id: Optional[str] = Field(None, alias="_id")
    chunk_id: str
    document_id: str
    engagement_id: str
    chunk_index: int
    content: str
    content_preview: str = ""
    token_count: int = 0
    doc_type: str = "other"
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"populate_by_name": True}


class ChunkWithEmbedding(BaseModel):
    """Chunk data ready for Neo4j upsert (includes embedding vector)."""

    chunk_id: str
    document_id: str
    engagement_id: str
    chunk_index: int
    content: str
    content_preview: str = ""
    doc_type: str = "other"
    page_number: Optional[int] = None
    section: Optional[str] = None
    heading: Optional[str] = None
    embedding: list[float] = Field(default_factory=list)
