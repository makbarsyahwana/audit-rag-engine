"""Document-related Pydantic models."""

from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocType(str, Enum):
    POLICY = "policy"
    FRAMEWORK = "framework"
    EVIDENCE = "evidence"
    WORKPAPER = "workpaper"
    REPORT = "report"
    TICKET = "ticket"
    STANDARD = "standard"
    REGULATION = "regulation"
    SOP = "sop"
    OTHER = "other"


class ConfidentialityLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceSystem(str, Enum):
    UPLOAD = "upload"
    SHAREPOINT = "sharepoint"
    CONFLUENCE = "confluence"
    GRC = "grc"


class DocumentMetadata(BaseModel):
    """Metadata attached to an ingested document."""

    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)
    framework: Optional[str] = None
    clause_id: Optional[str] = None
    control_id: Optional[str] = None
    process: Optional[str] = None
    system: Optional[str] = None
    custom_fields: dict = Field(default_factory=dict)


class DocumentACL(BaseModel):
    """Access control list for a document."""

    engagement_ids: list[str] = Field(default_factory=list)
    user_ids: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    """Full document record stored in MongoDB."""

    id: Optional[str] = Field(None, alias="_id")
    engagement_id: str
    source_system: SourceSystem = SourceSystem.UPLOAD
    source_path: Optional[str] = None
    filename: str
    mime_type: Optional[str] = None
    file_size: Optional[int] = None
    doc_type: DocType = DocType.OTHER
    confidentiality_level: ConfidentialityLevel = ConfidentialityLevel.INTERNAL
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    acl: DocumentACL = Field(default_factory=DocumentACL)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    content_hash: Optional[str] = None
    version: int = 1
    supersedes_document_id: Optional[str] = None
    docling_json: Optional[dict] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"populate_by_name": True}


class IngestRequest(BaseModel):
    """Request body fields for document ingestion (form data mapped)."""

    engagement_id: str
    doc_type: DocType = DocType.OTHER
    confidentiality_level: ConfidentialityLevel = ConfidentialityLevel.INTERNAL
    source_system: SourceSystem = SourceSystem.UPLOAD
    title: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    framework: Optional[str] = None
    clause_id: Optional[str] = None
    control_id: Optional[str] = None


class IngestResponse(BaseModel):
    """Response from document ingestion endpoint."""

    job_id: str
    document_id: str
    filename: str
    status: ProcessingStatus
    message: str = ""


class JobStatusResponse(BaseModel):
    """Response for ingestion job status check."""

    job_id: str
    document_id: Optional[str] = None
    status: ProcessingStatus
    filename: Optional[str] = None
    chunks_created: int = 0
    entities_extracted: int = 0
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
