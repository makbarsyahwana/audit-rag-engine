"""Unit tests for Pydantic models."""

from src.models.chunk import ChunkMetadata, ChunkRecord
from src.models.document import (
    ConfidentialityLevel,
    DocType,
    DocumentRecord,
    IngestResponse,
    ProcessingStatus,
)
from src.models.entity import EntityRecord, EntityType, RelationshipRecord, RelationshipType
from src.models.retrieval import GenerateRequest, RetrievalMode, RetrieveRequest


def test_document_record_defaults():
    doc = DocumentRecord(
        engagement_id="eng-1",
        filename="test.pdf",
    )
    assert doc.doc_type == DocType.OTHER
    assert doc.confidentiality_level == ConfidentialityLevel.INTERNAL
    assert doc.processing_status == ProcessingStatus.PENDING
    assert doc.version == 1


def test_chunk_record():
    chunk = ChunkRecord(
        chunk_id="c-1",
        document_id="d-1",
        engagement_id="e-1",
        chunk_index=0,
        content="Test content",
    )
    assert chunk.content_preview == ""
    assert chunk.token_count == 0


def test_chunk_metadata():
    meta = ChunkMetadata(page_number=5, heading="Section 1")
    assert meta.page_number == 5
    assert meta.section is None


def test_entity_record_defaults():
    entity = EntityRecord(
        entity_id="ent-1",
        name="CHG-01",
        type=EntityType.CONTROL,
        engagement_id="e-1",
    )
    assert entity.description == ""
    assert entity.schema_version == "1.0"


def test_relationship_record():
    rel = RelationshipRecord(
        relationship_id="r-1",
        from_entity_id="e-1",
        to_entity_id="e-2",
        type=RelationshipType.IMPLEMENTS,
        confidence=0.95,
    )
    assert rel.confidence == 0.95


def test_retrieve_request_defaults():
    req = RetrieveRequest(
        query="test",
        engagement_id="e-1",
    )
    assert req.mode == RetrievalMode.HYBRID
    assert req.top_k == 10
    assert req.filters.doc_types == []


def test_generate_request_defaults():
    req = GenerateRequest(
        query="What controls exist?",
        engagement_id="e-1",
    )
    assert req.mode == RetrievalMode.HYBRID


def test_ingest_response():
    resp = IngestResponse(
        job_id="j-1",
        document_id="d-1",
        filename="test.pdf",
        status=ProcessingStatus.COMPLETED,
    )
    assert resp.message == ""


def test_retrieval_mode_values():
    assert RetrievalMode.VECTOR.value == "vector"
    assert RetrievalMode.FULLTEXT.value == "fulltext"
    assert RetrievalMode.GRAPH.value == "graph"
    assert RetrievalMode.HYBRID.value == "hybrid"
    assert RetrievalMode.AUTO.value == "auto"
