"""Entity and relationship models for the knowledge graph."""

from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    CONTROL = "control"
    REQUIREMENT = "requirement"
    SYSTEM = "system"
    PROCESS = "process"
    PERSON = "person"
    ORGANIZATION = "organization"
    DOCUMENT_REF = "document_ref"
    RISK = "risk"
    POLICY = "policy"
    FRAMEWORK = "framework"
    OTHER = "other"


class RelationshipType(str, Enum):
    IMPLEMENTS = "implements"
    MAPS_TO = "maps_to"
    OWNS = "owns"
    SUPPORTS = "supports"
    REFERENCES = "references"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    MITIGATES = "mitigates"
    RELATED_TO = "related_to"


class EntityRecord(BaseModel):
    """Entity node for the knowledge graph."""

    entity_id: str
    name: str
    type: EntityType = EntityType.OTHER
    engagement_id: str
    description: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    extraction_model: str = ""
    schema_version: str = "1.0"
    confidentiality_level: str = "internal"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RelationshipRecord(BaseModel):
    """Relationship edge for the knowledge graph."""

    relationship_id: str
    from_entity_id: str
    to_entity_id: str
    type: RelationshipType = RelationshipType.RELATED_TO
    confidence: float = 0.0
    source_chunk_id: Optional[str] = None
    source_document_id: Optional[str] = None
    engagement_id: str = ""
    extraction_model: str = ""


class ExtractionResult(BaseModel):
    """Result of LLM-based entity/relationship extraction from a chunk."""

    entities: list[EntityRecord] = Field(default_factory=list)
    relationships: list[RelationshipRecord] = Field(default_factory=list)
