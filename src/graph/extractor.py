"""Entity/relationship extraction from chunks via LLM (OpenAI function calling)."""

import json
import logging
import uuid
from typing import Optional

from langchain_openai import ChatOpenAI

from src.config import settings
from src.graph.schema import GRAPH_SCHEMA_DESCRIPTION
from src.models.entity import (
    EntityRecord,
    EntityType,
    ExtractionResult,
    RelationshipRecord,
    RelationshipType,
)

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = f"""You are an expert audit knowledge graph extractor.
Given a text chunk from an audit document, extract entities and relationships.

{GRAPH_SCHEMA_DESCRIPTION}

Rules:
- Only extract entities and relationships that are explicitly mentioned in the text.
- Use the exact entity and relationship types listed above.
- Each entity must have a name, type, and brief description.
- Each relationship must specify from_entity, to_entity, and type.
- Assign a confidence score (0.0 to 1.0) to each relationship.
- If no entities or relationships are found, return empty lists.
"""

EXTRACTION_USER_PROMPT = """Extract entities and relationships from this text chunk:

---
{chunk_text}
---

Return a JSON object with this exact structure:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}}
  ],
  "relationships": [
    {{"from_entity": "...", "to_entity": "...", "type": "...", "confidence": 0.9}}
  ]
}}
"""

_llm_client: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    """Get or create the LLM client for extraction."""
    global _llm_client
    if _llm_client is None:
        _llm_client = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.0,
            openai_api_key=settings.openai_api_key,
        )
    return _llm_client


def _safe_entity_type(raw_type: str) -> EntityType:
    """Map a raw type string to EntityType enum, defaulting to OTHER."""
    raw = raw_type.lower().strip()
    for et in EntityType:
        if et.value == raw:
            return et
    return EntityType.OTHER


def _safe_relationship_type(raw_type: str) -> RelationshipType:
    """Map a raw type string to RelationshipType enum, defaulting to RELATED_TO."""
    raw = raw_type.lower().strip()
    for rt in RelationshipType:
        if rt.value == raw:
            return rt
    return RelationshipType.RELATED_TO


async def extract_entities_and_relationships(
    chunk_text: str,
    chunk_id: str,
    document_id: str,
    engagement_id: str,
    confidentiality_level: str = "internal",
) -> ExtractionResult:
    """Extract entities and relationships from a single chunk via LLM.

    Args:
        chunk_text: The text content of the chunk.
        chunk_id: ID of the source chunk (for provenance).
        document_id: ID of the source document (for provenance).
        engagement_id: Engagement scope.
        confidentiality_level: Inherited from document.

    Returns:
        ExtractionResult with entities and relationships.
    """
    llm = _get_llm()

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": EXTRACTION_USER_PROMPT.format(chunk_text=chunk_text)},
        ])

        # Parse the JSON from LLM response
        content = response.content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        parsed = json.loads(content)

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("ER extraction failed for chunk %s: %s", chunk_id, exc)
        return ExtractionResult()

    # Build entity records
    entities = []
    entity_name_to_id: dict[str, str] = {}

    for raw_entity in parsed.get("entities", []):
        name = raw_entity.get("name", "").strip()
        if not name:
            continue

        entity_id = str(uuid.uuid4())
        entity_name_to_id[name] = entity_id

        entities.append(EntityRecord(
            entity_id=entity_id,
            name=name,
            type=_safe_entity_type(raw_entity.get("type", "other")),
            engagement_id=engagement_id,
            description=raw_entity.get("description", ""),
            source_chunk_ids=[chunk_id],
            source_document_ids=[document_id],
            extraction_model=settings.llm_model,
            confidentiality_level=confidentiality_level,
        ))

    # Build relationship records
    relationships = []
    for raw_rel in parsed.get("relationships", []):
        from_name = raw_rel.get("from_entity", "").strip()
        to_name = raw_rel.get("to_entity", "").strip()

        from_id = entity_name_to_id.get(from_name)
        to_id = entity_name_to_id.get(to_name)

        if not from_id or not to_id:
            continue

        relationships.append(RelationshipRecord(
            relationship_id=str(uuid.uuid4()),
            from_entity_id=from_id,
            to_entity_id=to_id,
            type=_safe_relationship_type(raw_rel.get("type", "related_to")),
            confidence=float(raw_rel.get("confidence", 0.5)),
            source_chunk_id=chunk_id,
            source_document_id=document_id,
            engagement_id=engagement_id,
            extraction_model=settings.llm_model,
        ))

    logger.info(
        "Extracted %d entities, %d relationships from chunk %s",
        len(entities),
        len(relationships),
        chunk_id,
    )

    return ExtractionResult(entities=entities, relationships=relationships)
