"""Discrete pipeline stage handlers for async RabbitMQ-based ingestion.

Each handler consumes from one queue and publishes to the next:
  1. process_handler  — Docling convert + chunk + embed + store chunks
  2. extract_handler  — LLM entity/rel extraction per chunk → store in Neo4j
  3. postprocess_handler — entity embeddings + MENTIONED_IN links + dedup

Also exports `run_ingestion_pipeline` for synchronous (non-queue) execution.
"""

import hashlib
import logging
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from src.config import settings
from src.graph.extractor import extract_entities_and_relationships
from src.ingestion.chunker import chunk_document
from src.ingestion.converter import convert_document, create_converter
from src.ingestion.embedder import embed_texts
from src.ingestion.task_broker import QUEUE_EXTRACT, QUEUE_POSTPROCESS, task_broker
from src.models.chunk import ChunkMetadata, ChunkRecord
from src.models.document import (
    DocumentACL,
    DocumentMetadata,
    DocumentRecord,
    ProcessingStatus,
)
from src.security.circuit_breaker import llm_circuit_breaker
from src.security.content_sanitizer import sanitize_text
from src.stores.document_store import document_store
from src.stores.neo4j_store import neo4j_store
from src.stores.object_store import object_store

logger = logging.getLogger(__name__)

_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        _converter = create_converter()
    return _converter


# ------------------------------------------------------------------
# Stage 1: Process (convert + chunk + embed + store chunks)
# ------------------------------------------------------------------

async def run_ingestion_pipeline(
    file_data: bytes,
    filename: str,
    engagement_id: str,
    doc_type: str = "other",
    corpus_scope: str = "engagement",
    confidentiality_level: str = "internal",
    source_system: str = "upload",
    title: Optional[str] = None,
    tags: Optional[list[str]] = None,
    framework: Optional[str] = None,
    clause_id: Optional[str] = None,
    control_id: Optional[str] = None,
    content_type: str = "application/octet-stream",
) -> dict:
    """Run the full ingestion pipeline for a single document (synchronous).

    Steps:
        1. Upload raw file to S3
        2. Create document record in MongoDB
        3. Convert document via Docling
        4. Chunk via HybridChunker
        5. Generate embeddings
        6. Upsert chunks to Neo4j (vector + fulltext)
        7. Store chunks in MongoDB
        8. Create Document node in Neo4j and link chunks

    Returns:
        Dict with job_id, document_id, status, and stats.
    """
    job_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    started_at = datetime.now(UTC)

    await document_store.create_job({
        "job_id": job_id,
        "document_id": document_id,
        "filename": filename,
        "status": ProcessingStatus.PROCESSING,
        "started_at": started_at,
        "chunks_created": 0,
        "entities_extracted": 0,
    })

    try:
        content_hash = hashlib.sha256(file_data).hexdigest()

        existing = await document_store.get_document_by_hash(content_hash, engagement_id)
        if existing:
            await document_store.update_job(
                job_id,
                status=ProcessingStatus.COMPLETED,
                completed_at=datetime.now(UTC),
                message=f"Duplicate of document {existing.get('_id')}",
            )
            return {
                "job_id": job_id,
                "document_id": str(existing.get("_id")),
                "status": "duplicate",
                "message": "Document already ingested",
            }

        if corpus_scope in ("global", "GLOBAL"):
            engagement_id = settings.global_engagement_id

        s3_key = f"{engagement_id}/{document_id}/{filename}"
        object_store.upload_file(s3_key, file_data, content_type)

        doc_record = DocumentRecord(
            _id=document_id,
            engagement_id=engagement_id,
            source_system=source_system,
            source_path=s3_key,
            filename=filename,
            mime_type=content_type,
            file_size=len(file_data),
            doc_type=doc_type,
            corpus_scope=corpus_scope,
            confidentiality_level=confidentiality_level,
            metadata=DocumentMetadata(
                title=title or filename,
                tags=tags or [],
                framework=framework,
                clause_id=clause_id,
                control_id=control_id,
            ),
            acl=DocumentACL(engagement_ids=[engagement_id]),
            processing_status=ProcessingStatus.PROCESSING,
            content_hash=content_hash,
        )
        await document_store.insert_document(doc_record.model_dump(by_alias=True))

        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(file_data)
            tmp.flush()
            docling_doc = convert_document(tmp.name, converter=_get_converter())

        try:
            docling_json = docling_doc.model_dump() if hasattr(docling_doc, "model_dump") else None
        except Exception:
            docling_json = None
        if docling_json:
            await document_store.update_document_status(
                document_id, ProcessingStatus.PROCESSING, docling_json=docling_json
            )

        chunk_dicts = chunk_document(docling_doc)
        for c in chunk_dicts:
            c["content"], _ = sanitize_text(c["content"], source_id=document_id)

        if not chunk_dicts:
            await document_store.update_document_status(document_id, ProcessingStatus.FAILED)
            await document_store.update_job(
                job_id,
                status=ProcessingStatus.FAILED,
                completed_at=datetime.now(UTC),
                error="No chunks produced from document",
            )
            return {
                "job_id": job_id,
                "document_id": document_id,
                "status": "failed",
                "message": "No chunks produced",
            }

        texts = [c["content"] for c in chunk_dicts]
        embeddings = await embed_texts(texts)

        mongo_chunks = []
        neo4j_chunks = []
        for idx, (chunk_dict, embedding) in enumerate(zip(chunk_dicts, embeddings)):
            chunk_id = str(uuid.uuid4())
            meta = chunk_dict.get("metadata", {})

            mongo_chunk = ChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                engagement_id=engagement_id,
                chunk_index=idx,
                content=chunk_dict["content"],
                content_preview=chunk_dict.get("content_preview", ""),
                token_count=len(chunk_dict["content"].split()),
                doc_type=doc_type,
                metadata=ChunkMetadata(
                    page_number=meta.get("page_number"),
                    section=meta.get("section"),
                    heading=meta.get("heading"),
                    heading_path=meta.get("heading_path", []),
                    bbox=meta.get("bbox"),
                ),
            )
            mongo_chunks.append(mongo_chunk.model_dump(by_alias=True))

            neo4j_chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "engagement_id": engagement_id,
                "content": chunk_dict["content"],
                "content_preview": chunk_dict.get("content_preview", ""),
                "chunk_index": idx,
                "page_number": meta.get("page_number"),
                "section": meta.get("section"),
                "doc_type": doc_type,
                "embedding": embedding,
            })

        await document_store.insert_chunks(mongo_chunks)
        await neo4j_store.upsert_chunks_batch(neo4j_chunks)
        await neo4j_store.upsert_document_node(
            document_id=document_id,
            engagement_id=engagement_id,
            filename=filename,
            doc_type=doc_type,
        )

        await document_store.update_document_status(document_id, ProcessingStatus.COMPLETED)

        completed_at = datetime.now(UTC)
        await document_store.update_job(
            job_id,
            status=ProcessingStatus.COMPLETED,
            completed_at=completed_at,
            chunks_created=len(neo4j_chunks),
        )

        logger.info(
            "Ingestion complete: doc=%s, chunks=%d, time=%.1fs",
            document_id,
            len(neo4j_chunks),
            (completed_at - started_at).total_seconds(),
        )

        return {
            "job_id": job_id,
            "document_id": document_id,
            "status": "completed",
            "chunks_created": len(neo4j_chunks),
            "filename": filename,
        }

    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", filename, exc, exc_info=True)
        await document_store.update_document_status(document_id, ProcessingStatus.FAILED)
        await document_store.update_job(
            job_id,
            status=ProcessingStatus.FAILED,
            completed_at=datetime.now(UTC),
            error=str(exc),
        )
        return {
            "job_id": job_id,
            "document_id": document_id,
            "status": "failed",
            "message": str(exc),
        }


async def process_handler(payload: dict[str, Any]) -> None:
    """Stage 1: Convert document, chunk, embed, store chunks.

    Consumes from QUEUE_PROCESS.
    Publishes to QUEUE_EXTRACT with chunk info for entity extraction.
    """
    job_id = payload["job_id"]
    document_id = payload["document_id"]
    engagement_id = payload["engagement_id"]
    s3_key = payload["s3_key"]
    filename = payload["filename"]
    doc_type = payload.get("doc_type", "other")
    corpus_scope = payload.get("corpus_scope", "engagement")
    confidentiality_level = payload.get("confidentiality_level", "internal")
    content_type = payload.get("content_type", "application/octet-stream")
    title = payload.get("title")
    tags = payload.get("tags", [])
    framework = payload.get("framework")
    clause_id = payload.get("clause_id")
    control_id = payload.get("control_id")
    source_system = payload.get("source_system", "upload")

    logger.info("Stage 1 (process): job=%s, doc=%s, file=%s", job_id, document_id, filename)

    await document_store.update_job(
        job_id, status=ProcessingStatus.PROCESSING, stage="process"
    )

    # Download file from S3
    file_data = object_store.download_file(s3_key)
    content_hash = hashlib.sha256(file_data).hexdigest()

    # Check for duplicate
    existing = await document_store.get_document_by_hash(content_hash, engagement_id)
    if existing:
        await document_store.update_job(
            job_id,
            status=ProcessingStatus.COMPLETED,
            completed_at=datetime.now(UTC),
            message=f"Duplicate of document {existing.get('_id')}",
        )
        logger.info("Duplicate detected: job=%s", job_id)
        return

    # Create document record in MongoDB
    doc_record = DocumentRecord(
        _id=document_id,
        engagement_id=engagement_id,
        source_system=source_system,
        source_path=s3_key,
        filename=filename,
        mime_type=content_type,
        file_size=len(file_data),
        doc_type=doc_type,
        corpus_scope=corpus_scope,
        confidentiality_level=confidentiality_level,
        metadata=DocumentMetadata(
            title=title or filename,
            tags=tags,
            framework=framework,
            clause_id=clause_id,
            control_id=control_id,
        ),
        acl=DocumentACL(engagement_ids=[engagement_id]),
        processing_status=ProcessingStatus.PROCESSING,
        content_hash=content_hash,
    )
    await document_store.insert_document(doc_record.model_dump(by_alias=True))

    # Convert document via Docling
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(file_data)
        tmp.flush()
        docling_doc = convert_document(tmp.name, converter=_get_converter())

    # Chunk
    chunk_dicts = chunk_document(docling_doc)

    # Sanitize chunk content (ASI06 — prevent RAG poisoning)
    for c in chunk_dicts:
        c["content"], _ = sanitize_text(c["content"], source_id=document_id)

    if not chunk_dicts:
        await document_store.update_document_status(document_id, ProcessingStatus.FAILED)
        await document_store.update_job(
            job_id,
            status=ProcessingStatus.FAILED,
            completed_at=datetime.now(UTC),
            error="No chunks produced from document",
        )
        return

    # Embed
    texts = [c["content"] for c in chunk_dicts]
    embeddings = await embed_texts(texts)

    # Build and store chunks in MongoDB + Neo4j
    mongo_chunks = []
    neo4j_chunks = []
    chunk_infos = []  # For next stage

    for idx, (chunk_dict, embedding) in enumerate(zip(chunk_dicts, embeddings)):
        chunk_id = str(uuid.uuid4())
        meta = chunk_dict.get("metadata", {})

        mongo_chunk = ChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            engagement_id=engagement_id,
            chunk_index=idx,
            content=chunk_dict["content"],
            content_preview=chunk_dict.get("content_preview", ""),
            token_count=len(chunk_dict["content"].split()),
            doc_type=doc_type,
            metadata=ChunkMetadata(
                page_number=meta.get("page_number"),
                section=meta.get("section"),
                heading=meta.get("heading"),
                heading_path=meta.get("heading_path", []),
                bbox=meta.get("bbox"),
            ),
        )
        mongo_chunks.append(mongo_chunk.model_dump(by_alias=True))

        neo4j_chunks.append({
            "chunk_id": chunk_id,
            "document_id": document_id,
            "engagement_id": engagement_id,
            "content": chunk_dict["content"],
            "content_preview": chunk_dict.get("content_preview", ""),
            "chunk_index": idx,
            "page_number": meta.get("page_number"),
            "section": meta.get("section"),
            "doc_type": doc_type,
            "embedding": embedding,
        })

        chunk_infos.append({
            "chunk_id": chunk_id,
            "content": chunk_dict["content"],
        })

    await document_store.insert_chunks(mongo_chunks)
    await neo4j_store.upsert_chunks_batch(neo4j_chunks)
    await neo4j_store.upsert_document_node(
        document_id=document_id,
        engagement_id=engagement_id,
        filename=filename,
        doc_type=doc_type,
    )

    await document_store.update_job(
        job_id, chunks_created=len(neo4j_chunks), stage="process_done"
    )

    logger.info(
        "Stage 1 done: job=%s, chunks=%d", job_id, len(neo4j_chunks)
    )

    # Publish to Stage 2 (entity extraction) if enabled
    if settings.entity_extraction_enabled:
        await task_broker.publish(QUEUE_EXTRACT, {
            "job_id": job_id,
            "document_id": document_id,
            "engagement_id": engagement_id,
            "confidentiality_level": confidentiality_level,
            "chunks": chunk_infos,
        })
    else:
        # Skip extraction, mark complete
        await document_store.update_document_status(
            document_id, ProcessingStatus.COMPLETED
        )
        await document_store.update_job(
            job_id,
            status=ProcessingStatus.COMPLETED,
            completed_at=datetime.now(UTC),
        )


# ------------------------------------------------------------------
# Stage 2: Extract (LLM entity/relationship extraction per chunk)
# ------------------------------------------------------------------

async def extract_handler(payload: dict[str, Any]) -> None:
    """Stage 2: Run LLM-based entity/relationship extraction on each chunk.

    Consumes from QUEUE_EXTRACT.
    Publishes to QUEUE_POSTPROCESS with extracted entities/relationships.
    """
    job_id = payload["job_id"]
    document_id = payload["document_id"]
    engagement_id = payload["engagement_id"]
    confidentiality_level = payload.get("confidentiality_level", "internal")
    chunks = payload["chunks"]

    logger.info(
        "Stage 2 (extract): job=%s, doc=%s, chunks=%d",
        job_id, document_id, len(chunks),
    )

    await document_store.update_job(job_id, stage="extract")

    all_entities = []
    all_relationships = []

    for chunk_info in chunks:
        chunk_id = chunk_info["chunk_id"]
        chunk_text = chunk_info["content"]

        # Security: circuit breaker for LLM calls (ASI08)
        if not llm_circuit_breaker.is_allowed():
            logger.warning(
                "Circuit breaker OPEN — skipping entity extraction: job=%s, chunk=%s",
                job_id, chunk_id,
            )
            continue

        try:
            result = await extract_entities_and_relationships(
                chunk_text=chunk_text,
                chunk_id=chunk_id,
                document_id=document_id,
                engagement_id=engagement_id,
                confidentiality_level=confidentiality_level,
            )
            llm_circuit_breaker.record_success()
        except Exception as exc:
            llm_circuit_breaker.record_failure()
            logger.warning(
                "LLM extraction failed (circuit breaker recorded): job=%s, chunk=%s, error=%s",
                job_id, chunk_id, exc,
            )
            continue

        for entity in result.entities:
            all_entities.append(entity.model_dump())

        for rel in result.relationships:
            all_relationships.append(rel.model_dump())

    # Store entities and relationships in Neo4j
    if all_entities:
        entity_dicts = []
        for ent in all_entities:
            entity_dicts.append({
                "entity_id": ent["entity_id"],
                "name": ent["name"],
                "type": ent["type"],
                "engagement_id": ent["engagement_id"],
                "description": ent["description"],
                "source_chunk_ids": ent["source_chunk_ids"],
                "source_document_ids": ent["source_document_ids"],
                "extraction_model": ent["extraction_model"],
                "schema_version": ent.get("schema_version", "1.0"),
                "confidentiality_level": ent.get("confidentiality_level", "internal"),
                "embedding": ent.get("embedding", []),
            })
        await neo4j_store.upsert_entities_batch(entity_dicts)

    for rel in all_relationships:
        await neo4j_store.upsert_relationship(rel)

    # Create MENTIONED_IN links: entity → chunk
    entity_chunk_map: dict[str, list[str]] = {}
    for ent in all_entities:
        eid = ent["entity_id"]
        for cid in ent["source_chunk_ids"]:
            entity_chunk_map.setdefault(eid, []).append(cid)

    for entity_id, chunk_ids in entity_chunk_map.items():
        await neo4j_store.create_entity_chunk_links(entity_id, chunk_ids)

    await document_store.update_job(
        job_id,
        entities_extracted=len(all_entities),
        stage="extract_done",
    )

    logger.info(
        "Stage 2 done: job=%s, entities=%d, relationships=%d",
        job_id, len(all_entities), len(all_relationships),
    )

    # Publish to Stage 3 (postprocess)
    entity_ids = [e["entity_id"] for e in all_entities]
    await task_broker.publish(QUEUE_POSTPROCESS, {
        "job_id": job_id,
        "document_id": document_id,
        "engagement_id": engagement_id,
        "entity_ids": entity_ids,
        "entity_data": [
            {"entity_id": e["entity_id"], "name": e["name"], "description": e["description"]}
            for e in all_entities
        ],
    })


# ------------------------------------------------------------------
# Stage 3: Postprocess (entity embeddings + dedup)
# ------------------------------------------------------------------

async def postprocess_handler(payload: dict[str, Any]) -> None:
    """Stage 3: Generate entity embeddings, run deduplication.

    Consumes from QUEUE_POSTPROCESS.
    Marks the job as complete.
    """
    job_id = payload["job_id"]
    document_id = payload["document_id"]
    engagement_id = payload["engagement_id"]
    entity_data = payload.get("entity_data", [])

    logger.info(
        "Stage 3 (postprocess): job=%s, doc=%s, entities=%d",
        job_id, document_id, len(entity_data),
    )

    await document_store.update_job(job_id, stage="postprocess")

    # Generate entity embeddings if enabled
    if settings.entity_embedding_enabled and entity_data:
        texts = [
            f"{e['name']}: {e['description']}" if e.get("description") else e["name"]
            for e in entity_data
        ]
        embeddings = await embed_texts(texts)

        # Update entity nodes with embeddings
        for ent, embedding in zip(entity_data, embeddings):
            await neo4j_store.upsert_entity({
                "entity_id": ent["entity_id"],
                "name": ent["name"],
                "type": "",  # won't overwrite existing since MERGE uses id
                "engagement_id": engagement_id,
                "description": ent.get("description", ""),
                "source_chunk_ids": [],
                "source_document_ids": [],
                "extraction_model": "",
                "schema_version": "1.0",
                "confidentiality_level": "internal",
                "embedding": embedding,
            })

        logger.info("Generated embeddings for %d entities", len(entity_data))

    # Deduplicate entities within the engagement
    merged = await neo4j_store.deduplicate_entities(engagement_id)
    if merged:
        logger.info("Deduplicated %d entity pairs in engagement %s", merged, engagement_id)

    # Mark document and job as complete
    await document_store.update_document_status(document_id, ProcessingStatus.COMPLETED)
    await document_store.update_job(
        job_id,
        status=ProcessingStatus.COMPLETED,
        completed_at=datetime.now(UTC),
        stage="complete",
    )

    logger.info("Stage 3 done: job=%s — ingestion complete", job_id)
