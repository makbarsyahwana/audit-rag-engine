"""Main ingestion pipeline: Docling convert → chunk → embed → Neo4j upsert."""

import hashlib
import logging
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from src.ingestion.chunker import chunk_document
from src.ingestion.converter import convert_document, create_converter
from src.ingestion.embedder import embed_texts
from src.models.chunk import ChunkMetadata, ChunkRecord
from src.models.document import (
    DocumentACL,
    DocumentMetadata,
    DocumentRecord,
    ProcessingStatus,
)
from src.stores.document_store import document_store
from src.stores.neo4j_store import neo4j_store
from src.stores.object_store import object_store

logger = logging.getLogger(__name__)

# Lazy-initialized shared converter
_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        _converter = create_converter()
    return _converter


async def run_ingestion_pipeline(
    file_data: bytes,
    filename: str,
    engagement_id: str,
    doc_type: str = "other",
    confidentiality_level: str = "internal",
    source_system: str = "upload",
    title: Optional[str] = None,
    tags: Optional[list[str]] = None,
    framework: Optional[str] = None,
    clause_id: Optional[str] = None,
    control_id: Optional[str] = None,
    content_type: str = "application/octet-stream",
) -> dict:
    """Run the full ingestion pipeline for a single document.

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

    # Create job tracking record
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
        # 1. Content hash for dedup
        content_hash = hashlib.sha256(file_data).hexdigest()

        # Check for duplicate
        existing = await document_store.get_document_by_hash(
            content_hash, engagement_id
        )
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

        # 2. Upload raw file to S3
        s3_key = f"{engagement_id}/{document_id}/{filename}"
        object_store.upload_file(s3_key, file_data, content_type)

        # 3. Create document record in MongoDB
        doc_record = DocumentRecord(
            _id=document_id,
            engagement_id=engagement_id,
            source_system=source_system,
            source_path=s3_key,
            filename=filename,
            mime_type=content_type,
            file_size=len(file_data),
            doc_type=doc_type,
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
        await document_store.insert_document(
            doc_record.model_dump(by_alias=True)
        )

        # 4. Convert document via Docling (write to temp file first)
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(file_data)
            tmp.flush()
            docling_doc = convert_document(tmp.name, converter=_get_converter())

        # Store DoclingDocument JSON in MongoDB
        try:
            docling_json = docling_doc.model_dump() if hasattr(docling_doc, "model_dump") else None
        except Exception:
            docling_json = None
        if docling_json:
            await document_store.update_document_status(
                document_id, ProcessingStatus.PROCESSING, docling_json=docling_json
            )

        # 5. Chunk via HybridChunker
        chunk_dicts = chunk_document(docling_doc)

        if not chunk_dicts:
            await document_store.update_document_status(
                document_id, ProcessingStatus.FAILED
            )
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

        # 6. Generate embeddings
        texts = [c["content"] for c in chunk_dicts]
        embeddings = await embed_texts(texts)

        # 7. Prepare and store chunks in MongoDB + Neo4j
        mongo_chunks = []
        neo4j_chunks = []

        for idx, (chunk_dict, embedding) in enumerate(zip(chunk_dicts, embeddings)):
            chunk_id = str(uuid.uuid4())
            meta = chunk_dict.get("metadata", {})

            # MongoDB chunk record
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

            # Neo4j chunk data
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

        # Insert chunks to MongoDB
        await document_store.insert_chunks(mongo_chunks)

        # Upsert chunks to Neo4j (vector + fulltext)
        await neo4j_store.upsert_chunks_batch(neo4j_chunks)

        # 8. Create Document node in Neo4j and link chunks
        await neo4j_store.upsert_document_node(
            document_id=document_id,
            engagement_id=engagement_id,
            filename=filename,
            doc_type=doc_type,
        )

        # Update document status
        await document_store.update_document_status(
            document_id, ProcessingStatus.COMPLETED
        )

        # Update job
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
        await document_store.update_document_status(
            document_id, ProcessingStatus.FAILED
        )
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
