import logging
import uuid
from datetime import UTC, datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from src.ingestion.pipeline import run_ingestion_pipeline
from src.ingestion.task_broker import QUEUE_PROCESS, task_broker
from src.models.document import IngestResponse, JobStatusResponse, ProcessingStatus
from src.security.file_validator import validate_file
from src.security.service_auth import parse_identity, verify_engagement_access
from src.stores.document_store import document_store
from src.stores.object_store import object_store

logger = logging.getLogger(__name__)

router = APIRouter()


async def _enqueue_document(
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
    """Upload file to S3 and publish ingestion job to RabbitMQ.

    Returns immediately with job_id and document_id.
    """
    job_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())

    # Upload to S3
    s3_key = f"{engagement_id}/{document_id}/{filename}"
    object_store.upload_file(s3_key, file_data, content_type)

    # Create job tracking record
    await document_store.create_job({
        "job_id": job_id,
        "document_id": document_id,
        "filename": filename,
        "status": ProcessingStatus.PENDING,
        "started_at": datetime.now(UTC),
        "chunks_created": 0,
        "entities_extracted": 0,
        "stage": "queued",
    })

    # Publish to RabbitMQ process queue
    await task_broker.publish(QUEUE_PROCESS, {
        "job_id": job_id,
        "document_id": document_id,
        "engagement_id": engagement_id,
        "s3_key": s3_key,
        "filename": filename,
        "doc_type": doc_type,
        "confidentiality_level": confidentiality_level,
        "source_system": source_system,
        "title": title,
        "tags": tags or [],
        "framework": framework,
        "clause_id": clause_id,
        "control_id": control_id,
        "content_type": content_type,
    })

    logger.info("Enqueued ingestion: job=%s, file=%s", job_id, filename)
    return {
        "job_id": job_id,
        "document_id": document_id,
        "filename": filename,
        "status": "queued",
    }


@router.post("/document", response_model=IngestResponse)
async def ingest_document(
    req: Request,
    file: UploadFile = File(...),
    engagement_id: str = Form(...),
    doc_type: str = Form("other"),
    confidentiality_level: str = Form("internal"),
    source_system: str = Form("upload"),
    title: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    framework: Optional[str] = Form(None),
    clause_id: Optional[str] = Form(None),
    control_id: Optional[str] = Form(None),
    sync: bool = Form(False),
):
    """Ingest a single document.

    By default, uploads to S3 and enqueues for async processing via RabbitMQ.
    Set sync=true to process synchronously (for testing/small files).
    """
    # Security: verify engagement access (ASI03)
    identity = parse_identity(req)
    verify_engagement_access(identity, engagement_id)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_data = await file.read()
    if not file_data:
        raise HTTPException(status_code=400, detail="Empty file")

    # Security: validate file (ASI05)
    validation = validate_file(
        file_data=file_data,
        filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
    )
    if not validation.valid:
        raise HTTPException(status_code=400, detail=validation.reason)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    if sync:
        # Synchronous fallback (original behavior)
        result = await run_ingestion_pipeline(
            file_data=file_data,
            filename=file.filename,
            engagement_id=engagement_id,
            doc_type=doc_type,
            confidentiality_level=confidentiality_level,
            source_system=source_system,
            title=title,
            tags=tag_list,
            framework=framework,
            clause_id=clause_id,
            control_id=control_id,
            content_type=file.content_type or "application/octet-stream",
        )
        status = ProcessingStatus.COMPLETED
        if result.get("status") == "failed":
            status = ProcessingStatus.FAILED
        elif result.get("status") == "duplicate":
            status = ProcessingStatus.COMPLETED
        return IngestResponse(
            job_id=result["job_id"],
            document_id=result["document_id"],
            filename=file.filename,
            status=status,
            message=result.get("message", ""),
        )

    # Async path: enqueue to RabbitMQ
    result = await _enqueue_document(
        file_data=file_data,
        filename=file.filename,
        engagement_id=engagement_id,
        doc_type=doc_type,
        confidentiality_level=confidentiality_level,
        source_system=source_system,
        title=title,
        tags=tag_list,
        framework=framework,
        clause_id=clause_id,
        control_id=control_id,
        content_type=file.content_type or "application/octet-stream",
    )

    return IngestResponse(
        job_id=result["job_id"],
        document_id=result["document_id"],
        filename=file.filename,
        status=ProcessingStatus.PENDING,
        message="Queued for async processing",
    )


@router.post("/batch")
async def ingest_batch(
    files: list[UploadFile] = File(...),
    engagement_id: str = Form(...),
    doc_type: str = Form("other"),
    sync: bool = Form(False),
):
    """Batch document ingestion.

    By default, enqueues all files for async processing via RabbitMQ.
    Set sync=true to process synchronously.
    """
    results = []
    for file in files:
        if not file.filename:
            continue
        file_data = await file.read()
        if not file_data:
            continue

        if sync:
            result = await run_ingestion_pipeline(
                file_data=file_data,
                filename=file.filename,
                engagement_id=engagement_id,
                doc_type=doc_type,
                content_type=file.content_type or "application/octet-stream",
            )
        else:
            result = await _enqueue_document(
                file_data=file_data,
                filename=file.filename,
                engagement_id=engagement_id,
                doc_type=doc_type,
                content_type=file.content_type or "application/octet-stream",
            )
        results.append(result)

    return {
        "total": len(results),
        "queued": sum(1 for r in results if r.get("status") == "queued"),
        "completed": sum(1 for r in results if r.get("status") == "completed"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "duplicates": sum(1 for r in results if r.get("status") == "duplicate"),
        "results": results,
    }


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def ingestion_status(job_id: str):
    """Check ingestion job status."""
    job = await document_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job["job_id"],
        document_id=job.get("document_id"),
        status=job.get("status", ProcessingStatus.PENDING),
        filename=job.get("filename"),
        chunks_created=job.get("chunks_created", 0),
        entities_extracted=job.get("entities_extracted", 0),
        error=job.get("error"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
    )


@router.get("/queue-stats")
async def queue_stats():
    """Get RabbitMQ queue statistics for monitoring."""
    stats = await task_broker.get_queue_stats()
    return {"queues": stats}
