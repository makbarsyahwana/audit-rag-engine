import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.ingestion.pipeline import run_ingestion_pipeline
from src.models.document import IngestResponse, JobStatusResponse, ProcessingStatus
from src.stores.document_store import document_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/document", response_model=IngestResponse)
async def ingest_document(
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
):
    """Ingest a single document via Docling pipeline.

    Accepts a file upload with metadata. Runs the full pipeline:
    convert → chunk → embed → Neo4j upsert → MongoDB storage.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_data = await file.read()
    if not file_data:
        raise HTTPException(status_code=400, detail="Empty file")

    # Parse tags from comma-separated string
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

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


@router.post("/batch")
async def ingest_batch(
    files: list[UploadFile] = File(...),
    engagement_id: str = Form(...),
    doc_type: str = Form("other"),
):
    """Batch document ingestion. Processes multiple files sequentially."""
    results = []
    for file in files:
        if not file.filename:
            continue
        file_data = await file.read()
        if not file_data:
            continue

        result = await run_ingestion_pipeline(
            file_data=file_data,
            filename=file.filename,
            engagement_id=engagement_id,
            doc_type=doc_type,
            content_type=file.content_type or "application/octet-stream",
        )
        results.append(result)

    return {
        "total": len(results),
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
