from fastapi import APIRouter, File, Form, UploadFile

router = APIRouter()


@router.post("/document")
async def ingest_document(
    file: UploadFile = File(...),
    engagement_id: str = Form(...),
    doc_type: str = Form(...),
):
    """Ingest a single document via Docling pipeline."""
    # TODO: Phase 1 - Implement Docling convert → chunk → embed → Neo4j upsert
    return {
        "status": "accepted",
        "filename": file.filename,
        "engagement_id": engagement_id,
        "doc_type": doc_type,
    }


@router.post("/batch")
async def ingest_batch():
    """Batch document ingestion."""
    # TODO: Phase 1
    return {"status": "not_implemented"}


@router.get("/status/{job_id}")
async def ingestion_status(job_id: str):
    """Check ingestion job status."""
    # TODO: Phase 1
    return {"job_id": job_id, "status": "not_implemented"}
