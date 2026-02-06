from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def generate_answer():
    """Generate answer with citations from retrieved context."""
    # TODO: Phase 1
    return {"status": "not_implemented"}
