"""Integration tests for ingestion endpoints.

These tests require running infrastructure (Neo4j, MongoDB, MinIO).
Run with: pytest tests/integration/ -v
"""

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app


@pytest.fixture
async def client():
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client):
    """Health endpoint should return ok."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "audit-rag-engine"


@pytest.mark.asyncio
async def test_ingest_no_file(client):
    """Ingest without file should return 422."""
    response = await client.post(
        "/ingest/document",
        data={"engagement_id": "test-eng", "doc_type": "policy"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ingest_status_not_found(client):
    """Non-existent job ID should return 404."""
    response = await client.get("/ingest/status/non-existent-job-id")
    assert response.status_code == 404
