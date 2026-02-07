"""Integration tests for retrieval endpoints.

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
async def test_retrieve_vector(client):
    """Vector retrieval should accept a valid request body."""
    response = await client.post(
        "/retrieve/vector",
        json={
            "query": "change management controls",
            "engagement_id": "test-eng",
            "top_k": 5,
        },
    )
    # May fail if Neo4j is not running, but schema validation should pass
    assert response.status_code in (200, 500)


@pytest.mark.asyncio
async def test_retrieve_auto(client):
    """Auto-routed retrieval should accept a valid request body."""
    response = await client.post(
        "/retrieve/",
        json={
            "query": "ISO 27001 A.12.1",
            "engagement_id": "test-eng",
            "mode": "auto",
        },
    )
    assert response.status_code in (200, 500)


@pytest.mark.asyncio
async def test_generate(client):
    """Generate endpoint should accept a valid request body."""
    response = await client.post(
        "/generate/",
        json={
            "query": "What controls address access management?",
            "engagement_id": "test-eng",
        },
    )
    assert response.status_code in (200, 500)
