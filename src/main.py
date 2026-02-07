import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import generate, health, ingest, retrieve, workflow
from src.stores.document_store import document_store
from src.stores.neo4j_store import neo4j_store
from src.stores.object_store import object_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize DB connections
    logger.info("Starting Audit RAG Engine...")

    # Connect to Neo4j and ensure indexes
    await neo4j_store.connect()
    await neo4j_store.ensure_indexes()

    # Connect to MongoDB and ensure indexes
    await document_store.connect()

    # Connect to S3/MinIO
    object_store.connect()

    logger.info("All connections established. Audit RAG Engine ready.")
    yield

    # Shutdown: close connections
    logger.info("Shutting down Audit RAG Engine...")
    await neo4j_store.close()
    await document_store.close()
    logger.info("Audit RAG Engine stopped.")


app = FastAPI(
    title="Audit RAG Engine",
    description="Document ingestion, hybrid retrieval, and LLM generation for AI Audit Assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
app.include_router(retrieve.router, prefix="/retrieve", tags=["retrieval"])
app.include_router(generate.router, prefix="/generate", tags=["generation"])
app.include_router(workflow.router, prefix="/workflow", tags=["workflow"])
