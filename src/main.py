import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import generate, health, ingest, observability, retrieve, workflow
from src.observability.logging import setup_logging
from src.observability.metrics import add_metrics_middleware
from src.observability.model_cards import register_default_models
from src.observability.tracing import instrument_fastapi, setup_tracing, shutdown_tracing
from src.stores.document_store import document_store
from src.stores.neo4j_store import neo4j_store
from src.stores.object_store import object_store

# Structured logging (JSON in production, plain in dev)
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_format=os.getenv("LOG_FORMAT", "json") == "json",
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

    # Register model cards
    register_default_models()

    # Initialize tracing
    setup_tracing(otlp_endpoint=os.getenv("OTLP_ENDPOINT"))

    logger.info("All connections established. Audit RAG Engine ready.")
    yield

    # Shutdown: close connections
    logger.info("Shutting down Audit RAG Engine...")
    shutdown_tracing()
    await neo4j_store.close()
    await document_store.close()
    logger.info("Audit RAG Engine stopped.")


app = FastAPI(
    title="Audit RAG Engine",
    description="Document ingestion, hybrid retrieval, and LLM generation for AI Audit Assistant",
    version="0.1.0",
    lifespan=lifespan,
)

# Metrics middleware (auto-tracks request count + latency)
add_metrics_middleware(app)

# OpenTelemetry auto-instrumentation (if SDK installed)
instrument_fastapi(app)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
app.include_router(retrieve.router, prefix="/retrieve", tags=["retrieval"])
app.include_router(generate.router, prefix="/generate", tags=["generation"])
app.include_router(workflow.router, prefix="/workflow", tags=["workflow"])
app.include_router(observability.router, prefix="/ops", tags=["observability"])
