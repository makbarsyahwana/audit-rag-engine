import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import (
    connectors,
    generate,
    health,
    ingest,
    observability,
    retrieve,
    rlm,
    workflow,
)
from src.ingestion.task_broker import task_broker
from src.observability.logging import setup_logging
from src.observability.metrics import add_metrics_middleware
from src.observability.model_cards import register_default_models
from src.observability.tracing import instrument_fastapi, setup_tracing, shutdown_tracing
from src.security.kill_switch import kill_switch
from src.security.prompt_integrity import prompt_integrity
from src.security.service_auth import ServiceAuthMiddleware
from src.security.token_budget import token_budget
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

    # Connect to RabbitMQ
    try:
        await task_broker.connect()
    except Exception as exc:
        logger.warning("RabbitMQ not available (async ingestion disabled): %s", exc)

    # Register model cards
    register_default_models()

    # Initialize tracing
    setup_tracing(otlp_endpoint=os.getenv("OTLP_ENDPOINT"))

    # Connect kill switch + token budget to Redis (ASI09 + ASI02)
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        await kill_switch.connect(redis_url)
        await token_budget.connect(redis_url)
    except Exception as exc:
        logger.warning("Redis not available (kill switch in-memory): %s", exc)

    # Register prompt template fingerprints (ASI04)
    from src.generation.prompts.qa import QA_SYSTEM_PROMPT, QA_USER_PROMPT

    prompt_integrity.register("qa_system", QA_SYSTEM_PROMPT)
    prompt_integrity.register("qa_user", QA_USER_PROMPT)

    logger.info("All connections established. Audit RAG Engine ready.")
    yield

    # Shutdown: close connections
    logger.info("Shutting down Audit RAG Engine...")
    shutdown_tracing()
    await kill_switch.close()
    await token_budget.close()
    await task_broker.close()
    await neo4j_store.close()
    await document_store.close()
    logger.info("Audit RAG Engine stopped.")


app = FastAPI(
    title="Audit RAG Engine",
    description="Document ingestion, hybrid retrieval, and LLM generation for AI Audit Assistant",
    version="0.1.0",
    lifespan=lifespan,
)

# Security: service-to-service authentication (ASI03)
app.add_middleware(ServiceAuthMiddleware)

# Metrics middleware (auto-tracks request count + latency)
add_metrics_middleware(app)

# OpenTelemetry auto-instrumentation (if SDK installed)
instrument_fastapi(app)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
app.include_router(retrieve.router, prefix="/retrieve", tags=["retrieval"])
app.include_router(generate.router, prefix="/generate", tags=["generation"])
app.include_router(rlm.router, prefix="/rlm", tags=["rlm"])
app.include_router(workflow.router, prefix="/workflow", tags=["workflow"])
app.include_router(observability.router, prefix="/ops", tags=["observability"])
app.include_router(connectors.router, prefix="/connectors", tags=["connectors"])
