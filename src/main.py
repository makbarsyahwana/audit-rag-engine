from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import health, ingest, retrieve, generate


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize DB connections
    print("Starting Audit RAG Engine...")
    yield
    # Shutdown: close connections
    print("Shutting down Audit RAG Engine...")


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
