# Audit RAG Engine

Document ingestion, hybrid retrieval, and LLM generation service for the AI Audit Assistant.

## Tech Stack

- **Python 3.11+** with **FastAPI**
- **Docling** — unified document processing (PDF, DOCX, PPTX, XLSX, HTML, images)
- **Neo4j 5.26** — unified vector (HNSW) + fulltext (Lucene) + knowledge graph
- **MongoDB** — document & chunk storage
- **MinIO / S3** — original file storage
- **LangChain** + **OpenAI** — embeddings & generation
- **OpenTelemetry** — distributed tracing & metrics
- **Prometheus** — metrics export

## Features

- **Ingestion Pipeline**: Docling convert → HybridChunker → embed → Neo4j + MongoDB upsert
- **Knowledge Graph Extraction**: LLM-based entity/relationship extraction with audit-domain schema
- **Hybrid Retrieval**: vector, fulltext, graph, and combined hybrid modes with query routing & reranking
- **Generation**: LLM Q&A with citation extraction, confidence scoring, and abstention detection
- **Workflow Support**: prompt templates for workpaper/finding drafting, evidence search, traceability matrix
- **Observability**: structured JSON logging, Prometheus metrics, OpenTelemetry tracing, model cards

## Getting Started

### Prerequisites

- Python 3.11+
- Running infrastructure (see parent `docker-compose.yml`)

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment config
cp .env.example .env
# Edit .env with your API keys and connection strings

# Initialize Neo4j indexes
python scripts/init_indexes.py

# Start dev server
uvicorn src.main:app --reload --port 8001
```

### API Docs

Once running, visit [http://localhost:8001/docs](http://localhost:8001/docs) for interactive Swagger documentation.

## Project Structure

```
src/
├── api/
│   ├── routes/          # FastAPI route handlers
│   │   ├── health.py    # Health check
│   │   ├── ingest.py    # Document ingestion endpoints
│   │   ├── retrieve.py  # Retrieval endpoints (vector/fulltext/graph/hybrid)
│   │   ├── generate.py  # LLM generation with citations
│   │   ├── workflow.py  # Workflow support endpoints
│   │   └── observability.py  # Ops & metrics endpoints
│   └── deps.py          # Dependency injection
├── ingestion/
│   ├── pipeline.py      # Main ingestion pipeline orchestration
│   ├── converter.py     # Docling DocumentConverter wrapper
│   ├── chunker.py       # Docling HybridChunker
│   ├── embedder.py      # Embedding generation
│   ├── enrichments.py   # Multimodal enrichments config
│   └── connectors/      # Source connectors
├── retrieval/           # Retrieval modes and query routing
├── generation/          # LLM generation with citations
├── graph/               # Knowledge graph extraction
├── stores/              # Database clients (Neo4j, MongoDB, S3)
├── models/              # Pydantic models
├── observability/       # Logging, metrics, tracing, model cards
└── main.py              # FastAPI application entry point
```

## Testing

```bash
pytest
pytest --cov=src        # With coverage
```

## Environment Variables

See [`.env.example`](.env.example) for all available configuration options.

## License

[MIT](LICENSE)
