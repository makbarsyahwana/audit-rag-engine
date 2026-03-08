# Audit RAG Engine

Document ingestion, hybrid retrieval, and LLM generation service for the AI Audit Assistant.

## Tech Stack

- **Python 3.11+** with **FastAPI**
- **Docling** — unified document processing (PDF, DOCX, PPTX, XLSX, HTML, images)
- **Neo4j 5.26** — unified vector (HNSW) + fulltext (Lucene) + knowledge graph
- **MongoDB** — document & chunk storage
- **MinIO / S3** — original file storage
- **LangChain** + **OpenAI** — embeddings & generation
- **RestrictedPython** — sandboxed code execution for RLM engine
- **RabbitMQ** — async ingestion via custom `src/lib/amqp` client
- **OpenTelemetry** — distributed tracing & metrics
- **Prometheus** — metrics export

## Features

- **Ingestion Pipeline**: Docling convert → HybridChunker → embed → Neo4j + MongoDB upsert (async via RabbitMQ)
- **Knowledge Graph Extraction**: LLM-based entity/relationship extraction with audit-domain schema
- **Hybrid Retrieval**: 6 modes (vector, fulltext, graph, hybrid, entity_vector, graph_vector_fulltext) with query routing & reranking
- **Generation**: LLM Q&A with citation extraction, confidence scoring, and abstention detection
- **Multi-Mode Prompts**: Mode-aware prompt routing for Audit, Legal, and Compliance modes
- **Workflow Support**: prompt templates for workpaper/finding drafting (audit), legal memo/issue analysis (legal), gap analysis/compliance findings (compliance)
- **RLM Engine**: Recursive Language Model with sandboxed REPL, sub_RLM recursion, and mode-aware context
- **Multi-Model Tier Routing**: small (🟢 classification), mid (🟡 planner/critic), frontier (🔴 synthesis) model selection
- **OWASP Security**: 12 defense-in-depth modules (prompt guard, output guard, kill switch, circuit breaker, token budget, etc.)
- **Custom AMQP Client**: In-house async RabbitMQ client replacing aio-pika
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
├── api/routes/              # FastAPI route handlers
│   ├── ingest.py            # Ingestion (async via RabbitMQ)
│   ├── retrieve.py          # 6 retrieval modes
│   ├── generate.py          # Mode-aware generation with citations
│   ├── rlm.py               # RLM deep-analysis endpoint
│   └── health.py            # Health + kill switch + prompt integrity
├── ingestion/               # Docling pipeline + async stages
├── connectors/              # Source connectors (SharePoint, GRC, upload)
├── retrieval/               # 6 retrieval modes + query router + reranker
├── generation/              # LLM client + mode-aware prompts + citations
│   └── prompts/             # Audit, Legal, Compliance prompt templates
├── graph/                   # Entity/relationship extraction + KG schema
├── stores/                  # Neo4j, MongoDB, S3/MinIO clients
├── models/                  # Pydantic request/response models
├── rlm/                     # Recursive Language Model engine + sandbox
├── security/                # 12 OWASP ASI defense-in-depth modules
├── lib/amqp/                # Custom async AMQP client
├── config.py                # Pydantic settings (env, tiers, security)
└── main.py                  # FastAPI entry point
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
