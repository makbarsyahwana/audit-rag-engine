"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Neo4j (unified vector + graph + fulltext)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "audit_pass"

    # MongoDB (document store)
    mongodb_uri: str = "mongodb://audit_user:audit_pass@localhost:27017/audit_rag?authSource=admin"
    mongodb_database: str = "audit_rag"

    # S3 / MinIO (object storage)
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "audit_minio"
    s3_secret_key: str = "audit_minio_pass"
    s3_bucket: str = "audit-documents"

    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Chunking
    chunk_max_tokens: int = 512
    chunk_merge_peers: bool = True

    # RabbitMQ
    rabbitmq_url: str = "amqp://audit_user:audit_pass@localhost:5672/audit"

    # Entity extraction
    entity_extraction_enabled: bool = True
    entity_embedding_enabled: bool = True

    # Retrieval defaults
    retrieval_top_k: int = 10
    rerank_enabled: bool = False

    # Security (ASI03 — service-to-service auth)
    service_auth_token: str = ""  # Set via SERVICE_AUTH_TOKEN env var

    # Security (ASI09 — kill switch)
    redis_url: str = "redis://localhost:6379"

    # Security (ASI02 — token budget)
    llm_request_timeout: int = 60  # seconds
    llm_daily_token_budget: int = 0  # 0 = unlimited
    llm_circuit_breaker_threshold: int = 5  # consecutive failures before open
    llm_circuit_breaker_reset: int = 60  # seconds before half-open

    # Security (ASI05 — file upload limits)
    max_upload_size_mb: int = 50
    allowed_mime_types: str = (
        "application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,"
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
        "application/vnd.openxmlformats-officedocument.presentationml.presentation,"
        "text/plain,text/csv,application/json,text/markdown"
    )

    # Model tiers — multi-model routing
    # 🟢 Small tier (classification, sub_RLM leaf calls)
    small_model_name: str = ""  # e.g. "qwen3-8b"; empty = use llm_model
    small_model_base_url: str = ""  # e.g. "http://localhost:8080/v1"; empty = use OpenAI
    small_model_api_key: str = ""  # empty = use openai_api_key
    # 🟡 Mid tier (planner, RLM controller, critic)
    mid_model_name: str = ""  # e.g. "deepseek-v3"; empty = use llm_model
    mid_model_base_url: str = ""  # empty = use OpenAI
    mid_model_api_key: str = ""  # empty = use openai_api_key
    # 🔴 Frontier tier (synthesis, final polish)
    frontier_model_name: str = ""  # e.g. "claude-sonnet-4-20250514"; empty = use llm_model
    frontier_model_base_url: str = ""  # empty = use OpenAI
    frontier_model_api_key: str = ""  # empty = use openai_api_key

    # RLM (Recursive Language Model) engine
    rlm_max_iterations: int = 15
    rlm_max_depth: int = 3
    rlm_max_sub_calls: int = 50
    rlm_code_timeout_seconds: int = 30
    rlm_controller_model: str = ""  # 🟡 mid-tier; empty = use mid_model_name or llm_model
    rlm_sub_model: str = ""  # 🟢 small; empty = use small_model_name or llm_model
    rlm_synthesis_model: str = ""  # 🔴 frontier; empty = use frontier_model_name or llm_model

    # Server
    host: str = "0.0.0.0"
    port: int = 8001

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
