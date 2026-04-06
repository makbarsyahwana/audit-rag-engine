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

    # Corpus scope — external/internal document separation
    global_engagement_id: str = "__global__"  # Reserved engagement for external/public documents

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

    # Provider abstraction — shared convenience key (OpenRouter / any single-key gateway)
    openrouter_api_key: str = ""  # fallback for all tier api_keys when tier key is empty

    # Model tiers — multi-model routing
    # 🟢 Small tier (classification, sub_RLM leaf calls)
    small_model_provider: str = "openai_compatible"  # openai_compatible | anthropic | google | ollama  # noqa: E501
    small_model_name: str = ""  # e.g. "qwen/qwen3-8b"; empty = use llm_model
    small_model_base_url: str = ""  # e.g. "https://openrouter.ai/api/v1"; empty = use OpenAI
    small_model_api_key: str = ""  # empty = openrouter_api_key → openai_api_key
    # 🟡 Mid tier (planner, RLM controller, critic)
    mid_model_provider: str = "openai_compatible"
    mid_model_name: str = ""  # e.g. "deepseek/deepseek-chat-v3-0324"; empty = use llm_model
    mid_model_base_url: str = ""  # empty = use OpenAI
    mid_model_api_key: str = ""  # empty = openrouter_api_key → openai_api_key
    # 🔴 Frontier tier (synthesis, final polish)
    frontier_model_provider: str = "openai_compatible"
    frontier_model_name: str = ""  # e.g. "anthropic/claude-sonnet-4-5"; empty = use llm_model
    frontier_model_base_url: str = ""  # empty = use OpenAI
    frontier_model_api_key: str = ""  # empty = openrouter_api_key → openai_api_key

    # Embeddings provider (OpenAI-compatible or future native)
    embedding_provider: str = "openai_compatible"
    embedding_base_url: str = ""  # e.g. "https://openrouter.ai/api/v1"
    embedding_api_key: str = ""  # empty = openrouter_api_key → openai_api_key

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
