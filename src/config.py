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

    # Retrieval defaults
    retrieval_top_k: int = 10
    rerank_enabled: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8001

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
