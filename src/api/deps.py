"""Dependency injection for FastAPI routes."""

from src.config import Settings, settings
from src.stores.document_store import DocumentStore, document_store
from src.stores.neo4j_store import Neo4jStore, neo4j_store
from src.stores.object_store import ObjectStore, object_store


def get_settings() -> Settings:
    """Return application settings."""
    return settings


def get_neo4j() -> Neo4jStore:
    """Return the Neo4j store singleton."""
    return neo4j_store


def get_document_store() -> DocumentStore:
    """Return the MongoDB document store singleton."""
    return document_store


def get_object_store() -> ObjectStore:
    """Return the S3 object store singleton."""
    return object_store
