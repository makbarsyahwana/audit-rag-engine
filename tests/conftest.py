"""Shared pytest fixtures and configuration."""

import os

# Set test environment variables before any imports
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017/audit_rag_test")
