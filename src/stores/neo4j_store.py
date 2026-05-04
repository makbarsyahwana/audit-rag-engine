"""Backwards-compatible re-export — all logic now lives in src.stores.neo4j.*."""

from src.stores.neo4j import Neo4jStore, neo4j_store

__all__ = ["Neo4jStore", "neo4j_store"]
