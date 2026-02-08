"""Redis-based caching layer for retrieval and generation results.

Provides TTL-based caching with configurable key prefixes,
query normalization, and cache invalidation per engagement.
"""

import hashlib
import json
import logging
import os
from typing import Any, Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

DEFAULT_TTL = 3600  # 1 hour
RETRIEVAL_PREFIX = "cache:retrieval:"
GENERATION_PREFIX = "cache:generation:"
EMBEDDING_PREFIX = "cache:embedding:"


class CacheStore:
    """Async Redis cache for retrieval and generation results."""

    def __init__(self) -> None:
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis = aioredis.from_url(
            redis_url,
            decode_responses=True,
            max_connections=20,
        )
        logger.info("Cache store connected (pool_size=20)")

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    @property
    def redis(self) -> aioredis.Redis:
        if self._redis is None:
            raise RuntimeError("Cache not connected. Call connect() first.")
        return self._redis

    # ------------------------------------------------------------------
    # Generic cache operations
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Optional[Any]:
        """Get a cached value. Returns None on miss."""
        raw = await self.redis.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def set(
        self, key: str, value: Any, ttl: int = DEFAULT_TTL
    ) -> None:
        """Set a cached value with TTL in seconds."""
        serialized = json.dumps(value, default=str)
        await self.redis.set(key, serialized, ex=ttl)

    async def delete(self, key: str) -> None:
        await self.redis.delete(key)

    async def invalidate_prefix(self, prefix: str) -> int:
        """Delete all keys matching a prefix. Returns count deleted."""
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor=cursor, match=f"{prefix}*", count=100
            )
            if keys:
                await self.redis.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        logger.info("Invalidated %d keys with prefix %s", deleted, prefix)
        return deleted

    # ------------------------------------------------------------------
    # Retrieval cache
    # ------------------------------------------------------------------

    def retrieval_key(
        self,
        query: str,
        engagement_id: str,
        mode: str,
        top_k: int,
    ) -> str:
        """Build a deterministic cache key for a retrieval query."""
        normalized = query.strip().lower()
        content = f"{normalized}:{engagement_id}:{mode}:{top_k}"
        digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{RETRIEVAL_PREFIX}{digest}"

    async def get_retrieval(
        self,
        query: str,
        engagement_id: str,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> Optional[list[dict]]:
        key = self.retrieval_key(query, engagement_id, mode, top_k)
        return await self.get(key)

    async def set_retrieval(
        self,
        query: str,
        engagement_id: str,
        results: list[dict],
        mode: str = "hybrid",
        top_k: int = 10,
        ttl: int = DEFAULT_TTL,
    ) -> None:
        key = self.retrieval_key(query, engagement_id, mode, top_k)
        await self.set(key, results, ttl)

    # ------------------------------------------------------------------
    # Generation cache
    # ------------------------------------------------------------------

    def generation_key(
        self, query: str, engagement_id: str
    ) -> str:
        normalized = query.strip().lower()
        content = f"{normalized}:{engagement_id}"
        digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{GENERATION_PREFIX}{digest}"

    async def get_generation(
        self, query: str, engagement_id: str
    ) -> Optional[dict]:
        key = self.generation_key(query, engagement_id)
        return await self.get(key)

    async def set_generation(
        self,
        query: str,
        engagement_id: str,
        result: dict,
        ttl: int = DEFAULT_TTL,
    ) -> None:
        key = self.generation_key(query, engagement_id)
        await self.set(key, result, ttl)

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def embedding_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{EMBEDDING_PREFIX}{digest}"

    async def get_embedding(self, text: str) -> Optional[list[float]]:
        key = self.embedding_key(text)
        return await self.get(key)

    async def set_embedding(
        self,
        text: str,
        embedding: list[float],
        ttl: int = 86400,  # 24h
    ) -> None:
        key = self.embedding_key(text)
        await self.set(key, embedding, ttl)

    # ------------------------------------------------------------------
    # Engagement-scoped invalidation
    # ------------------------------------------------------------------

    async def invalidate_engagement(self, engagement_id: str) -> int:
        """Invalidate all cached data for an engagement.

        Called after new documents are ingested.
        """
        total = 0
        # Retrieval and generation caches include engagement_id
        # in the hash, but we can't reverse the hash. Instead, we
        # track engagement keys in a set.
        set_key = f"cache:engagement_keys:{engagement_id}"
        keys = await self.redis.smembers(set_key)
        if keys:
            await self.redis.delete(*keys)
            total = len(keys)
        await self.redis.delete(set_key)
        logger.info(
            "Invalidated %d cached entries for engagement %s",
            total,
            engagement_id,
        )
        return total

    async def track_engagement_key(
        self, engagement_id: str, cache_key: str
    ) -> None:
        """Track a cache key for engagement-scoped invalidation."""
        set_key = f"cache:engagement_keys:{engagement_id}"
        await self.redis.sadd(set_key, cache_key)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        info = await self.redis.info("memory")
        db_size = await self.redis.dbsize()
        return {
            "total_keys": db_size,
            "used_memory_human": info.get("used_memory_human", "?"),
            "connected_clients": info.get("connected_clients", 0),
            "maxmemory_human": info.get("maxmemory_human", "?"),
        }


# Global cache instance
cache_store = CacheStore()
