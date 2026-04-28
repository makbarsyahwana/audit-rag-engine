"""LLM token budget and rate limiting (ASI02 — Tool Misuse).

Tracks per-engagement daily token usage and enforces budgets
to prevent runaway LLM costs.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """Daily token usage for an engagement."""

    date: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    request_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenBudget:
    """In-memory token budget tracker (Redis-upgradeable)."""

    def __init__(self) -> None:
        self._usage: dict[str, UsageRecord] = {}
        self._redis: Optional[object] = None

    async def connect(self, redis_url: str) -> None:
        """Optionally connect to Redis for distributed tracking."""
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                redis_url, decode_responses=True
            )
        except Exception as exc:
            logger.debug("Token budget using in-memory (no Redis): %s", exc)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()  # type: ignore[union-attr]
            self._redis = None

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d")

    def _key(self, engagement_id: str) -> str:
        return f"{engagement_id}:{self._today()}"

    async def record_usage(
        self,
        engagement_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record token usage for an engagement."""
        key = self._key(engagement_id)

        if self._redis:
            try:
                pipe = self._redis.pipeline()  # type: ignore[union-attr]
                pipe.hincrby(f"budget:{key}", "prompt", prompt_tokens)
                pipe.hincrby(f"budget:{key}", "completion", completion_tokens)
                pipe.hincrby(f"budget:{key}", "requests", 1)
                pipe.expire(f"budget:{key}", 86400 * 2)  # 2-day TTL
                await pipe.execute()
                return
            except Exception as exc:
                logger.warning("Redis record_usage failed, falling back to in-memory: %s", exc)

        # Fallback: in-memory
        if key not in self._usage:
            self._usage[key] = UsageRecord(date=self._today())
        rec = self._usage[key]
        rec.prompt_tokens += prompt_tokens
        rec.completion_tokens += completion_tokens
        rec.request_count += 1

    async def get_usage(self, engagement_id: str) -> UsageRecord:
        """Get today's usage for an engagement."""
        key = self._key(engagement_id)

        if self._redis:
            try:
                data = await self._redis.hgetall(  # type: ignore[union-attr]
                    f"budget:{key}"
                )
                if data:
                    return UsageRecord(
                        date=self._today(),
                        prompt_tokens=int(data.get("prompt", 0)),
                        completion_tokens=int(data.get("completion", 0)),
                        request_count=int(data.get("requests", 0)),
                    )
            except Exception as exc:
                logger.warning("Redis get_usage failed, falling back to in-memory: %s", exc)

        return self._usage.get(key, UsageRecord(date=self._today()))

    async def check_budget(self, engagement_id: str) -> bool:
        """Check if engagement is within daily token budget.

        Returns True if allowed, False if budget exceeded.
        """
        budget = settings.llm_daily_token_budget
        if budget <= 0:
            return True  # Unlimited

        usage = await self.get_usage(engagement_id)
        if usage.total_tokens >= budget:
            logger.warning(
                "Token budget exceeded for engagement %s: "
                "%d/%d tokens used",
                engagement_id,
                usage.total_tokens,
                budget,
            )
            return False
        return True


token_budget = TokenBudget()
