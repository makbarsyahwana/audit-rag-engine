"""Server-side kill switch enforcement (ASI09 — Human-Agent Trust).

Redis-backed kill switch that can halt all LLM generation requests.
Levels:
  - active: normal operation
  - soft_stop: finish current requests, reject new ones
  - hard_stop: reject all requests immediately
"""

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

KILL_SWITCH_KEY = "audit:kill_switch"
KILL_SWITCH_REASON_KEY = "audit:kill_switch:reason"


class KillSwitchLevel(str, Enum):
    ACTIVE = "active"
    SOFT_STOP = "soft_stop"
    HARD_STOP = "hard_stop"


class KillSwitch:
    """Redis-backed kill switch for LLM generation."""

    def __init__(self) -> None:
        self._redis: Optional[object] = None
        self._fallback_level = KillSwitchLevel.ACTIVE

    async def connect(self, redis_url: str) -> None:
        """Connect to Redis for kill switch state."""
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                redis_url, decode_responses=True
            )
            logger.info("Kill switch connected to Redis")
        except Exception as exc:
            logger.warning(
                "Kill switch Redis unavailable (using in-memory): %s",
                exc,
            )
            self._redis = None

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()  # type: ignore[union-attr]
            self._redis = None

    async def get_level(self) -> KillSwitchLevel:
        """Get current kill switch level."""
        if self._redis is None:
            return self._fallback_level
        try:
            val = await self._redis.get(KILL_SWITCH_KEY)  # type: ignore[union-attr]
            if val and val in KillSwitchLevel.__members__.values():
                return KillSwitchLevel(val)
            return KillSwitchLevel.ACTIVE
        except Exception as exc:
            logger.warning("Kill switch Redis read failed (get_level): %s", exc)
            return self._fallback_level

    async def get_reason(self) -> str:
        """Get the reason for the current kill switch state."""
        if self._redis is None:
            return ""
        try:
            return (
                await self._redis.get(KILL_SWITCH_REASON_KEY)  # type: ignore[union-attr]
                or ""
            )
        except Exception as exc:
            logger.warning("Kill switch Redis read failed (get_reason): %s", exc)
            return ""

    async def set_level(
        self, level: KillSwitchLevel, reason: str = ""
    ) -> None:
        """Set the kill switch level."""
        if self._redis is None:
            self._fallback_level = level
            logger.info(
                "Kill switch set to %s (in-memory): %s",
                level.value,
                reason,
            )
            return

        try:
            await self._redis.set(  # type: ignore[union-attr]
                KILL_SWITCH_KEY, level.value
            )
            if reason:
                await self._redis.set(  # type: ignore[union-attr]
                    KILL_SWITCH_REASON_KEY, reason
                )
            logger.info(
                "Kill switch set to %s: %s", level.value, reason
            )
        except Exception as exc:
            logger.error("Failed to set kill switch: %s", exc)
            self._fallback_level = level

    def is_generation_allowed(
        self, level: KillSwitchLevel
    ) -> bool:
        """Check if generation is allowed at the given level."""
        return level == KillSwitchLevel.ACTIVE


kill_switch = KillSwitch()
