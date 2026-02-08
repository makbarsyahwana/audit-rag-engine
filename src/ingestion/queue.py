"""Redis-based message queue for async document ingestion.

Uses Redis Streams for reliable, at-least-once delivery with consumer
groups. Failed jobs are retried via pending entry list (PEL).
"""

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any, Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

STREAM_NAME = "ingestion:jobs"
CONSUMER_GROUP = "ingestion-workers"
DEAD_LETTER_STREAM = "ingestion:dead-letter"
MAX_RETRIES = 3


class IngestionQueue:
    """Async ingestion queue backed by Redis Streams."""

    def __init__(self) -> None:
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis = aioredis.from_url(
            redis_url, decode_responses=True
        )
        # Create consumer group if it doesn't exist
        try:
            await self._redis.xgroup_create(
                STREAM_NAME, CONSUMER_GROUP, id="0", mkstream=True
            )
            logger.info("Created consumer group: %s", CONSUMER_GROUP)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                logger.debug("Consumer group already exists")
            else:
                raise
        logger.info("Ingestion queue connected")

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    @property
    def redis(self) -> aioredis.Redis:
        if self._redis is None:
            raise RuntimeError("Queue not connected. Call connect() first.")
        return self._redis

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        filename: str,
        engagement_id: str,
        s3_key: str,
        doc_type: str = "other",
        source_system: str = "upload",
        content_type: str = "application/octet-stream",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Enqueue a document for async ingestion.

        The file should already be uploaded to S3. The worker will
        download it from S3 and run the ingestion pipeline.

        Returns:
            The Redis stream message ID.
        """
        payload = {
            "filename": filename,
            "engagement_id": engagement_id,
            "s3_key": s3_key,
            "doc_type": doc_type,
            "source_system": source_system,
            "content_type": content_type,
            "metadata": json.dumps(metadata or {}),
            "enqueued_at": datetime.now(UTC).isoformat(),
            "retries": "0",
        }
        msg_id = await self.redis.xadd(STREAM_NAME, payload)
        logger.info(
            "Enqueued ingestion job: %s (file=%s, engagement=%s)",
            msg_id,
            filename,
            engagement_id,
        )
        return msg_id

    # ------------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------------

    async def consume(
        self,
        consumer_name: str,
        count: int = 1,
        block_ms: int = 5000,
    ) -> list[tuple[str, dict[str, str]]]:
        """Read new messages from the stream.

        Args:
            consumer_name: Unique name for this consumer instance.
            count: Max messages to read per call.
            block_ms: How long to block waiting for messages.

        Returns:
            List of (message_id, payload) tuples.
        """
        results = await self.redis.xreadgroup(
            groupname=CONSUMER_GROUP,
            consumername=consumer_name,
            streams={STREAM_NAME: ">"},
            count=count,
            block=block_ms,
        )

        messages = []
        if results:
            for _stream, entries in results:
                for msg_id, data in entries:
                    messages.append((msg_id, data))
        return messages

    async def acknowledge(self, message_id: str) -> None:
        """Acknowledge a successfully processed message."""
        await self.redis.xack(STREAM_NAME, CONSUMER_GROUP, message_id)
        logger.debug("Acknowledged message: %s", message_id)

    async def retry_or_dead_letter(
        self, message_id: str, payload: dict[str, str], error: str
    ) -> None:
        """Retry a failed message or move to dead letter stream."""
        retries = int(payload.get("retries", "0")) + 1

        if retries >= MAX_RETRIES:
            # Move to dead letter stream
            dead_payload = {
                **payload,
                "retries": str(retries),
                "error": error,
                "failed_at": datetime.now(UTC).isoformat(),
            }
            await self.redis.xadd(DEAD_LETTER_STREAM, dead_payload)
            await self.redis.xack(
                STREAM_NAME, CONSUMER_GROUP, message_id
            )
            logger.warning(
                "Message %s moved to dead letter after %d retries: %s",
                message_id,
                retries,
                error,
            )
        else:
            # Re-enqueue with incremented retry count
            retry_payload = {
                **payload,
                "retries": str(retries),
                "last_error": error,
            }
            await self.redis.xadd(STREAM_NAME, retry_payload)
            await self.redis.xack(
                STREAM_NAME, CONSUMER_GROUP, message_id
            )
            logger.info(
                "Message %s re-enqueued (retry %d/%d)",
                message_id,
                retries,
                MAX_RETRIES,
            )

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get queue length and consumer group info."""
        stream_len = await self.redis.xlen(STREAM_NAME)
        dead_letter_len = await self.redis.xlen(DEAD_LETTER_STREAM)

        # Consumer group info
        groups = await self.redis.xinfo_groups(STREAM_NAME)
        group_info = {}
        for grp in groups:
            group_info[grp["name"]] = {
                "consumers": grp.get("consumers", 0),
                "pending": grp.get("pending", 0),
                "last_delivered_id": grp.get(
                    "last-delivered-id", "0-0"
                ),
            }

        return {
            "stream_length": stream_len,
            "dead_letter_length": dead_letter_len,
            "consumer_groups": group_info,
        }


# Global queue instance
ingestion_queue = IngestionQueue()
