"""RabbitMQ-based task broker for async ingestion pipeline.

Provides publish/consume helpers with stage-based queues,
dead-letter exchange, and retry logic using aio-pika.
"""

import json
import logging
from typing import Any, Callable, Optional

from aio_pika import DeliveryMode, ExchangeType, Message, connect_robust
from aio_pika.abc import AbstractChannel, AbstractRobustConnection

from src.config import settings

logger = logging.getLogger(__name__)

# Queue names for each pipeline stage
QUEUE_PROCESS = "ingestion.process"
QUEUE_EXTRACT = "ingestion.extract"
QUEUE_POSTPROCESS = "ingestion.postprocess"

# Dead-letter exchange
DLX_EXCHANGE = "ingestion.dlx"
DLX_QUEUE = "ingestion.dead-letter"

ALL_QUEUES = [QUEUE_PROCESS, QUEUE_EXTRACT, QUEUE_POSTPROCESS]

MAX_RETRIES = 3


class TaskBroker:
    """Async RabbitMQ task broker for pipeline stages."""

    def __init__(self) -> None:
        self._connection: Optional[AbstractRobustConnection] = None
        self._channel: Optional[AbstractChannel] = None

    async def connect(self) -> None:
        """Connect to RabbitMQ and declare exchanges/queues."""
        self._connection = await connect_robust(settings.rabbitmq_url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=1)

        # Declare dead-letter exchange and queue
        dlx = await self._channel.declare_exchange(
            DLX_EXCHANGE, ExchangeType.FANOUT, durable=True
        )
        dlq = await self._channel.declare_queue(DLX_QUEUE, durable=True)
        await dlq.bind(dlx)

        # Declare stage queues with dead-letter routing
        for queue_name in ALL_QUEUES:
            await self._channel.declare_queue(
                queue_name,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": DLX_EXCHANGE,
                },
            )

        logger.info("TaskBroker connected to RabbitMQ, queues declared")

    async def close(self) -> None:
        """Close the RabbitMQ connection."""
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
            logger.info("TaskBroker connection closed")

    @property
    def channel(self) -> AbstractChannel:
        if self._channel is None:
            raise RuntimeError("TaskBroker not connected. Call connect() first.")
        return self._channel

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish(
        self,
        queue_name: str,
        payload: dict[str, Any],
        retry_count: int = 0,
    ) -> None:
        """Publish a message to a specific stage queue.

        Args:
            queue_name: Target queue name.
            payload: JSON-serializable dict.
            retry_count: Current retry attempt (stored in headers).
        """
        body = json.dumps(payload).encode()
        message = Message(
            body=body,
            delivery_mode=DeliveryMode.PERSISTENT,
            content_type="application/json",
            headers={"x-retry-count": retry_count},
        )
        await self.channel.default_exchange.publish(
            message, routing_key=queue_name
        )
        logger.debug("Published to %s: %s", queue_name, payload.get("job_id", "?"))

    # ------------------------------------------------------------------
    # Consume
    # ------------------------------------------------------------------

    async def consume(
        self,
        queue_name: str,
        handler: Callable[[dict[str, Any]], Any],
    ) -> None:
        """Start consuming messages from a queue.

        The handler receives the decoded payload dict. If it raises,
        the message is retried or sent to dead-letter.

        Args:
            queue_name: Queue to consume from.
            handler: Async callable that processes the message payload.
        """
        queue = await self.channel.declare_queue(
            queue_name, durable=True, passive=True
        )

        async with queue.iterator() as qi:
            async for message in qi:
                async with message.process(requeue=False):
                    retry_count = (message.headers or {}).get("x-retry-count", 0)
                    try:
                        payload = json.loads(message.body.decode())
                        await handler(payload)
                    except Exception as exc:
                        logger.error(
                            "Handler failed for %s (retry %d/%d): %s",
                            queue_name,
                            retry_count,
                            MAX_RETRIES,
                            exc,
                            exc_info=True,
                        )
                        if retry_count < MAX_RETRIES:
                            await self.publish(
                                queue_name,
                                json.loads(message.body.decode()),
                                retry_count=retry_count + 1,
                            )
                        else:
                            logger.warning(
                                "Message exhausted retries, sent to DLQ: %s",
                                message.body[:200],
                            )

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get message counts for all queues."""
        stats = {}
        for queue_name in ALL_QUEUES + [DLX_QUEUE]:
            try:
                queue = await self.channel.declare_queue(
                    queue_name, durable=True, passive=True
                )
                stats[queue_name] = {
                    "message_count": queue.declaration_result.message_count,
                    "consumer_count": queue.declaration_result.consumer_count,
                }
            except Exception:
                stats[queue_name] = {"message_count": -1, "consumer_count": -1}
        return stats


# Global broker instance
task_broker = TaskBroker()
