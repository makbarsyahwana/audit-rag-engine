"""Public types for the AMQP client."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, AsyncIterator


class DeliveryMode(IntEnum):
    TRANSIENT = 1
    PERSISTENT = 2


class ExchangeType(str, Enum):
    DIRECT = "direct"
    FANOUT = "fanout"
    TOPIC = "topic"
    HEADERS = "headers"


@dataclass
class QueueInfo:
    """Result from Queue.Declare-Ok."""

    name: str = ""
    message_count: int = 0
    consumer_count: int = 0


@dataclass
class Message:
    """Outgoing AMQP message."""

    body: bytes
    delivery_mode: DeliveryMode = DeliveryMode.PERSISTENT
    content_type: str = "application/octet-stream"
    headers: dict[str, Any] = field(default_factory=dict)


@dataclass
class IncomingMessage:
    """Incoming AMQP message with ack/nack support."""

    body: bytes
    headers: dict[str, Any]
    delivery_tag: int
    exchange: str
    routing_key: str
    redelivered: bool
    consumer_tag: str
    _channel: Any = field(repr=False, default=None)

    async def ack(self) -> None:
        """Acknowledge this message."""
        if self._channel:
            await self._channel.basic_ack(self.delivery_tag)

    async def nack(self, requeue: bool = False) -> None:
        """Negatively acknowledge this message."""
        if self._channel:
            await self._channel.basic_nack(self.delivery_tag, requeue=requeue)

    @asynccontextmanager
    async def process(self, requeue: bool = False) -> AsyncIterator[None]:
        """Context manager: ack on success, nack on exception."""
        try:
            yield
            await self.ack()
        except Exception:
            await self.nack(requeue=requeue)
            raise
