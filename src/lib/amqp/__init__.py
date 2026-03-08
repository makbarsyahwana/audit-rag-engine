"""Lightweight async AMQP 0-9-1 client for RabbitMQ.

Built in-house — no third-party AMQP dependencies.
Supports exactly what the ingestion task broker needs:
connect, declare exchanges/queues, publish, consume, ack.
"""

from src.lib.amqp.client import Channel, Connection, Exchange, Queue
from src.lib.amqp.types import DeliveryMode, ExchangeType, IncomingMessage, Message

__all__ = [
    "Connection",
    "Channel",
    "Exchange",
    "Queue",
    "Message",
    "IncomingMessage",
    "DeliveryMode",
    "ExchangeType",
]


async def connect_robust(url: str) -> "Connection":
    """Connect to RabbitMQ with automatic reconnection.

    Args:
        url: AMQP URL, e.g. ``amqp://user:pass@host:5672/vhost``
    """
    conn = Connection(url)
    await conn.connect()
    return conn
