"""Async AMQP 0-9-1 client: Connection, Channel, Exchange, Queue.

Uses only asyncio + the sibling ``_protocol`` module — no third-party deps.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from urllib.parse import unquote, urlparse

from src.lib.amqp._protocol import (
    CLS_BASIC,
    CLS_CHANNEL,
    CLS_CONNECTION,
    CLS_EXCHANGE,
    CLS_QUEUE,
    FRAME_BODY,
    FRAME_HEADER,
    FRAME_HEARTBEAT,
    FRAME_METHOD,
    METH_BASIC_CONSUME_OK,
    METH_BASIC_DELIVER,
    METH_BASIC_QOS_OK,
    METH_CHANNEL_CLOSE,
    METH_CHANNEL_OPEN_OK,
    METH_CONNECTION_CLOSE,
    METH_CONNECTION_OPEN_OK,
    METH_CONNECTION_START,
    METH_CONNECTION_TUNE,
    METH_CONNECTION_TUNE_OK,
    METH_EXCHANGE_DECLARE_OK,
    METH_QUEUE_BIND_OK,
    METH_QUEUE_DECLARE_OK,
    PROTOCOL_HEADER,
    ProtocolError,
    dec_basic_consume_ok,
    dec_basic_deliver,
    dec_channel_close,
    dec_connection_close,
    dec_connection_start,
    dec_connection_tune,
    dec_queue_declare_ok,
    decode_header,
    decode_method,
    enc_basic_ack,
    enc_basic_consume,
    enc_basic_nack,
    enc_basic_publish,
    enc_basic_qos,
    enc_channel_open,
    enc_connection_close,
    enc_connection_open,
    enc_connection_start_ok,
    enc_connection_tune_ok,
    enc_exchange_declare,
    enc_queue_bind,
    enc_queue_declare,
    encode_body_frames,
    encode_frame,
    encode_header_frame,
    encode_method_frame,
    read_frame,
)
from src.lib.amqp.types import (
    ExchangeType,
    IncomingMessage,
    Message,
    QueueInfo,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


def _parse_amqp_url(url: str) -> dict[str, Any]:
    p = urlparse(url)
    return {
        "host": p.hostname or "localhost",
        "port": p.port or 5672,
        "user": unquote(p.username or "guest"),
        "password": unquote(p.password or "guest"),
        "vhost": unquote(p.path.lstrip("/")) or "/",
    }


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


class Connection:
    """Async AMQP connection."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._params = _parse_amqp_url(url)
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._channels: dict[int, Channel] = {}
        self._next_channel_id = 1
        self._frame_max = 131072
        self._heartbeat = 60
        self._loop_task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._waiters: dict[tuple[int, int, int], asyncio.Future[Any]] = {}

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def connect(self) -> None:
        """Open TCP connection and perform AMQP handshake."""
        self._reader, self._writer = await asyncio.open_connection(
            self._params["host"], self._params["port"]
        )
        self._writer.write(PROTOCOL_HEADER)
        await self._writer.drain()

        # Connection.Start
        ft, ch, payload = await read_frame(self._reader)
        cls, meth, buf = decode_method(payload)
        if cls != CLS_CONNECTION or meth != METH_CONNECTION_START:
            raise ProtocolError(f"Expected Connection.Start, got {cls}.{meth}")
        dec_connection_start(buf)

        # Connection.Start-Ok
        creds = f"\x00{self._params['user']}\x00{self._params['password']}"
        args = enc_connection_start_ok(
            properties={"product": "audit-amqp", "version": "1.0.0"},
            mechanism="PLAIN",
            response=creds,
            locale="en_US",
        )
        self._writer.write(
            encode_method_frame(0, CLS_CONNECTION, 11, args)
        )
        await self._writer.drain()

        # Connection.Tune
        ft, ch, payload = await read_frame(self._reader)
        cls, meth, buf = decode_method(payload)
        if cls != CLS_CONNECTION or meth != METH_CONNECTION_TUNE:
            raise ProtocolError(f"Expected Connection.Tune, got {cls}.{meth}")
        tune = dec_connection_tune(buf)
        self._frame_max = tune["frame_max"] or 131072
        self._heartbeat = tune["heartbeat"]

        # Connection.Tune-Ok
        args = enc_connection_tune_ok(
            channel_max=tune["channel_max"],
            frame_max=self._frame_max,
            heartbeat=self._heartbeat,
        )
        self._writer.write(
            encode_method_frame(0, CLS_CONNECTION, METH_CONNECTION_TUNE_OK, args)
        )
        await self._writer.drain()

        # Connection.Open
        args = enc_connection_open(self._params["vhost"])
        self._writer.write(
            encode_method_frame(0, CLS_CONNECTION, 40, args)
        )
        await self._writer.drain()

        # Connection.Open-Ok
        ft, ch, payload = await read_frame(self._reader)
        cls, meth, buf = decode_method(payload)
        if cls != CLS_CONNECTION or meth != METH_CONNECTION_OPEN_OK:
            raise ProtocolError(f"Expected Connection.Open-Ok, got {cls}.{meth}")

        # Start frame dispatch loop
        self._loop_task = asyncio.create_task(self._frame_loop())
        logger.info("AMQP connected to %s:%d", self._params["host"], self._params["port"])

    async def channel(self) -> "Channel":
        """Open a new channel."""
        ch_id = self._next_channel_id
        self._next_channel_id += 1
        ch = Channel(self, ch_id)
        self._channels[ch_id] = ch
        await ch._open()
        return ch

    async def close(self) -> None:
        """Close the connection gracefully."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._writer and not self._writer.is_closing():
                args = enc_connection_close()
                self._writer.write(
                    encode_method_frame(0, CLS_CONNECTION, 50, args)
                )
                await self._writer.drain()
        except Exception:
            pass
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._writer:
            self._writer.close()
        logger.info("AMQP connection closed")

    # -- internal --

    async def _send(self, data: bytes) -> None:
        if self._writer and not self._writer.is_closing():
            self._writer.write(data)
            await self._writer.drain()

    def _set_waiter(self, channel: int, class_id: int, method_id: int) -> asyncio.Future[Any]:
        key = (channel, class_id, method_id)
        fut: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._waiters[key] = fut
        return fut

    async def _frame_loop(self) -> None:
        """Background task: read frames and dispatch."""
        assert self._reader is not None
        try:
            while not self._closed:
                ft, ch_id, payload = await read_frame(self._reader)

                if ft == FRAME_HEARTBEAT:
                    await self._send(encode_frame(FRAME_HEARTBEAT, 0, b""))
                    continue

                if ft == FRAME_METHOD:
                    cls, meth, buf = decode_method(payload)

                    # Connection-level methods
                    if ch_id == 0:
                        if cls == CLS_CONNECTION and meth == METH_CONNECTION_CLOSE:
                            info = dec_connection_close(buf)
                            logger.warning("Server closing connection: %s", info["reply_text"])
                            await self._send(
                                encode_method_frame(0, CLS_CONNECTION, 51)
                            )
                            self._closed = True
                            return
                        continue

                    # Channel-level methods
                    channel = self._channels.get(ch_id)
                    if not channel:
                        continue

                    # Check waiters first
                    key = (ch_id, cls, meth)
                    waiter = self._waiters.pop(key, None)
                    if waiter and not waiter.done():
                        waiter.set_result((cls, meth, buf))
                        continue

                    # Async deliveries
                    if cls == CLS_BASIC and meth == METH_BASIC_DELIVER:
                        channel._start_delivery(dec_basic_deliver(buf))
                        continue

                    if cls == CLS_CHANNEL and meth == METH_CHANNEL_CLOSE:
                        info = dec_channel_close(buf)
                        logger.warning("Channel %d closed: %s", ch_id, info["reply_text"])
                        await self._send(
                            encode_method_frame(ch_id, CLS_CHANNEL, 41)
                        )
                        continue

                elif ft == FRAME_HEADER:
                    channel = self._channels.get(ch_id)
                    if channel:
                        _, body_size, props = decode_header(payload)
                        channel._set_delivery_header(body_size, props)

                elif ft == FRAME_BODY:
                    channel = self._channels.get(ch_id)
                    if channel:
                        channel._append_delivery_body(payload)

        except asyncio.CancelledError:
            return
        except asyncio.IncompleteReadError:
            if not self._closed:
                logger.error("AMQP connection lost")
                self._closed = True
        except Exception as exc:
            if not self._closed:
                logger.error("AMQP frame loop error: %s", exc)
                self._closed = True


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------


class Channel:
    """AMQP channel."""

    def __init__(self, connection: Connection, channel_id: int) -> None:
        self._conn = connection
        self._id = channel_id
        self._default_exchange: Optional[Exchange] = None
        self._delivery_queue: asyncio.Queue[IncomingMessage] = asyncio.Queue()
        # Pending delivery state
        self._pending_deliver: Optional[dict[str, Any]] = None
        self._pending_body_size = 0
        self._pending_props: dict[str, Any] = {}
        self._pending_body = bytearray()

    @property
    def default_exchange(self) -> "Exchange":
        if self._default_exchange is None:
            self._default_exchange = Exchange(self, "")
        return self._default_exchange

    async def _open(self) -> None:
        waiter = self._conn._set_waiter(self._id, CLS_CHANNEL, METH_CHANNEL_OPEN_OK)
        await self._conn._send(
            encode_method_frame(self._id, CLS_CHANNEL, 10, enc_channel_open())
        )
        await asyncio.wait_for(waiter, timeout=10.0)

    async def set_qos(self, prefetch_count: int = 0) -> None:
        waiter = self._conn._set_waiter(self._id, CLS_BASIC, METH_BASIC_QOS_OK)
        await self._conn._send(
            encode_method_frame(
                self._id, CLS_BASIC, 10, enc_basic_qos(prefetch_count=prefetch_count)
            )
        )
        await asyncio.wait_for(waiter, timeout=10.0)

    async def declare_exchange(
        self,
        name: str,
        type_: ExchangeType | str = ExchangeType.DIRECT,
        durable: bool = False,
    ) -> "Exchange":
        type_str = type_.value if isinstance(type_, ExchangeType) else type_
        waiter = self._conn._set_waiter(self._id, CLS_EXCHANGE, METH_EXCHANGE_DECLARE_OK)
        await self._conn._send(
            encode_method_frame(
                self._id,
                CLS_EXCHANGE,
                10,
                enc_exchange_declare(name=name, type_=type_str, durable=durable),
            )
        )
        await asyncio.wait_for(waiter, timeout=10.0)
        return Exchange(self, name)

    async def declare_queue(
        self,
        name: str,
        durable: bool = False,
        exclusive: bool = False,
        auto_delete: bool = False,
        passive: bool = False,
        arguments: dict[str, Any] | None = None,
    ) -> "Queue":
        waiter = self._conn._set_waiter(self._id, CLS_QUEUE, METH_QUEUE_DECLARE_OK)
        await self._conn._send(
            encode_method_frame(
                self._id,
                CLS_QUEUE,
                10,
                enc_queue_declare(
                    name=name,
                    passive=passive,
                    durable=durable,
                    exclusive=exclusive,
                    auto_delete=auto_delete,
                    arguments=arguments,
                ),
            )
        )
        _, _, buf = await asyncio.wait_for(waiter, timeout=10.0)
        info = dec_queue_declare_ok(buf)
        return Queue(
            self,
            info["queue"],
            QueueInfo(
                name=info["queue"],
                message_count=info["message_count"],
                consumer_count=info["consumer_count"],
            ),
        )

    async def basic_ack(self, delivery_tag: int, multiple: bool = False) -> None:
        await self._conn._send(
            encode_method_frame(
                self._id, CLS_BASIC, 80, enc_basic_ack(delivery_tag, multiple)
            )
        )

    async def basic_nack(
        self, delivery_tag: int, multiple: bool = False, requeue: bool = False
    ) -> None:
        await self._conn._send(
            encode_method_frame(
                self._id, CLS_BASIC, 120, enc_basic_nack(delivery_tag, multiple, requeue)
            )
        )

    async def _publish(
        self, exchange: str, routing_key: str, message: Message
    ) -> None:
        # Method frame
        args = enc_basic_publish(exchange=exchange, routing_key=routing_key)
        method_frame = encode_method_frame(self._id, CLS_BASIC, 40, args)

        # Header frame
        props: dict[str, Any] = {
            "content_type": message.content_type,
            "delivery_mode": int(message.delivery_mode),
        }
        if message.headers:
            props["headers"] = message.headers
        header_frame = encode_header_frame(
            self._id, CLS_BASIC, len(message.body), props
        )

        # Body frames
        body_frames = encode_body_frames(
            self._id, message.body, self._conn._frame_max
        )

        data = method_frame + header_frame + b"".join(body_frames)
        await self._conn._send(data)

    async def _consume(self, queue_name: str) -> str:
        waiter = self._conn._set_waiter(self._id, CLS_BASIC, METH_BASIC_CONSUME_OK)
        await self._conn._send(
            encode_method_frame(
                self._id, CLS_BASIC, 20, enc_basic_consume(queue=queue_name)
            )
        )
        _, _, buf = await asyncio.wait_for(waiter, timeout=10.0)
        info = dec_basic_consume_ok(buf)
        return info["consumer_tag"]

    # -- delivery assembly --

    def _start_delivery(self, deliver_args: dict[str, Any]) -> None:
        self._pending_deliver = deliver_args
        self._pending_body = bytearray()
        self._pending_body_size = 0
        self._pending_props = {}

    def _set_delivery_header(self, body_size: int, props: dict[str, Any]) -> None:
        self._pending_body_size = body_size
        self._pending_props = props

    def _append_delivery_body(self, data: bytes) -> None:
        self._pending_body.extend(data)
        if self._pending_deliver and len(self._pending_body) >= self._pending_body_size:
            self._finish_delivery()

    def _finish_delivery(self) -> None:
        if not self._pending_deliver:
            return
        msg = IncomingMessage(
            body=bytes(self._pending_body),
            headers=self._pending_props.get("headers", {}),
            delivery_tag=self._pending_deliver["delivery_tag"],
            exchange=self._pending_deliver["exchange"],
            routing_key=self._pending_deliver["routing_key"],
            redelivered=self._pending_deliver["redelivered"],
            consumer_tag=self._pending_deliver["consumer_tag"],
            _channel=self,
        )
        self._delivery_queue.put_nowait(msg)
        self._pending_deliver = None


# ---------------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------------


class Exchange:
    """AMQP exchange handle."""

    def __init__(self, channel: Channel, name: str) -> None:
        self._channel = channel
        self.name = name

    async def publish(self, message: Message, routing_key: str = "") -> None:
        await self._channel._publish(self.name, routing_key, message)


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


class Queue:
    """AMQP queue handle with async iteration."""

    def __init__(self, channel: Channel, name: str, info: QueueInfo) -> None:
        self._channel = channel
        self.name = name
        self.declaration_result = info
        self._consumer_tag: Optional[str] = None

    async def bind(self, exchange: Exchange, routing_key: str = "") -> None:
        waiter = self._channel._conn._set_waiter(
            self._channel._id, CLS_QUEUE, METH_QUEUE_BIND_OK
        )
        await self._channel._conn._send(
            encode_method_frame(
                self._channel._id,
                CLS_QUEUE,
                20,
                enc_queue_bind(queue=self.name, exchange=exchange.name, routing_key=routing_key),
            )
        )
        await asyncio.wait_for(waiter, timeout=10.0)

    async def iterator(self) -> _QueueIterator:
        """Start consuming and return an async iterator of messages."""
        tag = await self._channel._consume(self.name)
        self._consumer_tag = tag
        return _QueueIterator(self._channel)


class _QueueIterator:
    """Async context manager + iterator for queue consumption."""

    def __init__(self, channel: Channel) -> None:
        self._channel = channel

    async def __aenter__(self) -> "_QueueIterator":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass

    def __aiter__(self) -> "_QueueIterator":
        return self

    async def __anext__(self) -> IncomingMessage:
        return await self._channel._delivery_queue.get()
