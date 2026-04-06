"""AMQP 0-9-1 binary protocol: frame encoding/decoding.

Pure functions — no I/O. Used internally by client.py.
Implements only the subset of methods needed by the task broker.
"""

from __future__ import annotations

import struct
from io import BytesIO
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTOCOL_HEADER = b"AMQP\x00\x00\x09\x01"

FRAME_METHOD = 1
FRAME_HEADER = 2
FRAME_BODY = 3
FRAME_HEARTBEAT = 8
FRAME_END = 0xCE

# Class IDs
CLS_CONNECTION = 10
CLS_CHANNEL = 20
CLS_EXCHANGE = 40
CLS_QUEUE = 50
CLS_BASIC = 60

# Method IDs
METH_CONNECTION_START = 10
METH_CONNECTION_START_OK = 11
METH_CONNECTION_TUNE = 30
METH_CONNECTION_TUNE_OK = 31
METH_CONNECTION_OPEN = 40
METH_CONNECTION_OPEN_OK = 41
METH_CONNECTION_CLOSE = 50
METH_CONNECTION_CLOSE_OK = 51

METH_CHANNEL_OPEN = 10
METH_CHANNEL_OPEN_OK = 11
METH_CHANNEL_CLOSE = 40
METH_CHANNEL_CLOSE_OK = 41

METH_EXCHANGE_DECLARE = 10
METH_EXCHANGE_DECLARE_OK = 11

METH_QUEUE_DECLARE = 10
METH_QUEUE_DECLARE_OK = 11
METH_QUEUE_BIND = 20
METH_QUEUE_BIND_OK = 21

METH_BASIC_QOS = 10
METH_BASIC_QOS_OK = 11
METH_BASIC_CONSUME = 20
METH_BASIC_CONSUME_OK = 21
METH_BASIC_PUBLISH = 40
METH_BASIC_DELIVER = 60
METH_BASIC_ACK = 80
METH_BASIC_NACK = 120

# Basic property flags
PROP_CONTENT_TYPE = 1 << 15
PROP_HEADERS = 1 << 13
PROP_DELIVERY_MODE = 1 << 12

# ---------------------------------------------------------------------------
# Primitive encoding
# ---------------------------------------------------------------------------


def _enc_short_str(s: str) -> bytes:
    b = s.encode("utf-8")
    if len(b) > 255:
        raise ValueError(f"Short string too long: {len(b)}")
    return struct.pack("!B", len(b)) + b


def _enc_long_str(b: bytes) -> bytes:
    return struct.pack("!I", len(b)) + b


def _enc_table(table: dict[str, Any]) -> bytes:
    buf = BytesIO()
    for key, value in table.items():
        buf.write(_enc_short_str(key))
        buf.write(_enc_field_value(value))
    payload = buf.getvalue()
    return struct.pack("!I", len(payload)) + payload


def _enc_field_value(value: Any) -> bytes:
    if isinstance(value, bool):
        return b"t" + struct.pack("!B", int(value))
    if isinstance(value, int):
        if -(2**31) <= value < 2**31:
            return b"I" + struct.pack("!i", value)
        return b"l" + struct.pack("!q", value)
    if isinstance(value, str):
        b = value.encode("utf-8")
        return b"S" + struct.pack("!I", len(b)) + b
    if isinstance(value, bytes):
        return b"S" + struct.pack("!I", len(value)) + value
    if isinstance(value, dict):
        return b"F" + _enc_table(value)
    if value is None:
        return b"V"
    raise TypeError(f"Unsupported AMQP field type: {type(value)}")


# ---------------------------------------------------------------------------
# Primitive decoding
# ---------------------------------------------------------------------------


def _dec_short_str(buf: BytesIO) -> str:
    (length,) = struct.unpack("!B", buf.read(1))
    return buf.read(length).decode("utf-8")


def _dec_long_str(buf: BytesIO) -> bytes:
    (length,) = struct.unpack("!I", buf.read(4))
    return buf.read(length)


def _dec_table(buf: BytesIO) -> dict[str, Any]:
    (size,) = struct.unpack("!I", buf.read(4))
    end = buf.tell() + size
    table: dict[str, Any] = {}
    while buf.tell() < end:
        key = _dec_short_str(buf)
        table[key] = _dec_field_value(buf)
    return table


def _dec_field_value(buf: BytesIO) -> Any:
    tag = buf.read(1)
    if tag == b"t":
        return bool(struct.unpack("!B", buf.read(1))[0])
    if tag == b"b":
        return struct.unpack("!b", buf.read(1))[0]
    if tag == b"B":
        return struct.unpack("!B", buf.read(1))[0]
    if tag == b"s":
        return struct.unpack("!h", buf.read(2))[0]
    if tag == b"u":
        return struct.unpack("!H", buf.read(2))[0]
    if tag == b"I":
        return struct.unpack("!i", buf.read(4))[0]
    if tag == b"l":
        return struct.unpack("!q", buf.read(8))[0]
    if tag == b"f":
        return struct.unpack("!f", buf.read(4))[0]
    if tag == b"d":
        return struct.unpack("!d", buf.read(8))[0]
    if tag == b"S":
        return _dec_long_str(buf)
    if tag == b"F":
        return _dec_table(buf)
    if tag == b"T":
        return struct.unpack("!Q", buf.read(8))[0]
    if tag == b"V":
        return None
    raise ValueError(f"Unknown AMQP field tag: {tag!r}")


# ---------------------------------------------------------------------------
# Frame encoding / decoding
# ---------------------------------------------------------------------------


def encode_frame(frame_type: int, channel: int, payload: bytes) -> bytes:
    return (
        struct.pack("!BHI", frame_type, channel, len(payload))
        + payload
        + struct.pack("!B", FRAME_END)
    )


def encode_method_frame(
    channel: int, class_id: int, method_id: int, args: bytes = b""
) -> bytes:
    payload = struct.pack("!HH", class_id, method_id) + args
    return encode_frame(FRAME_METHOD, channel, payload)


def encode_header_frame(
    channel: int,
    class_id: int,
    body_size: int,
    properties: dict[str, Any],
) -> bytes:
    flags = 0
    prop_buf = BytesIO()
    if "content_type" in properties:
        flags |= PROP_CONTENT_TYPE
        prop_buf.write(_enc_short_str(properties["content_type"]))
    if "headers" in properties:
        flags |= PROP_HEADERS
        prop_buf.write(_enc_table(properties["headers"]))
    if "delivery_mode" in properties:
        flags |= PROP_DELIVERY_MODE
        prop_buf.write(struct.pack("!B", properties["delivery_mode"]))
    payload = (
        struct.pack("!HHQ", class_id, 0, body_size)
        + struct.pack("!H", flags)
        + prop_buf.getvalue()
    )
    return encode_frame(FRAME_HEADER, channel, payload)


def encode_body_frames(
    channel: int, body: bytes, frame_max: int = 131072
) -> list[bytes]:
    max_payload = frame_max - 8  # 7-byte header + 1-byte end
    frames = []
    offset = 0
    while offset < len(body):
        chunk = body[offset : offset + max_payload]
        frames.append(encode_frame(FRAME_BODY, channel, chunk))
        offset += max_payload
    if not frames:
        frames.append(encode_frame(FRAME_BODY, channel, b""))
    return frames


async def read_frame(reader: Any) -> tuple[int, int, bytes]:
    """Read one AMQP frame from an asyncio StreamReader.

    Returns (frame_type, channel, payload).
    """
    header = await reader.readexactly(7)
    frame_type, channel, size = struct.unpack("!BHI", header)
    payload = await reader.readexactly(size) if size else b""
    end = await reader.readexactly(1)
    if end[0] != FRAME_END:
        raise ProtocolError(f"Bad frame end: 0x{end[0]:02x}")
    return frame_type, channel, payload


def decode_method(payload: bytes) -> tuple[int, int, BytesIO]:
    """Decode a method frame payload → (class_id, method_id, args_buf)."""
    class_id, method_id = struct.unpack("!HH", payload[:4])
    return class_id, method_id, BytesIO(payload[4:])


def decode_header(payload: bytes) -> tuple[int, int, dict[str, Any]]:
    """Decode a content header → (class_id, body_size, properties)."""
    class_id, _weight, body_size = struct.unpack("!HHQ", payload[:12])
    (flags,) = struct.unpack("!H", payload[12:14])
    buf = BytesIO(payload[14:])
    props: dict[str, Any] = {}
    if flags & PROP_CONTENT_TYPE:
        props["content_type"] = _dec_short_str(buf)
    if flags & PROP_HEADERS:
        props["headers"] = _dec_table(buf)
    if flags & PROP_DELIVERY_MODE:
        props["delivery_mode"] = struct.unpack("!B", buf.read(1))[0]
    return class_id, body_size, props


# ---------------------------------------------------------------------------
# Method argument encoders
# ---------------------------------------------------------------------------


def enc_connection_start_ok(
    properties: dict[str, Any], mechanism: str, response: str, locale: str
) -> bytes:
    buf = BytesIO()
    buf.write(_enc_table(properties))
    buf.write(_enc_short_str(mechanism))
    buf.write(_enc_long_str(response.encode("utf-8")))
    buf.write(_enc_short_str(locale))
    return buf.getvalue()


def enc_connection_tune_ok(
    channel_max: int, frame_max: int, heartbeat: int
) -> bytes:
    return struct.pack("!HIH", channel_max, frame_max, heartbeat)


def enc_connection_open(vhost: str) -> bytes:
    return _enc_short_str(vhost) + b"\x00\x00"


def enc_connection_close() -> bytes:
    return struct.pack("!H", 200) + _enc_short_str("Normal close") + struct.pack("!HH", 0, 0)


def enc_channel_open() -> bytes:
    return _enc_short_str("")


def enc_exchange_declare(
    name: str,
    type_: str,
    passive: bool = False,
    durable: bool = False,
    auto_delete: bool = False,
    internal: bool = False,
    no_wait: bool = False,
    arguments: dict[str, Any] | None = None,
) -> bytes:
    buf = BytesIO()
    buf.write(struct.pack("!H", 0))  # reserved-1
    buf.write(_enc_short_str(name))
    buf.write(_enc_short_str(type_))
    bits = (
        (int(passive) << 0)
        | (int(durable) << 1)
        | (int(auto_delete) << 2)
        | (int(internal) << 3)
        | (int(no_wait) << 4)
    )
    buf.write(struct.pack("!B", bits))
    buf.write(_enc_table(arguments or {}))
    return buf.getvalue()


def enc_queue_declare(
    name: str,
    passive: bool = False,
    durable: bool = False,
    exclusive: bool = False,
    auto_delete: bool = False,
    no_wait: bool = False,
    arguments: dict[str, Any] | None = None,
) -> bytes:
    buf = BytesIO()
    buf.write(struct.pack("!H", 0))
    buf.write(_enc_short_str(name))
    bits = (
        (int(passive) << 0)
        | (int(durable) << 1)
        | (int(exclusive) << 2)
        | (int(auto_delete) << 3)
        | (int(no_wait) << 4)
    )
    buf.write(struct.pack("!B", bits))
    buf.write(_enc_table(arguments or {}))
    return buf.getvalue()


def enc_queue_bind(
    queue: str,
    exchange: str,
    routing_key: str = "",
    no_wait: bool = False,
    arguments: dict[str, Any] | None = None,
) -> bytes:
    buf = BytesIO()
    buf.write(struct.pack("!H", 0))
    buf.write(_enc_short_str(queue))
    buf.write(_enc_short_str(exchange))
    buf.write(_enc_short_str(routing_key))
    buf.write(struct.pack("!B", int(no_wait)))
    buf.write(_enc_table(arguments or {}))
    return buf.getvalue()


def enc_basic_qos(
    prefetch_size: int = 0, prefetch_count: int = 0, global_: bool = False
) -> bytes:
    return struct.pack("!IHB", prefetch_size, prefetch_count, int(global_))


def enc_basic_publish(
    exchange: str = "",
    routing_key: str = "",
    mandatory: bool = False,
    immediate: bool = False,
) -> bytes:
    buf = BytesIO()
    buf.write(struct.pack("!H", 0))
    buf.write(_enc_short_str(exchange))
    buf.write(_enc_short_str(routing_key))
    bits = (int(mandatory) << 0) | (int(immediate) << 1)
    buf.write(struct.pack("!B", bits))
    return buf.getvalue()


def enc_basic_consume(
    queue: str,
    consumer_tag: str = "",
    no_local: bool = False,
    no_ack: bool = False,
    exclusive: bool = False,
    no_wait: bool = False,
    arguments: dict[str, Any] | None = None,
) -> bytes:
    buf = BytesIO()
    buf.write(struct.pack("!H", 0))
    buf.write(_enc_short_str(queue))
    buf.write(_enc_short_str(consumer_tag))
    bits = (
        (int(no_local) << 0)
        | (int(no_ack) << 1)
        | (int(exclusive) << 2)
        | (int(no_wait) << 3)
    )
    buf.write(struct.pack("!B", bits))
    buf.write(_enc_table(arguments or {}))
    return buf.getvalue()


def enc_basic_ack(delivery_tag: int, multiple: bool = False) -> bytes:
    return struct.pack("!QB", delivery_tag, int(multiple))


def enc_basic_nack(
    delivery_tag: int, multiple: bool = False, requeue: bool = False
) -> bytes:
    bits = (int(multiple) << 0) | (int(requeue) << 1)
    return struct.pack("!QB", delivery_tag, bits)


# ---------------------------------------------------------------------------
# Method argument decoders
# ---------------------------------------------------------------------------


def dec_connection_start(buf: BytesIO) -> dict[str, Any]:
    major = struct.unpack("!B", buf.read(1))[0]
    minor = struct.unpack("!B", buf.read(1))[0]
    server_properties = _dec_table(buf)
    mechanisms = _dec_long_str(buf).decode("utf-8")
    locales = _dec_long_str(buf).decode("utf-8")
    return {
        "version_major": major,
        "version_minor": minor,
        "server_properties": server_properties,
        "mechanisms": mechanisms,
        "locales": locales,
    }


def dec_connection_tune(buf: BytesIO) -> dict[str, Any]:
    channel_max, frame_max, heartbeat = struct.unpack("!HIH", buf.read(8))
    return {
        "channel_max": channel_max,
        "frame_max": frame_max,
        "heartbeat": heartbeat,
    }


def dec_queue_declare_ok(buf: BytesIO) -> dict[str, Any]:
    name = _dec_short_str(buf)
    message_count, consumer_count = struct.unpack("!II", buf.read(8))
    return {
        "queue": name,
        "message_count": message_count,
        "consumer_count": consumer_count,
    }


def dec_basic_consume_ok(buf: BytesIO) -> dict[str, Any]:
    return {"consumer_tag": _dec_short_str(buf)}


def dec_basic_deliver(buf: BytesIO) -> dict[str, Any]:
    consumer_tag = _dec_short_str(buf)
    delivery_tag = struct.unpack("!Q", buf.read(8))[0]
    redelivered = bool(struct.unpack("!B", buf.read(1))[0])
    exchange = _dec_short_str(buf)
    routing_key = _dec_short_str(buf)
    return {
        "consumer_tag": consumer_tag,
        "delivery_tag": delivery_tag,
        "redelivered": redelivered,
        "exchange": exchange,
        "routing_key": routing_key,
    }


def dec_connection_close(buf: BytesIO) -> dict[str, Any]:
    reply_code = struct.unpack("!H", buf.read(2))[0]
    reply_text = _dec_short_str(buf)
    class_id, method_id = struct.unpack("!HH", buf.read(4))
    return {
        "reply_code": reply_code,
        "reply_text": reply_text,
        "class_id": class_id,
        "method_id": method_id,
    }


def dec_channel_close(buf: BytesIO) -> dict[str, Any]:
    reply_code = struct.unpack("!H", buf.read(2))[0]
    reply_text = _dec_short_str(buf)
    class_id, method_id = struct.unpack("!HH", buf.read(4))
    return {
        "reply_code": reply_code,
        "reply_text": reply_text,
        "class_id": class_id,
        "method_id": method_id,
    }


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProtocolError(Exception):
    """AMQP protocol-level error."""
