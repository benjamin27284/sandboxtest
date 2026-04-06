"""Minimal protobuf wire-format codec for abms.market messages.

Encodes/decodes the messages defined in proto/market.proto using only
the Python standard library (struct module).  Wire-compatible with
the C++ protobuf serialization used by the LOB engine.

Protobuf wire format reference:
  - Field: (field_number << 3) | wire_type
  - Wire type 0: varint  (int32, int64, enum, bool)
  - Wire type 1: 64-bit  (double)
  - Wire type 2: length-delimited (string, bytes, embedded message)
"""

from __future__ import annotations

import struct
from typing import Any


# ─── Low-level wire format helpers ──────────────────────────────────────────

def _encode_varint(value: int) -> bytes:
    """Encode an unsigned varint."""
    if value < 0:
        # Protobuf uses two's complement for negative varints (10 bytes)
        value = value & 0xFFFFFFFFFFFFFFFF
    parts = []
    while value > 0x7F:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    parts.append(value & 0x7F)
    return bytes(parts)


def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode a varint starting at pos. Returns (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if (b & 0x80) == 0:
            break
        shift += 7
    return result, pos


def _encode_field_varint(field_num: int, value: int) -> bytes:
    """Encode a varint field (wire type 0)."""
    if value == 0:
        return b""
    tag = _encode_varint((field_num << 3) | 0)
    return tag + _encode_varint(value)


def _encode_field_double(field_num: int, value: float) -> bytes:
    """Encode a double field (wire type 1)."""
    if value == 0.0:
        return b""
    tag = _encode_varint((field_num << 3) | 1)
    return tag + struct.pack("<d", value)


def _encode_field_string(field_num: int, value: str) -> bytes:
    """Encode a string field (wire type 2)."""
    if not value:
        return b""
    encoded = value.encode("utf-8")
    tag = _encode_varint((field_num << 3) | 2)
    return tag + _encode_varint(len(encoded)) + encoded


def _encode_field_bytes(field_num: int, value: bytes) -> bytes:
    """Encode a bytes/embedded-message field (wire type 2)."""
    if not value:
        return b""
    tag = _encode_varint((field_num << 3) | 2)
    return tag + _encode_varint(len(value)) + value


def _decode_fields(data: bytes) -> list[tuple[int, int, Any]]:
    """Parse all fields from a protobuf message.

    Returns list of (field_number, wire_type, value) tuples.
    For wire_type 0: value is int
    For wire_type 1: value is bytes (8 bytes, decode as double)
    For wire_type 2: value is bytes (raw payload)
    """
    fields = []
    pos = 0
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07

        if wire_type == 0:  # varint
            value, pos = _decode_varint(data, pos)
            fields.append((field_num, wire_type, value))
        elif wire_type == 1:  # 64-bit (double)
            value = data[pos:pos + 8]
            pos += 8
            fields.append((field_num, wire_type, value))
        elif wire_type == 2:  # length-delimited
            length, pos = _decode_varint(data, pos)
            value = data[pos:pos + length]
            pos += length
            fields.append((field_num, wire_type, value))
        elif wire_type == 5:  # 32-bit
            value = data[pos:pos + 4]
            pos += 4
            fields.append((field_num, wire_type, value))
        else:
            break  # unknown wire type

    return fields


def _fields_to_dict(
    fields: list[tuple[int, int, Any]],
    schema: dict[int, tuple[str, str]],
) -> dict[str, Any]:
    """Convert parsed fields to a dict using a schema.

    schema: {field_number: (field_name, type_hint)}
    type_hint: "string", "double", "int32", "int64", "enum", "message", "repeated_message"
    """
    result: dict[str, Any] = {}
    for field_num, wire_type, value in fields:
        if field_num not in schema:
            continue
        name, type_hint = schema[field_num]
        if type_hint == "string":
            result[name] = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        elif type_hint == "double":
            result[name] = struct.unpack("<d", value)[0] if isinstance(value, bytes) else float(value)
        elif type_hint in ("int32", "int64", "enum"):
            result[name] = int(value)
        elif type_hint == "message":
            result[name] = value  # raw bytes, decode separately
        elif type_hint == "repeated_message":
            result.setdefault(name, []).append(value)
    return result


# ─── Side / OrderType enums ────────────────────────────────────────────────

SIDE_BUY = 1
SIDE_SELL = 2
ORDER_TYPE_LIMIT = 1
ORDER_TYPE_MARKET = 2


# ─── Order ──────────────────────────────────────────────────────────────────

_ORDER_SCHEMA = {
    1: ("order_id", "string"),
    2: ("agent_id", "string"),
    3: ("side", "enum"),
    4: ("type", "enum"),
    5: ("price", "double"),
    6: ("quantity", "int32"),
    7: ("timestamp", "int64"),
}


def encode_order(order: dict) -> bytes:
    """Encode an order dict to protobuf wire format."""
    side_val = SIDE_BUY if order.get("action") == "buy" or order.get("side") == "buy" else SIDE_SELL
    type_val = ORDER_TYPE_MARKET if order.get("type") == "market" else ORDER_TYPE_LIMIT

    return (
        _encode_field_string(1, order.get("order_id", ""))
        + _encode_field_string(2, order.get("agent_id", ""))
        + _encode_field_varint(3, side_val)
        + _encode_field_varint(4, type_val)
        + _encode_field_double(5, float(order.get("price", 0.0)))
        + _encode_field_varint(6, int(order.get("quantity", 0)))
        + _encode_field_varint(7, int(order.get("timestamp", 0)))
    )


def decode_order(data: bytes) -> dict:
    """Decode protobuf bytes to an order dict."""
    fields = _decode_fields(data)
    return _fields_to_dict(fields, _ORDER_SCHEMA)


# ─── Execution ──────────────────────────────────────────────────────────────

_EXECUTION_SCHEMA = {
    1: ("exec_id", "string"),
    2: ("order_id", "string"),
    3: ("agent_id", "string"),
    4: ("fill_price", "double"),
    5: ("fill_qty", "int32"),
    6: ("timestamp", "int64"),
    7: ("counter_order_id", "string"),
    8: ("counter_agent_id", "string"),
    9: ("aggressor_side", "enum"),
}


def decode_execution(data: bytes) -> dict:
    """Decode protobuf Execution bytes to a dict with buyer_id/seller_id."""
    fields = _decode_fields(data)
    raw = _fields_to_dict(fields, _EXECUTION_SCHEMA)

    # Derive buyer_id / seller_id from aggressor_side
    aggressor_side = raw.get("aggressor_side", 0)
    agent_id = raw.get("agent_id", "")
    counter_agent_id = raw.get("counter_agent_id", "")

    if aggressor_side == SIDE_BUY:
        raw["buyer_id"] = agent_id
        raw["seller_id"] = counter_agent_id
    else:
        raw["buyer_id"] = counter_agent_id
        raw["seller_id"] = agent_id

    raw["price"] = raw.get("fill_price", 0.0)
    raw["quantity"] = raw.get("fill_qty", 0)

    return raw


# ─── PriceLevel ─────────────────────────────────────────────────────────────

_PRICE_LEVEL_SCHEMA = {
    1: ("price", "double"),
    2: ("quantity", "int64"),
    3: ("count", "int32"),
}


def decode_price_level(data: bytes) -> dict:
    fields = _decode_fields(data)
    return _fields_to_dict(fields, _PRICE_LEVEL_SCHEMA)


# ─── OrderBookSnapshot ──────────────────────────────────────────────────────

_BOOK_SNAPSHOT_SCHEMA = {
    1: ("tick", "int64"),
    2: ("mid_price", "double"),
    3: ("best_bid", "double"),
    4: ("best_ask", "double"),
    5: ("spread", "double"),
    6: ("bids", "repeated_message"),
    7: ("asks", "repeated_message"),
    8: ("timestamp_ns", "int64"),
}


def decode_book_snapshot(data: bytes) -> dict:
    fields = _decode_fields(data)
    raw = _fields_to_dict(fields, _BOOK_SNAPSHOT_SCHEMA)
    raw["bids"] = [decode_price_level(b) for b in raw.get("bids", [])]
    raw["asks"] = [decode_price_level(a) for a in raw.get("asks", [])]
    return raw


def encode_book_snapshot(snap: dict) -> bytes:
    """Encode an OrderBookSnapshot dict to protobuf."""
    bids_bytes = b""
    for level in snap.get("bids", []):
        level_bytes = (
            _encode_field_double(1, level.get("price", 0.0))
            + _encode_field_varint(2, int(level.get("quantity", 0)))
            + _encode_field_varint(3, int(level.get("count", 0)))
        )
        bids_bytes += _encode_field_bytes(6, level_bytes)

    asks_bytes = b""
    for level in snap.get("asks", []):
        level_bytes = (
            _encode_field_double(1, level.get("price", 0.0))
            + _encode_field_varint(2, int(level.get("quantity", 0)))
            + _encode_field_varint(3, int(level.get("count", 0)))
        )
        asks_bytes += _encode_field_bytes(7, level_bytes)

    return (
        _encode_field_varint(1, int(snap.get("tick", 0)))
        + _encode_field_double(2, snap.get("mid_price", 0.0))
        + _encode_field_double(3, snap.get("best_bid", 0.0))
        + _encode_field_double(4, snap.get("best_ask", 0.0))
        + _encode_field_double(5, snap.get("spread", 0.0))
        + bids_bytes
        + asks_bytes
        + _encode_field_varint(8, int(snap.get("timestamp_ns", 0)))
    )


# ─── MarketSnapshot ────────────────────────────────────────────────────────

_MARKET_SNAPSHOT_SCHEMA = {
    1: ("mid_price", "double"),
    2: ("spread", "double"),
    3: ("total_volume", "int32"),
    4: ("book_snapshot", "message"),
}


def decode_market_snapshot(data: bytes) -> dict:
    fields = _decode_fields(data)
    raw = _fields_to_dict(fields, _MARKET_SNAPSHOT_SCHEMA)
    if "book_snapshot" in raw:
        raw["book_snapshot"] = decode_book_snapshot(raw["book_snapshot"])
    return raw


# ─── TickSummary ────────────────────────────────────────────────────────────

_TICK_SUMMARY_SCHEMA = {
    1: ("tick", "int64"),
    2: ("open_price", "double"),
    3: ("close_price", "double"),
    4: ("high_price", "double"),
    5: ("low_price", "double"),
    6: ("volume", "int64"),
    7: ("num_trades", "int32"),
    8: ("book_snapshot", "message"),
    9: ("consensus", "double"),
    10: ("disagreement", "double"),
    11: ("mass_signal", "double"),
}


def encode_tick_summary(summary: dict) -> bytes:
    """Encode a TickSummary dict to protobuf wire format."""
    book_bytes = b""
    if "book_snapshot" in summary and summary["book_snapshot"]:
        book_bytes = _encode_field_bytes(8, encode_book_snapshot(summary["book_snapshot"]))

    return (
        _encode_field_varint(1, int(summary.get("tick", 0)))
        + _encode_field_double(2, summary.get("open_price", 0.0))
        + _encode_field_double(3, summary.get("close_price", 0.0))
        + _encode_field_double(4, summary.get("high_price", 0.0))
        + _encode_field_double(5, summary.get("low_price", 0.0))
        + _encode_field_varint(6, int(summary.get("volume", 0)))
        + _encode_field_varint(7, int(summary.get("num_trades", 0)))
        + book_bytes
        + _encode_field_double(9, summary.get("consensus", 0.0))
        + _encode_field_double(10, summary.get("disagreement", 0.0))
        + _encode_field_double(11, summary.get("mass_signal", 0.0))
    )


def decode_tick_summary(data: bytes) -> dict:
    fields = _decode_fields(data)
    raw = _fields_to_dict(fields, _TICK_SUMMARY_SCHEMA)
    if "book_snapshot" in raw:
        raw["book_snapshot"] = decode_book_snapshot(raw["book_snapshot"])
    # Protobuf omits zero-valued fields; ensure all fields have defaults
    defaults = {
        "tick": 0, "open_price": 0.0, "close_price": 0.0,
        "high_price": 0.0, "low_price": 0.0, "volume": 0,
        "num_trades": 0, "consensus": 0.0, "disagreement": 0.0,
        "mass_signal": 0.0,
    }
    for key, val in defaults.items():
        raw.setdefault(key, val)
    return raw
