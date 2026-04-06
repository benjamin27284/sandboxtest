"""Integration test: validates protobuf serialization contract across services.

Ensures that proto_codec.py round-trips correctly for every message type
and that the decoded shapes match what the frontend/gateway expect.
"""

import sys
import os
import struct

# Allow import of proto_codec from the orchestrator service
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../services/agent-orchestrator")
)

from proto_codec import (
    encode_order,
    decode_order,
    decode_execution,
    encode_tick_summary,
    decode_tick_summary,
    encode_book_snapshot,
    decode_book_snapshot,
    decode_market_snapshot,
    SIDE_BUY,
    SIDE_SELL,
    _encode_field_string,
    _encode_field_double,
    _encode_field_varint,
    _encode_field_bytes,
)


def test_order_round_trip():
    """Order encode → decode preserves all fields."""
    order = {
        "order_id": "ORD-001",
        "agent_id": "AGT-0042",
        "action": "buy",
        "type": "limit",
        "price": 101.50,
        "quantity": 25,
        "timestamp": 1700000000,
    }
    encoded = encode_order(order)
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0

    decoded = decode_order(encoded)
    assert decoded["order_id"] == "ORD-001"
    assert decoded["agent_id"] == "AGT-0042"
    assert decoded["side"] == SIDE_BUY
    assert decoded["price"] == 101.50
    assert decoded["quantity"] == 25
    assert decoded["timestamp"] == 1700000000
    print("[PASS] test_order_round_trip")


def test_order_sell_side():
    """Sell orders encode Side=SELL."""
    order = {
        "order_id": "ORD-002",
        "agent_id": "AGT-0001",
        "action": "sell",
        "price": 99.0,
        "quantity": 10,
    }
    decoded = decode_order(encode_order(order))
    assert decoded["side"] == SIDE_SELL
    print("[PASS] test_order_sell_side")


def _build_execution_bytes(
    exec_id: str,
    order_id: str,
    agent_id: str,
    fill_price: float,
    fill_qty: int,
    timestamp: int,
    counter_order_id: str,
    counter_agent_id: str,
    aggressor_side: int,
) -> bytes:
    """Build a protobuf Execution message (simulates C++ LOB engine output)."""
    return (
        _encode_field_string(1, exec_id)
        + _encode_field_string(2, order_id)
        + _encode_field_string(3, agent_id)
        + _encode_field_double(4, fill_price)
        + _encode_field_varint(5, fill_qty)
        + _encode_field_varint(6, timestamp)
        + _encode_field_string(7, counter_order_id)
        + _encode_field_string(8, counter_agent_id)
        + _encode_field_varint(9, aggressor_side)
    )


def test_execution_decode_buy_aggressor():
    """Execution with BUY aggressor → buyer_id = agent_id, seller_id = counter_agent_id."""
    raw = _build_execution_bytes(
        exec_id="EXE-001",
        order_id="ORD-BUY",
        agent_id="AGT-BUYER",
        fill_price=100.25,
        fill_qty=15,
        timestamp=1700000001,
        counter_order_id="ORD-SELL",
        counter_agent_id="AGT-SELLER",
        aggressor_side=SIDE_BUY,
    )
    decoded = decode_execution(raw)

    assert decoded["exec_id"] == "EXE-001"
    assert decoded["fill_price"] == 100.25
    assert decoded["fill_qty"] == 15
    assert decoded["buyer_id"] == "AGT-BUYER"
    assert decoded["seller_id"] == "AGT-SELLER"
    assert decoded["price"] == 100.25
    assert decoded["quantity"] == 15
    print("[PASS] test_execution_decode_buy_aggressor")


def test_execution_decode_sell_aggressor():
    """Execution with SELL aggressor → seller_id = agent_id, buyer_id = counter_agent_id."""
    raw = _build_execution_bytes(
        exec_id="EXE-002",
        order_id="ORD-SELL",
        agent_id="AGT-SELLER",
        fill_price=99.75,
        fill_qty=8,
        timestamp=1700000002,
        counter_order_id="ORD-BUY",
        counter_agent_id="AGT-BUYER",
        aggressor_side=SIDE_SELL,
    )
    decoded = decode_execution(raw)

    assert decoded["buyer_id"] == "AGT-BUYER"
    assert decoded["seller_id"] == "AGT-SELLER"
    print("[PASS] test_execution_decode_sell_aggressor")


def test_tick_summary_round_trip():
    """TickSummary encode → decode preserves OHLCV and MASS fields."""
    summary = {
        "tick": 42,
        "open_price": 100.0,
        "close_price": 101.5,
        "high_price": 102.0,
        "low_price": 99.5,
        "volume": 1234,
        "num_trades": 56,
        "consensus": 0.35,
        "disagreement": 0.12,
        "mass_signal": 0.18,
    }
    encoded = encode_tick_summary(summary)
    decoded = decode_tick_summary(encoded)

    assert decoded["tick"] == 42
    assert decoded["open_price"] == 100.0
    assert decoded["close_price"] == 101.5
    assert decoded["high_price"] == 102.0
    assert decoded["low_price"] == 99.5
    assert decoded["volume"] == 1234
    assert decoded["num_trades"] == 56
    assert abs(decoded["consensus"] - 0.35) < 1e-10
    assert abs(decoded["disagreement"] - 0.12) < 1e-10
    assert abs(decoded["mass_signal"] - 0.18) < 1e-10
    print("[PASS] test_tick_summary_round_trip")


def test_tick_summary_with_book_snapshot():
    """TickSummary with embedded OrderBookSnapshot round-trips."""
    book = {
        "tick": 42,
        "mid_price": 100.5,
        "best_bid": 100.0,
        "best_ask": 101.0,
        "spread": 1.0,
        "bids": [{"price": 100.0, "quantity": 50, "count": 3}],
        "asks": [{"price": 101.0, "quantity": 30, "count": 2}],
    }
    summary = {
        "tick": 42,
        "open_price": 100.0,
        "close_price": 100.5,
        "high_price": 101.0,
        "low_price": 99.8,
        "volume": 500,
        "num_trades": 10,
        "book_snapshot": book,
        "consensus": 0.1,
        "disagreement": 0.05,
        "mass_signal": 0.04,
    }
    decoded = decode_tick_summary(encode_tick_summary(summary))

    assert "book_snapshot" in decoded
    snap = decoded["book_snapshot"]
    assert snap["mid_price"] == 100.5
    assert snap["spread"] == 1.0
    assert len(snap["bids"]) == 1
    assert snap["bids"][0]["price"] == 100.0
    assert snap["bids"][0]["quantity"] == 50
    assert len(snap["asks"]) == 1
    assert snap["asks"][0]["price"] == 101.0
    print("[PASS] test_tick_summary_with_book_snapshot")


def test_book_snapshot_round_trip():
    """OrderBookSnapshot encode → decode preserves all levels."""
    snap = {
        "tick": 10,
        "mid_price": 50.0,
        "best_bid": 49.5,
        "best_ask": 50.5,
        "spread": 1.0,
        "bids": [
            {"price": 49.5, "quantity": 100, "count": 5},
            {"price": 49.0, "quantity": 200, "count": 8},
        ],
        "asks": [
            {"price": 50.5, "quantity": 80, "count": 4},
        ],
    }
    encoded = encode_book_snapshot(snap)
    decoded = decode_book_snapshot(encoded)

    assert decoded["tick"] == 10
    assert decoded["mid_price"] == 50.0
    assert len(decoded["bids"]) == 2
    assert len(decoded["asks"]) == 1
    assert decoded["bids"][1]["quantity"] == 200
    print("[PASS] test_book_snapshot_round_trip")


def test_market_snapshot_decode():
    """MarketSnapshot with embedded OrderBookSnapshot decodes correctly."""
    book_bytes = encode_book_snapshot({
        "tick": 5,
        "mid_price": 100.0,
        "best_bid": 99.5,
        "best_ask": 100.5,
        "spread": 1.0,
        "bids": [{"price": 99.5, "quantity": 40, "count": 2}],
        "asks": [{"price": 100.5, "quantity": 60, "count": 3}],
    })
    # Build MarketSnapshot: mid_price=100.0, spread=1.0, total_volume=200, book_snapshot
    market_bytes = (
        _encode_field_double(1, 100.0)
        + _encode_field_double(2, 1.0)
        + _encode_field_varint(3, 200)
        + _encode_field_bytes(4, book_bytes)
    )
    decoded = decode_market_snapshot(market_bytes)

    assert decoded["mid_price"] == 100.0
    assert decoded["spread"] == 1.0
    assert decoded["total_volume"] == 200
    assert "book_snapshot" in decoded
    assert decoded["book_snapshot"]["best_bid"] == 99.5
    print("[PASS] test_market_snapshot_decode")


def test_frontend_execution_shape():
    """Decoded Execution has all fields the frontend TypeScript Execution interface expects."""
    raw = _build_execution_bytes(
        exec_id="E1", order_id="O1", agent_id="A1",
        fill_price=100.0, fill_qty=10, timestamp=123,
        counter_order_id="O2", counter_agent_id="A2",
        aggressor_side=SIDE_BUY,
    )
    decoded = decode_execution(raw)

    required_keys = {
        "exec_id", "order_id", "agent_id",
        "fill_price", "fill_qty", "timestamp",
        "counter_order_id", "counter_agent_id", "aggressor_side",
        "buyer_id", "seller_id", "price", "quantity",
    }
    missing = required_keys - set(decoded.keys())
    assert not missing, f"Missing keys for frontend: {missing}"
    print("[PASS] test_frontend_execution_shape")


def test_frontend_tick_summary_shape():
    """Decoded TickSummary has all fields the frontend TypeScript TickSummary interface expects."""
    summary = {
        "tick": 1, "open_price": 100.0, "close_price": 100.5,
        "high_price": 101.0, "low_price": 99.5,
        "volume": 100, "num_trades": 5,
        "consensus": 0.1, "disagreement": 0.05, "mass_signal": 0.04,
    }
    decoded = decode_tick_summary(encode_tick_summary(summary))

    required_keys = {
        "tick", "open_price", "close_price", "high_price", "low_price",
        "volume", "num_trades", "consensus", "disagreement", "mass_signal",
    }
    missing = required_keys - set(decoded.keys())
    assert not missing, f"Missing keys for frontend: {missing}"
    print("[PASS] test_frontend_tick_summary_shape")


if __name__ == "__main__":
    test_order_round_trip()
    test_order_sell_side()
    test_execution_decode_buy_aggressor()
    test_execution_decode_sell_aggressor()
    test_tick_summary_round_trip()
    test_tick_summary_with_book_snapshot()
    test_book_snapshot_round_trip()
    test_market_snapshot_decode()
    test_frontend_execution_shape()
    test_frontend_tick_summary_shape()
    print("\nAll proto contract tests passed.")
