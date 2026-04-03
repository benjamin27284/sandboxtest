"""Tests for the Python matching engine used in multi-asset simulation."""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../services/agent-orchestrator")
)

from matching_engine import PyOrderBook, Order


def test_basic_match():
    """Buy and sell orders crossing should produce a fill."""
    book = PyOrderBook("TEST", 100.0)

    # Resting bid
    book.add_order(Order("B1", "A1", "buy", 100.0, 10))
    fills = book.match_orders()
    assert len(fills) == 0
    assert book.best_bid == 100.0

    # Crossing sell
    book.add_order(Order("S1", "A2", "sell", 99.0, 5))
    fills = book.match_orders()
    assert len(fills) == 1
    assert fills[0].fill_qty == 5
    assert fills[0].fill_price == 100.0  # passive (bid) price
    assert fills[0].buyer_id == "A1"
    assert fills[0].seller_id == "A2"
    print("[PASS] test_basic_match")


def test_no_crossing():
    """Orders that don't cross should not match."""
    book = PyOrderBook("TEST", 100.0)
    book.add_order(Order("B1", "A1", "buy", 99.0, 10))
    book.add_order(Order("S1", "A2", "sell", 101.0, 10))
    fills = book.match_orders()
    assert len(fills) == 0
    assert book.mid_price == 100.0
    assert book.spread == 2.0
    print("[PASS] test_no_crossing")


def test_partial_fill():
    """Partial fills should leave remaining quantity."""
    book = PyOrderBook("TEST", 100.0)
    book.add_order(Order("B1", "A1", "buy", 100.0, 20))
    book.match_orders()  # no match yet
    book.add_order(Order("S1", "A2", "sell", 100.0, 8))
    fills = book.match_orders()
    assert len(fills) == 1
    assert fills[0].fill_qty == 8
    # Bid should have 12 remaining
    assert book.best_bid == 100.0
    print("[PASS] test_partial_fill")


def test_multi_level_sweep():
    """Aggressive order sweeps multiple price levels."""
    book = PyOrderBook("TEST", 100.0)
    # 3 bid levels
    book.add_order(Order("B0", "A0", "buy", 100.0, 10))
    book.add_order(Order("B1", "A1", "buy", 99.5, 10))
    book.add_order(Order("B2", "A2", "buy", 99.0, 10))
    book.match_orders()  # no asks

    # Aggressive sell sweeps 25 through all levels
    book.add_order(Order("S0", "SELLER", "sell", 98.0, 25))
    fills = book.match_orders()
    assert len(fills) == 3
    assert fills[0].fill_qty == 10  # 100.0
    assert fills[1].fill_qty == 10  # 99.5
    assert fills[2].fill_qty == 5   # 99.0 partial
    print("[PASS] test_multi_level_sweep")


def test_price_time_priority():
    """Within same price level, earlier orders fill first (FIFO)."""
    book = PyOrderBook("TEST", 100.0)
    book.add_order(Order("B1", "FIRST", "buy", 100.0, 10))
    book.add_order(Order("B2", "SECOND", "buy", 100.0, 10))
    book.match_orders()

    # Sell 5 — should fill against FIRST
    book.add_order(Order("S1", "SELLER", "sell", 100.0, 5))
    fills = book.match_orders()
    assert len(fills) == 1
    assert fills[0].buyer_id == "FIRST"
    print("[PASS] test_price_time_priority")


def test_fill_price_is_passive():
    """Fill price should be the passive (resting) order's price."""
    book = PyOrderBook("TEST", 100.0)
    # Resting ask at 101
    book.add_order(Order("S1", "A1", "sell", 101.0, 10))
    book.match_orders()

    # Aggressive buy at 102 — should fill at 101 (passive price)
    book.add_order(Order("B1", "A2", "buy", 102.0, 5))
    fills = book.match_orders()
    assert len(fills) == 1
    assert fills[0].fill_price == 101.0
    print("[PASS] test_fill_price_is_passive")


def test_mid_price_and_spread():
    """Mid price and spread calculations."""
    book = PyOrderBook("TEST", 100.0)
    book.add_order(Order("B1", "A1", "buy", 99.0, 10))
    book.add_order(Order("S1", "A2", "sell", 101.0, 10))
    book.match_orders()
    assert book.mid_price == 100.0
    assert book.spread == 2.0
    print("[PASS] test_mid_price_and_spread")


def test_snapshot():
    """Book snapshot contains bid/ask levels."""
    book = PyOrderBook("TEST", 100.0)
    book.add_order(Order("B1", "A1", "buy", 99.0, 10))
    book.add_order(Order("B2", "A2", "buy", 98.0, 20))
    book.add_order(Order("S1", "A3", "sell", 101.0, 15))
    book.match_orders()

    snap = book.snapshot(depth=5)
    assert len(snap.bid_levels) == 2
    assert len(snap.ask_levels) == 1
    assert snap.bid_levels[0]["price"] == 99.0
    assert snap.bid_levels[0]["quantity"] == 10
    assert snap.ask_levels[0]["price"] == 101.0
    print("[PASS] test_snapshot")


def test_last_trade_price():
    """Last trade price updates on fill."""
    book = PyOrderBook("TEST", 50.0)
    assert book.last_trade_price == 50.0

    book.add_order(Order("B1", "A1", "buy", 55.0, 10))
    book.add_order(Order("S1", "A2", "sell", 55.0, 10))
    fills = book.match_orders()
    assert len(fills) == 1
    assert book.last_trade_price == 55.0
    print("[PASS] test_last_trade_price")


def test_empty_book():
    """Empty book returns initial price as mid."""
    book = PyOrderBook("GOLD", 2350.0)
    assert book.mid_price == 2350.0
    assert book.last_trade_price == 2350.0
    assert book.best_bid is None
    assert book.best_ask is None
    print("[PASS] test_empty_book")


if __name__ == "__main__":
    test_basic_match()
    test_no_crossing()
    test_partial_fill()
    test_multi_level_sweep()
    test_price_time_priority()
    test_fill_price_is_passive()
    test_mid_price_and_spread()
    test_snapshot()
    test_last_trade_price()
    test_empty_book()
    print("\nAll matching engine tests passed.")
