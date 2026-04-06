"""Lightweight Python matching engine — mirrors the C++ LOB logic.

Used for multi-asset event-driven simulation where each impacted asset
gets its own OrderBook.  The agents submit orders, the book matches
them with price-time FIFO priority, and prices emerge from interaction.

Not a replacement for the high-performance C++ engine (which handles the
primary asset at 854K+ orders/sec), but efficient enough for running
15 parallel short simulations of 30 ticks each.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


# ─── Data types ────────────────────────────────────────────────────────────

@dataclass
class Order:
    order_id: str
    agent_id: str
    side: str            # "buy" or "sell"
    price: float
    quantity: int
    timestamp: int = 0
    remaining: int = 0
    sequence: int = 0

    def is_filled(self) -> bool:
        return self.remaining <= 0


@dataclass
class Execution:
    exec_id: str
    order_id: str
    agent_id: str
    fill_price: float
    fill_qty: int
    timestamp: int
    counter_order_id: str
    counter_agent_id: str
    aggressor_side: str   # "buy" or "sell"

    @property
    def buyer_id(self) -> str:
        return self.agent_id if self.aggressor_side == "buy" else self.counter_agent_id

    @property
    def seller_id(self) -> str:
        return self.agent_id if self.aggressor_side == "sell" else self.counter_agent_id


@dataclass
class BookSnapshot:
    mid_price: float
    best_bid: float
    best_ask: float
    spread: float
    bid_levels: list[dict] = field(default_factory=list)
    ask_levels: list[dict] = field(default_factory=list)


# ─── OrderBook ─────────────────────────────────────────────────────────────

class PyOrderBook:
    """Price-time FIFO matching engine in pure Python.

    Mirrors the C++ OrderBook semantics:
      - Bids sorted descending by price, FIFO within level
      - Asks sorted ascending by price, FIFO within level
      - Fill price = passive (resting) order's price
      - Market orders get synthetic prices to guarantee crossing
    """

    def __init__(self, symbol: str = "SIM", initial_price: float = 100.0) -> None:
        self.symbol = symbol
        self._bids: dict[float, list[Order]] = {}   # price → [orders] (descending)
        self._asks: dict[float, list[Order]] = {}   # price → [orders] (ascending)
        self._seq = 0
        self._exec_counter = 0
        self._initial_price = initial_price
        self._last_trade_price = initial_price

    def add_order(self, order: Order) -> None:
        order.remaining = order.quantity
        order.sequence = self._seq
        self._seq += 1

        if order.side == "buy":
            self._bids.setdefault(order.price, []).append(order)
        else:
            self._asks.setdefault(order.price, []).append(order)

    def match_orders(self) -> list[Execution]:
        """Sweep the book while best_bid >= best_ask."""
        fills: list[Execution] = []

        while self._bids and self._asks:
            best_bid_price = max(self._bids.keys())
            best_ask_price = min(self._asks.keys())

            if best_bid_price < best_ask_price:
                break

            bid_queue = self._bids[best_bid_price]
            ask_queue = self._asks[best_ask_price]

            # Drain stale orders
            while bid_queue and bid_queue[0].is_filled():
                bid_queue.pop(0)
            while ask_queue and ask_queue[0].is_filled():
                ask_queue.pop(0)

            if not bid_queue:
                del self._bids[best_bid_price]
                continue
            if not ask_queue:
                del self._asks[best_ask_price]
                continue

            bid = bid_queue[0]
            ask = ask_queue[0]

            fill_qty = min(bid.remaining, ask.remaining)

            # Fill at passive order's price
            bid_is_aggressor = bid.sequence > ask.sequence
            fill_price = ask.price if bid_is_aggressor else bid.price
            aggressor = bid if bid_is_aggressor else ask
            passive = ask if bid_is_aggressor else bid

            self._exec_counter += 1
            fills.append(Execution(
                exec_id=f"EXE-{self.symbol}-{self._exec_counter:06d}",
                order_id=aggressor.order_id,
                agent_id=aggressor.agent_id,
                fill_price=fill_price,
                fill_qty=fill_qty,
                timestamp=int(time.time_ns()),
                counter_order_id=passive.order_id,
                counter_agent_id=passive.agent_id,
                aggressor_side=aggressor.side,
            ))

            self._last_trade_price = fill_price

            bid.remaining -= fill_qty
            ask.remaining -= fill_qty

            if bid.is_filled():
                bid_queue.pop(0)
            if ask.is_filled():
                ask_queue.pop(0)

            if not bid_queue:
                del self._bids[best_bid_price]
            if not ask_queue:
                del self._asks[best_ask_price]

        return fills

    @property
    def mid_price(self) -> float:
        if not self._bids or not self._asks:
            return self._last_trade_price
        best_bid = max(self._bids.keys())
        best_ask = min(self._asks.keys())
        return (best_bid + best_ask) / 2.0

    @property
    def spread(self) -> float:
        if not self._bids or not self._asks:
            return 0.01
        return min(self._asks.keys()) - max(self._bids.keys())

    @property
    def best_bid(self) -> Optional[float]:
        return max(self._bids.keys()) if self._bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return min(self._asks.keys()) if self._asks else None

    @property
    def last_trade_price(self) -> float:
        return self._last_trade_price

    def snapshot(self, depth: int = 5) -> BookSnapshot:
        bid_prices = sorted(self._bids.keys(), reverse=True)[:depth]
        ask_prices = sorted(self._asks.keys())[:depth]

        bid_levels = []
        for p in bid_prices:
            orders = [o for o in self._bids[p] if not o.is_filled()]
            if orders:
                bid_levels.append({
                    "price": p,
                    "quantity": sum(o.remaining for o in orders),
                    "count": len(orders),
                })

        ask_levels = []
        for p in ask_prices:
            orders = [o for o in self._asks[p] if not o.is_filled()]
            if orders:
                ask_levels.append({
                    "price": p,
                    "quantity": sum(o.remaining for o in orders),
                    "count": len(orders),
                })

        bb = max(self._bids.keys()) if self._bids else self._last_trade_price
        ba = min(self._asks.keys()) if self._asks else self._last_trade_price

        return BookSnapshot(
            mid_price=self.mid_price,
            best_bid=bb,
            best_ask=ba,
            spread=ba - bb if self._bids and self._asks else 0.01,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
        )
