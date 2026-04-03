"""Discrete-tick Limit Order Book (LOB) and Matching Engine.

A lightweight prototype for a multi-agent financial simulation.
Uses heapq for O(log n) order insertion and best-price retrieval.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(order=False)
class Order:
    agent_id: str
    side: Side
    price: float
    quantity: int
    tick_submitted: int
    _seq: int = field(default=0, repr=False)  # insertion sequence for FIFO tie-breaking

    # Remaining quantity tracks partial fills; starts equal to quantity.
    remaining: int = field(init=False)

    def __post_init__(self) -> None:
        self.remaining = self.quantity


class LimitOrderBook:
    """Limit Order Book backed by two heapq priority queues.

    Bids: max-heap (negated prices so heapq min-heap acts as max-heap).
    Asks: min-heap (natural ordering).

    Tie-breaking within the same price level uses insertion sequence (FIFO).
    """

    def __init__(self) -> None:
        # Each heap entry: (sort_key, sequence, Order)
        self._bids: list[tuple[float, int, Order]] = []
        self._asks: list[tuple[float, int, Order]] = []
        self._seq: int = 0

    # -- Public API -----------------------------------------------------------

    def submit(self, order: Order) -> None:
        """Insert an order into the appropriate side of the book."""
        order._seq = self._seq
        self._seq += 1

        if order.side is Side.BUY:
            # Negate price so highest bid sits at heap top
            heapq.heappush(self._bids, (-order.price, order._seq, order))
        else:
            heapq.heappush(self._asks, (order.price, order._seq, order))

    def match_orders(self) -> list[dict]:
        """Match crossing orders and return a list of execution records.

        An execution happens when the best bid price >= the best ask price.
        The execution price is the price of the *resting* (earlier) order.
        """
        executions: list[dict] = []

        while self._bids and self._asks:
            # Peek at best bid / ask
            neg_bid_price, bid_seq, best_bid = self._bids[0]
            ask_price, ask_seq, best_ask = self._asks[0]

            best_bid_price = -neg_bid_price

            if best_bid_price < ask_price:
                break  # no crossing

            # Determine execution price: earlier order's price (price-time priority)
            if best_bid._seq < best_ask._seq:
                exec_price = best_bid.price
            else:
                exec_price = best_ask.price

            # Determine fill quantity
            fill_qty = min(best_bid.remaining, best_ask.remaining)

            executions.append(
                {
                    "price": exec_price,
                    "quantity": fill_qty,
                    "buyer_id": best_bid.agent_id,
                    "seller_id": best_ask.agent_id,
                }
            )

            best_bid.remaining -= fill_qty
            best_ask.remaining -= fill_qty

            # Remove fully-filled orders from the heap
            if best_bid.remaining == 0:
                heapq.heappop(self._bids)
            if best_ask.remaining == 0:
                heapq.heappop(self._asks)

        return executions

    def get_mid_price(self) -> Optional[float]:
        """Return the mid-price (average of best bid and best ask).

        Returns None if either side of the book is empty.
        """
        self._prune_filled(self._bids)
        self._prune_filled(self._asks)

        if not self._bids or not self._asks:
            return None

        best_bid = -self._bids[0][0]
        best_ask = self._asks[0][0]
        return (best_bid + best_ask) / 2.0

    # -- Introspection --------------------------------------------------------

    def best_bid(self) -> Optional[float]:
        self._prune_filled(self._bids)
        return -self._bids[0][0] if self._bids else None

    def best_ask(self) -> Optional[float]:
        self._prune_filled(self._asks)
        return self._asks[0][0] if self._asks else None

    def bid_depth(self) -> list[tuple[float, int]]:
        """Return list of (price, total_remaining_qty) for all bid levels."""
        return self._aggregate(self._bids, negate=True)

    def ask_depth(self) -> list[tuple[float, int]]:
        """Return list of (price, total_remaining_qty) for all ask levels."""
        return self._aggregate(self._asks, negate=False)

    # -- Internals ------------------------------------------------------------

    @staticmethod
    def _prune_filled(heap: list) -> None:
        """Remove fully-filled orders sitting at the top of a heap."""
        while heap and heap[0][2].remaining == 0:
            heapq.heappop(heap)

    @staticmethod
    def _aggregate(heap: list, *, negate: bool) -> list[tuple[float, int]]:
        levels: dict[float, int] = {}
        for entry in heap:
            order = entry[2]
            if order.remaining == 0:
                continue
            price = -entry[0] if negate else entry[0]
            levels[price] = levels.get(price, 0) + order.remaining
        return sorted(levels.items(), reverse=negate)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lob = LimitOrderBook()

    # Tick 0 – seed the book with resting orders
    orders = [
        Order(agent_id="A1", side=Side.BUY,  price=100.0, quantity=10, tick_submitted=0),
        Order(agent_id="A2", side=Side.BUY,  price=99.5,  quantity=5,  tick_submitted=0),
        Order(agent_id="A3", side=Side.BUY,  price=99.0,  quantity=20, tick_submitted=0),
        Order(agent_id="B1", side=Side.SELL, price=101.0, quantity=8,  tick_submitted=0),
        Order(agent_id="B2", side=Side.SELL, price=101.5, quantity=15, tick_submitted=0),
        Order(agent_id="B3", side=Side.SELL, price=102.0, quantity=10, tick_submitted=0),
    ]

    for o in orders:
        lob.submit(o)

    print("=== LOB after tick 0 (no crossing orders) ===")
    print(f"  Best bid: {lob.best_bid()}  |  Best ask: {lob.best_ask()}")
    print(f"  Mid price: {lob.get_mid_price()}")
    print(f"  Bid depth: {lob.bid_depth()}")
    print(f"  Ask depth: {lob.ask_depth()}")

    execs = lob.match_orders()
    print(f"  Executions: {execs}\n")

    # Tick 1 – aggressive buy crosses the spread
    aggressive_buy = Order(
        agent_id="A4", side=Side.BUY, price=101.5, quantity=12, tick_submitted=1
    )
    lob.submit(aggressive_buy)

    print("=== After tick 1: aggressive BUY @ 101.5 x 12 ===")
    execs = lob.match_orders()
    for e in execs:
        print(f"  FILL  {e['buyer_id']} <- {e['seller_id']}  "
              f"{e['quantity']}@{e['price']}")

    print(f"\n  Best bid: {lob.best_bid()}  |  Best ask: {lob.best_ask()}")
    print(f"  Mid price: {lob.get_mid_price()}")
    print(f"  Bid depth: {lob.bid_depth()}")
    print(f"  Ask depth: {lob.ask_depth()}")

    # Tick 2 – aggressive sell crosses the spread
    aggressive_sell = Order(
        agent_id="B4", side=Side.SELL, price=99.0, quantity=18, tick_submitted=2
    )
    lob.submit(aggressive_sell)

    print("\n=== After tick 2: aggressive SELL @ 99.0 x 18 ===")
    execs = lob.match_orders()
    for e in execs:
        print(f"  FILL  {e['buyer_id']} <- {e['seller_id']}  "
              f"{e['quantity']}@{e['price']}")

    print(f"\n  Best bid: {lob.best_bid()}  |  Best ask: {lob.best_ask()}")
    print(f"  Mid price: {lob.get_mid_price()}")
    print(f"  Bid depth: {lob.bid_depth()}")
    print(f"  Ask depth: {lob.ask_depth()}")
