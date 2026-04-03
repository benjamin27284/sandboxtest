"""Programmatic trading agents for the LOB simulation.

BaseAgent  – abstract base with cash / inventory accounting.
NoiseTrader    – random limit orders around the mid-price.
MomentumTrader – trades on a 5-period moving average crossover signal.
InstitutionalMarketMaker – liquidity provider with inventory risk limits.
FundamentalValueFund     – DCF-driven with VaR + drawdown constraints.
QuantMomentumFund        – volatility-targeted momentum with Sharpe guard.
RetailSentimentTrader    – narrative-driven with margin call liquidation.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

from lob import LimitOrderBook, Order, Side


class BaseAgent(ABC):
    """Common state and interface for every trading agent."""

    def __init__(self, agent_id: str, cash: float = 10_000.0, inventory: int = 0) -> None:
        self.agent_id = agent_id
        self.cash = cash
        self.inventory = inventory

    @abstractmethod
    def act(self, tick: int, lob: LimitOrderBook, last_price: Optional[float]) -> None:
        """Called once per tick. The agent may submit orders to *lob*."""

    def update_on_fill(self, price: float, quantity: int, side: Side) -> None:
        """Adjust cash and inventory after one of our orders is filled."""
        if side is Side.BUY:
            self.cash -= price * quantity
            self.inventory += quantity
        else:
            self.cash += price * quantity
            self.inventory -= quantity

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.agent_id!r}, "
            f"cash={self.cash:.2f}, inv={self.inventory})"
        )


# ---------------------------------------------------------------------------
# Noise trader
# ---------------------------------------------------------------------------

class NoiseTrader(BaseAgent):
    """Randomly submits buy/sell limit orders near the mid-price.

    At each tick it randomly chooses BUY, SELL, or HOLD (equal probability).
    Order price is mid ± uniform(0, spread_width), quantity in [1, max_qty].
    """

    def __init__(
        self,
        agent_id: str,
        cash: float = 10_000.0,
        inventory: int = 0,
        max_qty: int = 5,
        spread_width: float = 2.0,
    ) -> None:
        super().__init__(agent_id, cash, inventory)
        self.max_qty = max_qty
        self.spread_width = spread_width

    def act(self, tick: int, lob: LimitOrderBook, last_price: Optional[float]) -> None:
        mid = lob.get_mid_price()
        ref = mid if mid is not None else (last_price if last_price is not None else 100.0)

        action = random.choice(["buy", "sell", "hold"])
        if action == "hold":
            return

        qty = random.randint(1, self.max_qty)
        # Offset may be positive or negative — allowing aggressive orders
        # that cross the spread roughly 40% of the time.
        offset = round(random.uniform(-0.4 * self.spread_width, self.spread_width), 2)

        if action == "buy":
            price = round(ref - offset, 2)   # negative offset → price above mid
            lob.submit(Order(self.agent_id, Side.BUY, price, qty, tick))
        else:
            price = round(ref + offset, 2)   # negative offset → price below mid
            lob.submit(Order(self.agent_id, Side.SELL, price, qty, tick))


# ---------------------------------------------------------------------------
# Momentum trader
# ---------------------------------------------------------------------------

class MomentumTrader(BaseAgent):
    """Trades on a simple moving-average crossover of execution prices.

    Maintains a window of the last *window* execution prices.
    If current price > MA  → submit a buy order (momentum is up).
    If current price < MA  → submit a sell order (momentum is down).
    If MA is undefined (< *window* observations) → do nothing.
    """

    def __init__(
        self,
        agent_id: str,
        cash: float = 10_000.0,
        inventory: int = 0,
        window: int = 5,
        order_qty: int = 3,
        offset: float = 0.10,
    ) -> None:
        super().__init__(agent_id, cash, inventory)
        self.window = window
        self.order_qty = order_qty
        self.offset = offset
        self._price_history: deque[float] = deque(maxlen=window)

    def observe_price(self, price: float) -> None:
        """Record an execution price into the rolling window."""
        self._price_history.append(price)

    def act(self, tick: int, lob: LimitOrderBook, last_price: Optional[float]) -> None:
        if last_price is not None:
            self.observe_price(last_price)

        if len(self._price_history) < self.window:
            return  # not enough data

        ma = sum(self._price_history) / len(self._price_history)
        current = self._price_history[-1]

        if current > ma:
            # Momentum up → buy slightly above mid
            ref = lob.get_mid_price() or current
            price = round(ref + self.offset, 2)
            lob.submit(Order(self.agent_id, Side.BUY, price, self.order_qty, tick))
        elif current < ma:
            # Momentum down → sell slightly below mid
            ref = lob.get_mid_price() or current
            price = round(ref - self.offset, 2)
            lob.submit(Order(self.agent_id, Side.SELL, price, self.order_qty, tick))
        # current == ma → no signal, do nothing


# ---------------------------------------------------------------------------
# Institutional Market Maker
# ---------------------------------------------------------------------------

class InstitutionalMarketMaker(BaseAgent):
    """Liquidity provider that posts both bid and ask around mid-price.

    Inventory risk limit: if absolute inventory exceeds max_inventory_imbalance,
    only the side that reduces inventory is submitted (mean reversion to zero).
    """

    def __init__(
        self,
        agent_id: str,
        cash: float = 10_000.0,
        inventory: int = 0,
        inventory_limit: int = 500,
        spread_width: float = 0.002,
        max_inventory_imbalance: int = 200,
    ) -> None:
        super().__init__(agent_id, cash, inventory)
        self.inventory_limit = inventory_limit
        self.spread_width = spread_width
        self.max_inventory_imbalance = max_inventory_imbalance

    def act(self, tick: int, lob: LimitOrderBook, last_price: Optional[float]) -> list[Order]:
        mid = lob.get_mid_price()
        ref = mid if mid is not None else (last_price if last_price is not None else 100.0)

        bid_price = round(ref * (1 - self.spread_width / 2), 2)
        ask_price = round(ref * (1 + self.spread_width / 2), 2)
        qty = random.randint(10, 50)

        orders: list[Order] = []

        if abs(self.inventory) > self.max_inventory_imbalance:
            # Only submit the side that reduces inventory
            if self.inventory > 0:
                # Too long → sell only
                o = Order(self.agent_id, Side.SELL, ask_price, qty, tick)
                lob.submit(o)
                orders.append(o)
            else:
                # Too short → buy only
                o = Order(self.agent_id, Side.BUY, bid_price, qty, tick)
                lob.submit(o)
                orders.append(o)
        else:
            bid = Order(self.agent_id, Side.BUY, bid_price, qty, tick)
            ask = Order(self.agent_id, Side.SELL, ask_price, qty, tick)
            lob.submit(bid)
            lob.submit(ask)
            orders.extend([bid, ask])

        return orders


# ---------------------------------------------------------------------------
# Fundamental Value Fund
# ---------------------------------------------------------------------------

class FundamentalValueFund(BaseAgent):
    """Trades toward a fundamental fair value with VaR and drawdown constraints.

    Capital preservation: stops trading when max drawdown is breached.
    Risk management: stops trading when position VaR exceeds limit.
    """

    def __init__(
        self,
        agent_id: str,
        cash: float = 10_000.0,
        inventory: int = 0,
        fair_value: float = 100.0,
        var_limit: float = 0.02,
        max_drawdown_limit: float = 0.15,
    ) -> None:
        super().__init__(agent_id, cash, inventory)
        self.fair_value = fair_value
        self.var_limit = var_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.peak_portfolio_value = cash

    def act(self, tick: int, lob: LimitOrderBook, last_price: Optional[float]) -> list[Order]:
        ref = last_price if last_price is not None else 100.0

        portfolio_value = self.cash + self.inventory * ref

        # Drawdown check
        if self.peak_portfolio_value > 0:
            drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            if drawdown > self.max_drawdown_limit:
                return []  # capital preservation mode

        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)

        # VaR check
        daily_vol = 0.02
        position_var = abs(self.inventory) * ref * daily_vol * 1.645
        if portfolio_value > 0 and position_var > self.var_limit * portfolio_value:
            return []

        orders: list[Order] = []
        if ref < self.fair_value * 0.98:
            qty = random.randint(5, 20)
            o = Order(self.agent_id, Side.BUY, round(self.fair_value, 2), qty, tick)
            lob.submit(o)
            orders.append(o)
        elif ref > self.fair_value * 1.02:
            qty = random.randint(5, 20)
            o = Order(self.agent_id, Side.SELL, round(self.fair_value, 2), qty, tick)
            lob.submit(o)
            orders.append(o)

        return orders


# ---------------------------------------------------------------------------
# Quant Momentum Fund
# ---------------------------------------------------------------------------

class QuantMomentumFund(BaseAgent):
    """Volatility-targeted momentum strategy with Sharpe guard.

    Scales position size inversely with realized volatility to maintain
    a constant risk budget. Halts trading if realized Sharpe is deeply negative.
    """

    def __init__(
        self,
        agent_id: str,
        cash: float = 10_000.0,
        inventory: int = 0,
        sharpe_target: float = 1.5,
        vol_target: float = 0.15,
        window: int = 20,
    ) -> None:
        super().__init__(agent_id, cash, inventory)
        self.sharpe_target = sharpe_target
        self.vol_target = vol_target
        self.window = window
        self.price_history: list[float] = []
        self.returns: list[float] = []

    def observe_price(self, price: float) -> None:
        self.price_history.append(price)
        if len(self.price_history) > 1:
            log_ret = math.log(self.price_history[-1] / self.price_history[-2])
            self.returns.append(log_ret)
        # Keep only last `window` entries
        if len(self.price_history) > self.window:
            self.price_history = self.price_history[-self.window:]
        if len(self.returns) > self.window:
            self.returns = self.returns[-self.window:]

    def act(self, tick: int, lob: LimitOrderBook, last_price: Optional[float]) -> list[Order]:
        ref = last_price if last_price is not None else 100.0
        self.observe_price(ref)

        if len(self.price_history) < 5:
            return []

        # Realized volatility
        if len(self.returns) >= 2:
            mean_r = sum(self.returns) / len(self.returns)
            variance = sum((r - mean_r) ** 2 for r in self.returns) / (len(self.returns) - 1)
            realized_vol = math.sqrt(variance) * math.sqrt(252)
        else:
            realized_vol = 0.15

        vol_scalar = self.vol_target / max(realized_vol, 0.01)
        base_qty = max(1, int(10 * vol_scalar))

        # Momentum signal: 5-period return
        momentum = ref / self.price_history[-5] - 1

        # Sharpe check (soft guard)
        if len(self.returns) >= self.window:
            mean_r = sum(self.returns) / len(self.returns)
            ann_mean = mean_r * 252
            realized_sharpe = ann_mean / max(realized_vol, 0.01)
            if realized_sharpe < -1.0:
                return []  # deeply negative Sharpe — stop trading

        orders: list[Order] = []
        if momentum > 0.005:
            price = round(ref * 1.001, 2)
            o = Order(self.agent_id, Side.BUY, price, base_qty, tick)
            lob.submit(o)
            orders.append(o)
        elif momentum < -0.005:
            price = round(ref * 0.999, 2)
            o = Order(self.agent_id, Side.SELL, price, base_qty, tick)
            lob.submit(o)
            orders.append(o)

        return orders


# ---------------------------------------------------------------------------
# Retail Sentiment Trader
# ---------------------------------------------------------------------------

class RetailSentimentTrader(BaseAgent):
    """Narrative-driven retail trader with margin call forced liquidation.

    Overweights noisy sentiment signals. Subject to margin calls when
    portfolio value drops below margin_rate * initial_cash.
    """

    def __init__(
        self,
        agent_id: str,
        cash: float = 10_000.0,
        inventory: int = 0,
        margin_rate: float = 0.5,
    ) -> None:
        super().__init__(agent_id, cash, inventory)
        self.margin_rate = margin_rate
        self.narrative_bias: float = 0.0
        self._initial_cash = cash

    def update_narrative(self, news_sentiment: float) -> None:
        """Update narrative bias from a news sentiment signal in [-1, 1]."""
        amplification = random.uniform(1.5, 3.0)
        self.narrative_bias = 0.7 * self.narrative_bias + 0.3 * news_sentiment * amplification
        self.narrative_bias = max(-1.0, min(1.0, self.narrative_bias))

    def act(self, tick: int, lob: LimitOrderBook, last_price: Optional[float]) -> list[Order]:
        ref = last_price if last_price is not None else 100.0
        orders: list[Order] = []

        # Margin call check
        portfolio_value = self.cash + self.inventory * ref
        if portfolio_value < self._initial_cash * self.margin_rate:
            if self.inventory > 0:
                o = Order(self.agent_id, Side.SELL, round(ref * 0.98, 2),
                          self.inventory, tick)
                lob.submit(o)
                orders.append(o)
            return orders

        qty = random.randint(1, 30)
        if self.narrative_bias > 0.2:
            price = round(ref * random.uniform(1.00, 1.02), 2)
            o = Order(self.agent_id, Side.BUY, price, qty, tick)
            lob.submit(o)
            orders.append(o)
        elif self.narrative_bias < -0.2:
            price = round(ref * random.uniform(0.98, 1.00), 2)
            o = Order(self.agent_id, Side.SELL, price, qty, tick)
            lob.submit(o)
            orders.append(o)
        else:
            roll = random.random()
            if roll < 0.4:
                price = round(ref * random.uniform(1.00, 1.01), 2)
                o = Order(self.agent_id, Side.BUY, price, qty, tick)
                lob.submit(o)
                orders.append(o)
            elif roll < 0.8:
                price = round(ref * random.uniform(0.99, 1.00), 2)
                o = Order(self.agent_id, Side.SELL, price, qty, tick)
                lob.submit(o)
                orders.append(o)
            # else: hold (20% probability)

        return orders


# ---------------------------------------------------------------------------
# Simulation harness
# ---------------------------------------------------------------------------

def settle_fills(
    executions: list[dict],
    agents_by_id: dict[str, BaseAgent],
    momentum_traders: list[MomentumTrader],
    lob: Optional[LimitOrderBook] = None,
    tick: int = 0,
) -> Optional[float]:
    """Update agent cash/inventory for every fill and feed prices to momentum traders.

    After settlement, agents with a var_limit attribute are checked for VaR
    breach and forced to partially deleverage (25% unwind) if necessary.

    Returns the last execution price (or None if no fills).
    """
    last_price: Optional[float] = None
    for ex in executions:
        price, qty = ex["price"], ex["quantity"]
        buyer = agents_by_id.get(ex["buyer_id"])
        seller = agents_by_id.get(ex["seller_id"])
        if buyer:
            buyer.update_on_fill(price, qty, Side.BUY)
        if seller:
            seller.update_on_fill(price, qty, Side.SELL)

        # VaR-triggered mandatory deleveraging
        if lob is not None:
            for agent in (buyer, seller):
                if agent is None:
                    continue
                if not hasattr(agent, "var_limit"):
                    continue
                portfolio_value = agent.cash + agent.inventory * price
                position_var = abs(agent.inventory) * price * 0.02 * 1.645
                if portfolio_value > 0 and position_var > agent.var_limit * portfolio_value:
                    unwind_qty = max(1, abs(agent.inventory) // 4)
                    side = Side.SELL if agent.inventory > 0 else Side.BUY
                    unwind_price = price * (0.97 if side is Side.SELL else 1.03)
                    lob.submit(Order(
                        agent_id=agent.agent_id, side=side,
                        price=round(unwind_price, 2), quantity=unwind_qty,
                        tick_submitted=tick,
                    ))

        # Feed execution price to momentum traders
        for mt in momentum_traders:
            mt.observe_price(price)
        last_price = price
    return last_price


if __name__ == "__main__":
    random.seed(42)

    lob = LimitOrderBook()

    # --- Create agents -------------------------------------------------------
    noise_traders: list[NoiseTrader] = [
        NoiseTrader(agent_id=f"N{i:02d}") for i in range(10)
    ]
    momentum_traders: list[MomentumTrader] = [
        MomentumTrader(agent_id="M00"),
        MomentumTrader(agent_id="M01"),
    ]
    all_agents: list[BaseAgent] = noise_traders + momentum_traders  # type: ignore[list-item]
    agents_by_id = {a.agent_id: a for a in all_agents}

    # Seed the book so there's an initial mid-price
    lob.submit(Order("SEED", Side.BUY,  price=99.0, quantity=50, tick_submitted=0))
    lob.submit(Order("SEED", Side.SELL, price=101.0, quantity=50, tick_submitted=0))

    last_price: Optional[float] = None
    total_fills = 0

    print(f"{'tick':>4}  {'mid':>8}  {'best_bid':>8}  {'best_ask':>8}  "
          f"{'fills':>5}  {'last_px':>8}")
    print("-" * 56)

    for tick in range(1, 21):
        # 1. Every agent acts (submits orders)
        for agent in all_agents:
            agent.act(tick, lob, last_price)

        # 2. Match crossing orders
        executions = lob.match_orders()

        # 3. Settle fills (cash / inventory accounting)
        px = settle_fills(executions, agents_by_id, momentum_traders)
        if px is not None:
            last_price = px
        total_fills += len(executions)

        mid = lob.get_mid_price()
        bb = lob.best_bid()
        ba = lob.best_ask()

        print(
            f"{tick:4d}  "
            f"{mid if mid is not None else 'N/A':>8}  "
            f"{bb if bb is not None else 'N/A':>8}  "
            f"{ba if ba is not None else 'N/A':>8}  "
            f"{len(executions):5d}  "
            f"{last_price if last_price is not None else 'N/A':>8}"
        )

    # --- Final summary -------------------------------------------------------
    print(f"\nTotal fills across 20 ticks: {total_fills}")
    print("\nAgent P&L snapshot (mark-to-market at last traded price):")
    mark = last_price if last_price is not None else 100.0
    print(f"  {'ID':<6} {'Cash':>10} {'Inv':>5} {'MTM P&L':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*5} {'-'*10}")
    for agent in all_agents:
        mtm = agent.cash + agent.inventory * mark - 10_000.0
        print(f"  {agent.agent_id:<6} {agent.cash:10.2f} {agent.inventory:5d} {mtm:10.2f}")
