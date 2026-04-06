"""Event Impact Analyzer — LLM-powered event analysis + agent-driven simulation.

When a news event occurs:
  1. The LLM identifies up to 15 impacted assets with direction/magnitude
  2. For each asset, a mini agent-based simulation runs:
     - A pool of heterogeneous agents (same personas as the main simulation)
       each call the LLM with the event context and the asset's market state
     - Agents submit orders into a per-asset Python OrderBook
     - The matching engine produces executions → prices emerge from interaction
     - Price trajectory is recorded over N ticks

This replaces the previous Monte Carlo SCM approach.  Prices now come from
agent interaction in a limit order book, not from mathematical equations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import statistics
import uuid
from typing import Any, Callable, Awaitable, Optional

from config.settings import settings
from matching_engine import PyOrderBook, Order

logger = logging.getLogger(__name__)

# ─── Prompt Template ────────────────────────────────────────────────────────

EVENT_ANALYSIS_SYSTEM_PROMPT = """\
You are a financial analyst AI. Given a news event, predict which assets \
will be impacted and in what direction.

## Instructions
1. Analyze the event for geopolitical, economic, and sector-specific implications
2. Identify assets that will likely be POSITIVELY or NEGATIVELY impacted
3. Assign magnitude: "high" (>5%), "medium" (2-5%), "low" (<2%) as an estimated real-world impact scale
4. Provide a brief causal reason for each prediction
5. Consider both direct (first-order) and indirect (second-order) effects
6. Do NOT simulate or forecast exact price movements — only classify the expected direction and relative magnitude
7. Your output will be used as input parameters for a separate market simulation engine

## Asset Universe
Consider these asset classes:
- Commodities: Crude Oil, Natural Gas, Gold, Silver, Copper, Wheat, Corn
- Indices: S&P 500, NASDAQ, Dow Jones, Russell 2000
- Sectors: Technology, Healthcare, Defense, Energy, Financials, Real Estate, \
Consumer Discretionary, Consumer Staples, Utilities, Industrials, Materials, Airlines, Tourism
- Currencies: USD, EUR, JPY, GBP, CNY
- Bonds: US Treasury 10Y, US Treasury 2Y, Corporate Bonds
- Crypto: Bitcoin, Ethereum

## Output Format (strict JSON)
{{
  "event": "<original event text>",
  "event_category": "<one of: military_conflict, trade_policy, monetary_policy, \
pandemic, natural_disaster, political_instability, technological_disruption, \
regulatory_change, energy_crisis, other>",
  "severity": <1-10>,
  "time_horizon": "<short_term | medium_term | long_term>",
  "impacts": [
    {{
      "asset": "<asset name>",
      "ticker": "<ticker symbol if applicable, else null>",
      "direction": "<up | down>",
      "magnitude": "<high | medium | low>",
      "confidence": <0.0 to 1.0>,
      "reason": "<one sentence causal explanation>"
    }}
  ]
}}

## Rules
- Return at least 3 and at most 5 impacted assets
- Sort by confidence descending
- Always consider safe-haven flows (gold, USD, treasuries) during crises
- Consider supply chain effects (e.g., conflict near shipping lanes → shipping costs up)
- Do NOT include assets with negligible or highly uncertain impact
- Return ONLY valid JSON, no additional text
"""

# Typical base prices for each asset
_ASSET_BASE_PRICES: dict[str, float] = {
    "Crude Oil": 75.0, "Natural Gas": 3.50, "Gold": 2350.0, "Silver": 28.0,
    "Copper": 4.20, "Wheat": 550.0, "Corn": 450.0,
    "S&P 500": 5200.0, "NASDAQ": 16500.0, "Dow Jones": 39000.0, "Russell 2000": 2050.0,
    "Technology": 200.0, "Healthcare": 150.0, "Defense": 130.0, "Energy": 95.0,
    "Financials": 42.0, "Real Estate": 38.0, "Consumer Discretionary": 180.0,
    "Consumer Staples": 78.0, "Utilities": 70.0, "Industrials": 120.0,
    "Materials": 85.0, "Airlines": 55.0, "Tourism": 45.0,
    "USD": 104.0, "EUR": 1.08, "JPY": 155.0, "GBP": 1.27, "CNY": 7.25,
    "US Treasury 10Y": 95.0, "US Treasury 2Y": 99.0, "Corporate Bonds": 90.0,
    "Bitcoin": 67000.0, "Ethereum": 3500.0,
}

# Magnitude → approximate spread factor (higher impact = wider spread = more volatility)
_MAGNITUDE_SPREAD: dict[str, float] = {
    "high": 0.005,    # 0.5% spread
    "medium": 0.003,  # 0.3% spread
    "low": 0.001,     # 0.1% spread
}

# Event category → default SCM variable (kept for to_scm_shock compatibility)
_CATEGORY_TO_SCM: dict[str, str] = {
    "military_conflict": "market_sentiment",
    "trade_policy": "market_sentiment",
    "monetary_policy": "interest_rate",
    "pandemic": "market_sentiment",
    "natural_disaster": "market_sentiment",
    "political_instability": "market_sentiment",
    "technological_disruption": "market_sentiment",
    "regulatory_change": "market_sentiment",
    "energy_crisis": "inflation",
    "other": "market_sentiment",
}

_MAGNITUDE_WEIGHT: dict[str, float] = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.2,
}

# ─── Agent personas — Volume-weighted (same as main simulation) ────────────
# Per-asset mini-sims use the same 5 persona archetypes weighted by daily
# trading volume so shock dynamics are consistent with the main loop.

_SIM_PERSONA_POOL = [
    {
        "name": "HFT Market Maker providing continuous liquidity with tight spreads",
        "volume_weight": 0.45,
        "strategy": "passive",      # quotes passively, captures spread
        "shock_behavior": "LIQUIDITY PULL — widen spreads, reduce size",
    },
    {
        "name": "Momentum Quantitative Trader executing statistical arbitrage and trend-following",
        "volume_weight": 0.25,
        "strategy": "aggressive",   # chases momentum aggressively
        "shock_behavior": "VOLATILITY CHASER — pile into directional move",
    },
    {
        "name": "Macro Event-Driven Fund repricing fundamentals based on macro analysis",
        "volume_weight": 0.15,
        "strategy": "aggressive",   # price-insensitive market-taker
        "shock_behavior": "FUNDAMENTAL REPRICER — dump/accumulate to target",
    },
    {
        "name": "Retail Sentiment Trader influenced by social media and crowd psychology",
        "volume_weight": 0.10,
        "strategy": "passive",      # delayed, mean-reverts
        "shock_behavior": "DELAYED MEAN-REVERTER — buy the dip / sell the rip",
    },
    {
        "name": "Passive Index Fund Manager executing only for rebalancing with minimal impact",
        "volume_weight": 0.05,
        "strategy": "twap",         # TWAP at close only
        "shock_behavior": "INERTIA — hold, do not trade intraday",
    },
]

_SIM_PERSONAS = [p["name"] for p in _SIM_PERSONA_POOL]

# Sophistication multipliers (same as main config)
_SOPHISTICATION = {
    "HFT Market Maker": 1.2,
    "Momentum Quantitative": 1.1,
    "Macro Event-Driven": 1.0,
    "Retail Sentiment": 2.5,
    "Passive Index Fund": 0.5,
}

# Execution strategy templates (kept for backward compat)
_STRATEGIES = ["aggressive", "passive", "twap", "passive"]

# ─── Agent-based per-asset simulation prompt ────────────────────────────────

_ASSET_SIM_PROMPT = """\
You are Agent {agent_id}, a {persona} operating in a simulated {asset} market.

BREAKING NEWS: {event_headline}

The LLM analysis predicts {asset} will go {direction} with {magnitude} magnitude.
Reason: {reason}

Current {asset} price: ${mid_price:.2f}
Spread: ${spread:.4f}
Tick: {tick} of {total_ticks}
Your inventory: {inventory} units
Your cash: ${cash:.2f}

Based on your persona and the news event, decide whether to BUY, SELL, or HOLD {asset}.
Consider:
- Your persona's typical response to this type of event
- The current price relative to pre-event price (${base_price:.2f})
- Risk management (don't overexpose)

Respond with ONLY valid JSON:
{{"action": "buy"|"sell"|"hold", "target_price": <float>, "confidence": <float 0.0-1.0>, "reasoning": "<1 sentence>"}}
"""


# ─── Analyzer (LLM call — unchanged) ──────────────────────────────────────

async def analyze_event(
    event_text: str,
    llm_call_fn: Callable[[str, str], Awaitable[str]],
    timeout: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Call the LLM to analyze a news event and return structured impact data."""
    if timeout is None:
        timeout = settings.llm_timeout

    user_message = f'"{event_text}"'

    try:
        raw = await asyncio.wait_for(
            llm_call_fn(EVENT_ANALYSIS_SYSTEM_PROMPT, user_message),
            timeout=timeout,
        )

        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = json.loads(text)

        if "impacts" not in data or not isinstance(data["impacts"], list):
            logger.warning("Event analysis missing 'impacts' list")
            return None

        data["severity"] = max(1, min(10, int(data.get("severity", 5))))

        validated: list[dict] = []
        for imp in data["impacts"]:
            direction = str(imp.get("direction", "")).lower()
            if direction not in ("up", "down"):
                continue
            validated.append({
                "asset": str(imp.get("asset", "Unknown")),
                "ticker": imp.get("ticker"),
                "direction": direction,
                "magnitude": str(imp.get("magnitude", "low")).lower(),
                "confidence": max(0.0, min(1.0, float(imp.get("confidence", 0.5)))),
                "reason": str(imp.get("reason", "")),
            })

        if not validated:
            logger.warning("Event analysis returned no valid impacts")
            return None

        validated.sort(key=lambda x: x["confidence"], reverse=True)
        data["impacts"] = validated
        return data

    except asyncio.TimeoutError:
        logger.warning("Event analysis LLM call timed out")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Event analysis malformed response: %s", exc)
    except Exception as exc:
        logger.warning("Event analysis error: %s", exc)

    return None


# ─── Agent-driven per-asset simulation ────────────────────────────────────

class SimAgent:
    """Lightweight agent for per-asset event simulation.

    Each agent has a persona, sophistication level, execution strategy,
    cash, and inventory. It calls the LLM each tick to decide buy/sell/hold,
    then submits orders to the asset's OrderBook.
    """

    def __init__(
        self,
        agent_id: str,
        persona: str,
        strategy: str,
        cash: float = 100_000.0,
    ) -> None:
        self.agent_id = agent_id
        self.persona = persona
        self.strategy = strategy
        self.cash = cash
        self.inventory: int = 0
        self._sophistication = 1.0
        for key, val in _SOPHISTICATION.items():
            if key in persona:
                self._sophistication = val
                break

    def generate_order(
        self,
        decision: dict,
        mid_price: float,
        spread: float,
        tick: int,
    ) -> Optional[dict]:
        """Convert LLM decision to an order, applying execution strategy."""
        action = decision.get("action", "hold")
        if action == "hold":
            return None

        confidence = decision.get("confidence", 0.5)
        target = decision.get("target_price", mid_price)

        # Apply execution strategy
        if self.strategy == "aggressive":
            slippage = 0.002 * confidence
            if action == "buy":
                price = target * (1 + slippage)
            else:
                price = target * (1 - slippage)
        elif self.strategy == "twap":
            price = target
        else:  # passive
            price = target

        # Position sizing: scale by confidence and sophistication
        qty = max(1, int(5 * confidence * self._sophistication))

        # Risk check: don't over-leverage
        if action == "buy" and self.cash < price * qty:
            qty = max(1, int(self.cash / price))
            if self.cash < price:
                return None
        if action == "sell" and self.inventory <= 0 and self.cash < price * qty:
            # Short selling: need margin
            qty = max(1, int(self.cash / (price * 2)))

        return {
            "agent_id": self.agent_id,
            "action": action,
            "price": round(price, 4),
            "quantity": qty,
            "tick": tick,
        }

    def on_fill(self, price: float, qty: int, side: str) -> None:
        """Update agent state after a fill."""
        if side == "buy":
            self.cash -= price * qty
            self.inventory += qty
        else:
            self.cash += price * qty
            self.inventory -= qty


def _create_sim_agents(n_agents: int = 16) -> list[SimAgent]:
    """Create a pool of heterogeneous agents weighted by daily trading volume."""
    agents = []
    idx = 0
    for p in _SIM_PERSONA_POOL:
        count = max(1, round(p["volume_weight"] * n_agents))
        for _ in range(count):
            if idx >= n_agents:
                break
            agents.append(SimAgent(
                agent_id=f"SIM-{idx:03d}",
                persona=p["name"],
                strategy=p["strategy"],
            ))
            idx += 1
    # Fill remaining slots with HFT (largest weight)
    while idx < n_agents:
        agents.append(SimAgent(
            agent_id=f"SIM-{idx:03d}",
            persona=_SIM_PERSONA_POOL[0]["name"],
            strategy=_SIM_PERSONA_POOL[0]["strategy"],
        ))
        idx += 1
    return agents


async def _agent_decide(
    agent: SimAgent,
    asset: str,
    event_headline: str,
    impact: dict,
    mid_price: float,
    spread: float,
    base_price: float,
    tick: int,
    total_ticks: int,
    llm_call_fn: Callable[[str, str], Awaitable[str]],
) -> Optional[dict]:
    """Have one agent make a trading decision for an asset via LLM."""
    prompt = _ASSET_SIM_PROMPT.format(
        agent_id=agent.agent_id,
        persona=agent.persona,
        asset=asset,
        event_headline=event_headline,
        direction=impact["direction"],
        magnitude=impact["magnitude"],
        reason=impact["reason"],
        mid_price=mid_price,
        spread=spread,
        tick=tick,
        total_ticks=total_ticks,
        inventory=agent.inventory,
        cash=agent.cash,
        base_price=base_price,
    )

    try:
        raw = await asyncio.wait_for(
            llm_call_fn(prompt, f"Tick {tick}. Decide now."),
            timeout=settings.llm_timeout,
        )
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(text)

        action = str(data.get("action", "hold")).lower()
        if action not in ("buy", "sell", "hold"):
            action = "hold"

        return {
            "action": action,
            "target_price": float(data.get("target_price", mid_price)),
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
        }
    except Exception:
        return None


async def simulate_single_asset(
    impact: dict[str, Any],
    event_headline: str,
    llm_call_fn: Callable[[str, str], Awaitable[str]],
    n_ticks: int = 100,
    n_agents: int = 16,
) -> dict[str, Any]:
    """Run an agent-based simulation for one impacted asset.

    Creates a pool of heterogeneous agents and a dedicated OrderBook.
    Each tick: agents call LLM → submit orders → book matches → price updates.
    Returns the price trajectory and trade statistics.
    """
    asset = impact["asset"]
    base_price = _ASSET_BASE_PRICES.get(asset, 100.0)
    spread_factor = _MAGNITUDE_SPREAD.get(impact["magnitude"], 0.002)
    initial_spread = base_price * spread_factor

    book = PyOrderBook(symbol=asset, initial_price=base_price)
    agents = _create_sim_agents(n_agents)

    # Seed the book with thin initial liquidity around base_price
    _seed_book(book, base_price, initial_spread, n_levels=3)

    price_trajectory: list[float] = [base_price]
    volume_trajectory: list[int] = [0]
    trade_count: list[int] = [0]
    ohlcv: list[dict] = []

    for tick in range(1, n_ticks + 1):
        current_mid = book.mid_price
        current_spread = max(book.spread, base_price * 0.0001)

        # All agents decide concurrently
        decision_tasks = [
            _agent_decide(
                agent=agent,
                asset=asset,
                event_headline=event_headline,
                impact=impact,
                mid_price=current_mid,
                spread=current_spread,
                base_price=base_price,
                tick=tick,
                total_ticks=n_ticks,
                llm_call_fn=llm_call_fn,
            )
            for agent in agents
        ]
        decisions = await asyncio.gather(*decision_tasks, return_exceptions=True)

        # Submit orders
        tick_orders = 0
        for agent, decision in zip(agents, decisions):
            if isinstance(decision, Exception) or decision is None:
                continue
            order_spec = agent.generate_order(
                decision, current_mid, current_spread, tick,
            )
            if order_spec is None:
                continue

            order = Order(
                order_id=f"ORD-{asset[:3]}-{uuid.uuid4().hex[:8]}",
                agent_id=agent.agent_id,
                side=order_spec["action"],
                price=order_spec["price"],
                quantity=order_spec["quantity"],
            )
            book.add_order(order)
            tick_orders += 1

        # Match
        fills = book.match_orders()

        # Route fills to agents
        tick_prices = []
        for fill in fills:
            tick_prices.append(fill.fill_price)
            # Find buyer and seller agents
            for agent in agents:
                if agent.agent_id == fill.buyer_id:
                    agent.on_fill(fill.fill_price, fill.fill_qty, "buy")
                elif agent.agent_id == fill.seller_id:
                    agent.on_fill(fill.fill_price, fill.fill_qty, "sell")

        # Record OHLCV
        if tick_prices:
            ohlcv.append({
                "tick": tick,
                "open": tick_prices[0],
                "high": max(tick_prices),
                "low": min(tick_prices),
                "close": tick_prices[-1],
                "volume": sum(f.fill_qty for f in fills),
                "trades": len(fills),
            })
        else:
            last_p = price_trajectory[-1]
            ohlcv.append({
                "tick": tick,
                "open": last_p, "high": last_p,
                "low": last_p, "close": last_p,
                "volume": 0, "trades": 0,
            })

        # Replenish thin liquidity (market makers continuously provide quotes)
        _seed_book(book, book.mid_price, current_spread, n_levels=2)

        price_trajectory.append(book.mid_price)
        volume_trajectory.append(sum(f.fill_qty for f in fills))
        trade_count.append(len(fills))

    final_price = price_trajectory[-1]
    final_effect_pct = ((final_price - base_price) / base_price) * 100
    peak_price = max(price_trajectory) if impact["direction"] == "up" else min(price_trajectory)
    peak_effect_pct = ((peak_price - base_price) / base_price) * 100
    peak_tick = price_trajectory.index(peak_price)

    return {
        "asset": asset,
        "ticker": impact["ticker"],
        "direction": impact["direction"],
        "magnitude": impact["magnitude"],
        "confidence": impact["confidence"],
        "reason": impact["reason"],
        # Agent-driven simulation results
        "simulation_method": "agent_lob",
        "n_agents": n_agents,
        "n_ticks": n_ticks,
        "base_price": base_price,
        "price_trajectory": [round(p, 4) for p in price_trajectory],
        "ohlcv": ohlcv,
        "total_volume": sum(volume_trajectory),
        "total_trades": sum(trade_count),
        "peak_effect_pct": round(peak_effect_pct, 2),
        "peak_tick": peak_tick,
        "final_price": round(final_price, 4),
        "final_effect_pct": round(final_effect_pct, 2),
    }


def _seed_book(
    book: PyOrderBook,
    mid_price: float,
    spread: float,
    n_levels: int = 5,
) -> None:
    """Seed the order book with liquidity around the mid price.

    Simulates market makers providing resting orders at multiple levels.
    """
    half_spread = max(spread / 2, mid_price * 0.0001)
    tick_size = max(mid_price * 0.0001, 0.01)

    for i in range(n_levels):
        bid_price = round(mid_price - half_spread - i * tick_size, 4)
        ask_price = round(mid_price + half_spread + i * tick_size, 4)
        qty = random.randint(1, 3)  # thin liquidity so agent orders move the price

        book.add_order(Order(
            order_id=f"MM-B-{uuid.uuid4().hex[:6]}",
            agent_id=f"MM-{i}",
            side="buy",
            price=bid_price,
            quantity=qty,
        ))
        book.add_order(Order(
            order_id=f"MM-A-{uuid.uuid4().hex[:6]}",
            agent_id=f"MM-{i}",
            side="sell",
            price=ask_price,
            quantity=qty,
        ))

    # Match any crossing orders from seeding
    book.match_orders()


async def simulate_all_impacts(
    analysis: dict[str, Any],
    event_headline: str,
    llm_call_fn: Callable[[str, str], Awaitable[str]],
    n_ticks: int = 100,
    n_agents: int = 16,
) -> list[dict[str, Any]]:
    """Run agent-based simulation for every impacted asset.

    Each asset gets its own OrderBook and agent pool. Simulations run
    sequentially to avoid overwhelming the LLM API. Each agent calls
    the LLM to decide based on their persona and the event context.
    """
    results: list[dict[str, Any]] = []

    for impact in analysis.get("impacts", []):
        sim = await simulate_single_asset(
            impact=impact,
            event_headline=str(analysis.get("event", "")),
            llm_call_fn=llm_call_fn,
            n_ticks=n_ticks,
            n_agents=n_agents,
        )
        results.append(sim)

        sign = "+" if sim["final_effect_pct"] >= 0 else ""
        logger.info(
            "  [LOB] %s: %s%s%.2f%% (%d trades, %d volume) base=$%.2f → final=$%.2f",
            sim["asset"], sim["direction"], sign,
            sim["final_effect_pct"], sim["total_trades"],
            sim["total_volume"], sim["base_price"], sim["final_price"],
        )

    return results


def to_scm_shock(analysis: dict[str, Any]) -> dict[str, Any]:
    """Convert an event analysis result into an SCM-compatible shock dict.

    Returns a dict with keys: category, severity (0-1), scm_var,
    intervention_val, and the full impacts list for downstream use.
    """
    category = str(analysis.get("event_category", "other")).lower()
    raw_severity = analysis.get("severity", 5)  # 1-10
    severity_01 = raw_severity / 10.0

    scm_var = _CATEGORY_TO_SCM.get(category, "market_sentiment")

    # Compute aggregate directional signal from impacts
    net_signal = 0.0
    total_weight = 0.0
    for imp in analysis.get("impacts", []):
        sign = 1.0 if imp["direction"] == "up" else -1.0
        weight = _MAGNITUDE_WEIGHT.get(imp["magnitude"], 0.2) * imp["confidence"]
        net_signal += sign * weight
        total_weight += weight

    if total_weight > 0:
        net_signal /= total_weight

    if scm_var == "interest_rate":
        intervention_val = 0.04 + severity_01 * 0.08
    elif scm_var == "inflation":
        intervention_val = 0.02 + severity_01 * 0.08
    elif scm_var == "gdp_growth":
        intervention_val = -0.02 + (1 - severity_01) * 0.05
    elif scm_var == "liquidity":
        intervention_val = -severity_01 * 0.5
    else:
        intervention_val = net_signal * severity_01

    return {
        "category": category,
        "severity": severity_01,
        "scm_var": scm_var,
        "intervention_val": intervention_val,
        "event": analysis.get("event", ""),
        "time_horizon": analysis.get("time_horizon", "short_term"),
        "impacts": analysis.get("impacts", []),
    }
