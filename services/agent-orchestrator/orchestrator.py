"""Orchestrator — main simulation loop connecting Ray actors to Kafka.

Flow per tick:
    1. Consume MarketData + Shocks from Kafka
    2. Fan-out: call actor.on_tick() concurrently for all 1,000 agents
    3. Collect LLM directives → route through TraderSubAgent execution layer
    4. Compute MASS consensus/disagreement signal
    5. Publish orders to Kafka → C++ LOB engine
    6. Consume executions from Kafka → route fills back to actors
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from typing import Any, Optional

import ray

from actors.base_actor import (
    TradingAgentActor, compute_dynamic_weights, get_persona_config,
    PERSONA_POOL,
)
from config.settings import settings
from svar_validation import (
    TimeSeriesData, run_svar_validation, compute_returns, compute_realized_vol,
)
from ddql_agent import DDQLAgent, Transition
from causal_engine import StructuralCausalModel
from event_impact_analyzer import analyze_event, to_scm_shock, simulate_all_impacts
from proto_codec import (
    encode_order, decode_execution, decode_market_snapshot,
    encode_tick_summary,
)

logger = logging.getLogger(__name__)


# ─── Kafka Helpers (aiokafka) ────────────────────────────────────────────────
# In production: pip install aiokafka
# These wrappers isolate Kafka from the orchestration logic.

class KafkaOrderPublisher:
    """Publishes agent orders to the orders_submit Kafka topic."""

    def __init__(self, brokers: str, topic: str) -> None:
        self._brokers = brokers
        self._topic = topic
        self._producer = None

    async def connect(self) -> None:
        from aiokafka import AIOKafkaProducer
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._brokers,
            value_serializer=lambda v: encode_order(v) if isinstance(v, dict) else v,
            key_serializer=lambda k: k.encode() if k else None,
        )
        await self._producer.start()
        logger.info("Kafka producer connected to %s", self._brokers)

    async def publish_order(self, order: dict) -> None:
        await self._producer.send_and_wait(
            self._topic,
            value=order,
            key=order.get("agent_id"),
        )

    async def close(self) -> None:
        if self._producer:
            await self._producer.stop()


class KafkaMarketConsumer:
    """Consumes market data and execution reports from Kafka."""

    def __init__(self, brokers: str, topics: list[str]) -> None:
        self._brokers = brokers
        self._topics = topics
        self._consumer = None

    async def connect(self) -> None:
        from aiokafka import AIOKafkaConsumer
        self._consumer = AIOKafkaConsumer(
            *self._topics,
            bootstrap_servers=self._brokers,
            group_id="agent-orchestrator",
            # Raw bytes — we decode per-topic using proto_codec
            auto_offset_reset="latest",
        )
        await self._consumer.start()
        logger.info("Kafka consumer subscribed to %s", self._topics)

    async def poll(self, timeout_ms: int = 100) -> list[dict]:
        """Poll for messages. Returns list of (topic, decoded_value) tuples."""
        records = await self._consumer.getmany(
            timeout_ms=timeout_ms, max_records=500
        )
        messages = []
        for tp, msgs in records.items():
            for msg in msgs:
                topic = tp.topic
                raw = msg.value  # bytes
                if topic == "executions":
                    value = decode_execution(raw)
                elif topic == "market_data":
                    value = decode_market_snapshot(raw)
                elif topic == "exogenous_shocks":
                    # Shocks are injected by the API gateway as JSON
                    value = json.loads(raw.decode())
                else:
                    # Fallback: try JSON for unknown topics
                    try:
                        value = json.loads(raw.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        value = {"raw": raw}
                messages.append({"topic": topic, "value": value})
        return messages

    async def close(self) -> None:
        if self._consumer:
            await self._consumer.stop()


# ─── Hierarchical Manager-Trader Framework ──────────────────────────────────

EXECUTION_STRATEGIES = ["aggressive", "passive", "twap", "ddql"]


@ray.remote(num_cpus=0.005)
class TraderSubAgent:
    """Lightweight programmatic execution agent.

    Receives strategy directive from LLM manager actor and converts it
    to an order with execution logic. No LLM calls — pure rule-based.
    """

    def __init__(self, agent_id: str, strategy: str) -> None:
        self.agent_id = agent_id
        self.strategy = strategy  # "aggressive", "passive", "twap"

    async def execute(
        self, directive: dict, mid_price: float, tick: int
    ) -> Optional[dict]:
        if directive["action"] == "hold":
            return None

        urgency = directive.get("urgency", directive.get("confidence", 0.5))
        qty = directive.get("quantity", 10)
        target = directive.get("target_price", mid_price)

        if self.strategy == "aggressive":
            slippage = 0.002 * urgency
            price = (target * (1 + slippage)
                     if directive["action"] == "buy"
                     else target * (1 - slippage))
        elif self.strategy == "twap":
            qty = max(1, qty // 3)
            price = target
        else:  # passive
            price = target

        return {
            "agent_id": self.agent_id,
            "action": directive["action"],
            "price": round(price, 2),
            "quantity": qty,
            "tick": tick,
        }


# ─── MASS Framework: Consensus / Disagreement Signal ────────────────────────

def compute_mass_signal(directives: list[dict], alpha: float = 0.6) -> dict:
    """MASS framework: Signal(s,j) = alpha * m_s(j) - (1 - alpha) * sigma_s(j)

    Converts each agent action to directional score: buy=+1, sell=-1, hold=0.
    Weighted by confidence.
    """
    scores: list[float] = []
    for d in directives:
        action = d.get("action", "hold")
        conf = d.get("confidence", 0.5)
        score = conf if action == "buy" else (-conf if action == "sell" else 0.0)
        scores.append(score)

    if not scores:
        return {"consensus": 0.0, "disagreement": 0.0, "mass_signal": 0.0}

    m_s = sum(scores) / len(scores)
    sigma_s = statistics.stdev(scores) if len(scores) > 1 else 0.0
    mass_signal = alpha * m_s - (1 - alpha) * sigma_s

    return {"consensus": m_s, "disagreement": sigma_s, "mass_signal": mass_signal}


# ─── Orchestrator ────────────────────────────────────────────────────────────

class SimulationOrchestrator:
    """Main orchestrator that drives the simulation tick loop."""

    def __init__(self) -> None:
        self.actors: list[ray.ObjectRef] = []
        self.actor_map: dict[str, ray.ObjectRef] = {}
        self.trader_agents: dict[str, ray.ObjectRef] = {}  # agent_id → TraderSubAgent
        self._publisher: Optional[KafkaOrderPublisher] = None
        self._consumer: Optional[KafkaMarketConsumer] = None
        self._current_mid: float = 100.0
        self._current_spread: float = 0.10

        # Price / volume history for SVAR validation
        self._price_history: list[float] = []
        self._volume_history: list[float] = []

        # Redis for storing event analysis results
        self._analysis_redis = None

        # DDQL execution agents (agent_id → DDQLAgent)
        self._ddql_agents: dict[str, DDQLAgent] = {}
        # Stores the previous tick's encoded state for each DDQL agent
        # so we can build complete (s, a, r, s') transitions after fills arrive.
        self._ddql_prev_states: dict[str, list[float]] = {}
        self._ddql_prev_actions: dict[str, int] = {}
        self._ddql_prev_prices: dict[str, float] = {}

        self._scm = StructuralCausalModel()
        # Tracks active SCM interventions: {variable: ticks_remaining}
        self._active_interventions: dict[str, int] = {}

        # ── Dynamic weighting state ──────────────────────────────────────
        self._base_volatility: float = 0.15       # calibrated from first N ticks
        self._current_volatility: float = 0.15
        self._shock_active: bool = False
        self._recent_returns: list[float] = []    # rolling window for realized vol
        # Maps agent_id → persona name (for weight lookups)
        self._agent_personas: dict[str, str] = {}

    async def initialize(self) -> None:
        """Spin up Ray, create all actors, connect Kafka."""
        import redis.asyncio as aioredis
        self._analysis_redis = aioredis.from_url(
            settings.redis_url, decode_responses=True
        )

        if not ray.is_initialized():
            ray.init(num_cpus=settings.ray_num_cpus, ignore_reinit_error=True)

        logger.info("Creating %d agent actors...", settings.num_agents)

        # ── LLM call via DashScope — multi-key support ───────────────────────
        # If DASHSCOPE_API_KEYS has multiple keys (comma-separated), each agent
        # gets its own key round-robin. This distributes rate limits across keys.
        # Example: 5 keys × 10 agents = 2 agents per key → 5x less 429 errors.
        import requests as _requests

        api_keys = settings.dashscope_api_keys
        num_keys = len(api_keys)
        logger.info("API keys available: %d", num_keys)

        def _make_llm_fn(api_key: str):
            """Create an LLM call function bound to a specific API key."""
            def _sync_llm_call(system_prompt: str, user_msg: str) -> str:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        resp = _requests.post(
                            f"{settings.dashscope_base_url}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": settings.primary_model,
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_msg},
                                ],
                                "temperature": 0.7,
                            },
                            timeout=settings.llm_timeout,
                        )
                        if resp.status_code == 429:
                            wait = (2 ** attempt) * 2.0
                            logger.warning("LLM 429 (key ..%s), retry %d/%d in %.0fs",
                                           api_key[-4:], attempt + 1, max_retries, wait)
                            time.sleep(wait)
                            continue
                        resp.raise_for_status()
                        return resp.json()["choices"][0]["message"]["content"]
                    except _requests.exceptions.Timeout:
                        logger.warning("LLM timeout, retry %d/%d", attempt + 1, max_retries)
                        continue
                    except _requests.exceptions.HTTPError:
                        if resp.status_code == 429:
                            time.sleep((2 ** attempt) * 2.0)
                            continue
                        raise
                raise RuntimeError("LLM call failed after all retries")

            async def _async_llm_call(system_prompt: str, user_msg: str) -> str:
                return await asyncio.get_event_loop().run_in_executor(
                    None, _sync_llm_call, system_prompt, user_msg,
                )
            return _async_llm_call

        def _make_slm_fn(api_key: str):
            """Create an SLM call function bound to a specific API key."""
            def _sync_slm_call(prompt: str) -> str:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        resp = _requests.post(
                            f"{settings.dashscope_base_url}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": settings.slm_model,
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.3,
                            },
                            timeout=settings.llm_timeout,
                        )
                        if resp.status_code == 429:
                            wait = (2 ** attempt) * 2.0
                            logger.warning("SLM 429 (key ..%s), retry %d/%d in %.0fs",
                                           api_key[-4:], attempt + 1, max_retries, wait)
                            time.sleep(wait)
                            continue
                        resp.raise_for_status()
                        return resp.json()["choices"][0]["message"]["content"]
                    except _requests.exceptions.Timeout:
                        continue
                    except _requests.exceptions.HTTPError:
                        if resp.status_code == 429:
                            time.sleep((2 ** attempt) * 2.0)
                            continue
                        raise
                raise RuntimeError("SLM call failed after all retries")

            async def _async_slm_call(prompt: str) -> str:
                return await asyncio.get_event_loop().run_in_executor(
                    None, _sync_slm_call, prompt,
                )
            return _async_slm_call

        # Default LLM/SLM using first key (for event analysis)
        default_llm = _make_llm_fn(api_keys[0]) if api_keys else _make_llm_fn("")
        default_slm = _make_slm_fn(api_keys[0]) if api_keys else _make_slm_fn("")
        self._llm_call = default_llm  # retained for event impact analysis

        async def embed_fn(text: str) -> list[float]:
            """Stub — replace with sentence-transformers or embedding API."""
            return [0.0] * 384

        # Create actors with volume-weighted persona assignment + per-key LLM
        from actors.base_actor import assign_persona
        for i in range(settings.num_agents):
            agent_id = f"AGT-{i:04d}"
            persona_name = assign_persona(i, settings.num_agents)
            self._agent_personas[agent_id] = persona_name

            # Round-robin key assignment: agent 0→key0, agent 1→key1, ...
            key_index = i % num_keys if num_keys > 0 else 0
            agent_key = api_keys[key_index] if api_keys else ""
            agent_llm = _make_llm_fn(agent_key)
            agent_slm = _make_slm_fn(agent_key)

            actor = TradingAgentActor.remote(
                agent_id=agent_id,
                redis_url=settings.redis_url,
                qdrant_host=settings.qdrant_host,
                qdrant_port=settings.qdrant_port,
                llm_call_fn_ref=agent_llm,
                slm_call_fn_ref=agent_slm,
                embed_fn_ref=embed_fn,
                ticks_per_summary=settings.ticks_per_summary,
                total_agents=settings.num_agents,
            )
            self.actors.append(actor)
            self.actor_map[agent_id] = actor

            # Create paired TraderSubAgent (round-robin strategy)
            strategy = EXECUTION_STRATEGIES[i % len(EXECUTION_STRATEGIES)]
            if strategy == "ddql":
                self._ddql_agents[agent_id] = DDQLAgent(agent_id)
                # Still create a passive TraderSubAgent as fallback
                trader = TraderSubAgent.remote(agent_id, "passive")
            else:
                trader = TraderSubAgent.remote(agent_id, strategy)
            self.trader_agents[agent_id] = trader

        # Log API key distribution
        if num_keys > 1:
            agents_per_key = settings.num_agents // num_keys
            remainder = settings.num_agents % num_keys
            logger.info("API key distribution: %d keys, ~%d agents/key", num_keys, agents_per_key)
            for ki in range(num_keys):
                assigned = [aid for j, aid in enumerate(
                    [f"AGT-{x:04d}" for x in range(settings.num_agents)]
                ) if j % num_keys == ki]
                logger.info("  Key ..%s: %s", api_keys[ki][-4:], ", ".join(assigned))

        # Log volume-weighted agent allocation
        from collections import Counter
        persona_counts = Counter(self._agent_personas.values())
        logger.info("Agent allocation (by daily trading volume):")
        for p in PERSONA_POOL:
            count = persona_counts.get(p["name"], 0)
            pct = count / settings.num_agents * 100 if settings.num_agents > 0 else 0
            logger.info(
                "  %s: %d agents (%.0f%%) — %s",
                p["shock_behavior"], count, pct, p["name"][:50],
            )

        # Initialize all actors concurrently
        init_refs = [actor.initialize.remote() for actor in self.actors]
        await asyncio.gather(*[ref_to_awaitable(r) for r in init_refs])
        logger.info("All %d actors initialized", len(self.actors))

        # Kafka connections
        self._publisher = KafkaOrderPublisher(
            settings.kafka_brokers, settings.orders_topic
        )
        self._consumer = KafkaMarketConsumer(
            settings.kafka_brokers,
            [settings.executions_topic, settings.market_data_topic,
             settings.shocks_topic],
        )
        await self._publisher.connect()
        await self._consumer.connect()

    async def run_tick(
        self,
        tick: int,
        news_headline: Optional[str] = None,
        news_severity: float = 0.0,
    ) -> dict:
        """Execute one simulation tick across all agents.

        Returns tick summary dict.
        """
        t0 = time.monotonic()

        # Tick down active SCM interventions and undo any that have expired
        expired = [var for var, remaining in self._active_interventions.items()
                   if remaining <= 1]
        for var in expired:
            self._scm.undo(var)
            del self._active_interventions[var]
            logger.info("SCM intervention expired: undo(%s)", var)
        for var in self._active_interventions:
            self._active_interventions[var] -= 1

        # ── Step the macro SCM to update the causal environment ──────────
        scm_state = self._scm.step()
        scm_mid = scm_state.get("asset_price", self._current_mid)
        scm_vol = scm_state.get("volatility", 0.15)
        # Always incorporate SCM price — LOB engine updates override when
        # there are real executions, but SCM drives price discovery otherwise.
        self._current_mid = scm_mid

        # ── Dynamic Weight Engine: compute volatility + adjust participation ─
        if len(self._price_history) >= 2:
            ret = (self._current_mid - self._price_history[-1]) / max(self._price_history[-1], 0.01)
            self._recent_returns.append(ret)
            # Rolling 20-tick realized volatility
            if len(self._recent_returns) > 20:
                self._recent_returns = self._recent_returns[-20:]
            if len(self._recent_returns) >= 5:
                self._current_volatility = (
                    statistics.stdev(self._recent_returns) * (252 ** 0.5)
                )
            # Calibrate base volatility from first 10 ticks
            if tick == 10 and self._current_volatility > 0:
                self._base_volatility = self._current_volatility
                logger.info(
                    "Base volatility calibrated: %.4f", self._base_volatility
                )

        # Check if any SCM intervention is active → shock mode
        self._shock_active = len(self._active_interventions) > 0

        # Compute dynamic participation weights
        dyn_weights = compute_dynamic_weights(
            self._base_volatility, self._current_volatility, self._shock_active
        )
        vol_ratio = self._current_volatility / max(self._base_volatility, 0.01)
        shock_intensity = max(0.0, min(1.0, (vol_ratio - 1.5) / 1.5))
        if self._shock_active:
            shock_intensity = max(shock_intensity, 0.5)

        # Push updated weights to all actors
        for agent_id, actor in self.actor_map.items():
            persona = self._agent_personas.get(agent_id, "")
            weight = dyn_weights.get(persona, 0.1)
            actor.update_participation.remote(weight, shock_intensity)

        if shock_intensity > 0.3 and tick % 5 == 0:
            logger.info(
                "DYNAMIC WEIGHTS tick=%d shock=%.0f%% vol=%.4f | %s",
                tick, shock_intensity * 100, self._current_volatility,
                " | ".join(
                    f"{p['shock_behavior']}={dyn_weights.get(p['name'], 0):.0%}"
                    for p in PERSONA_POOL
                ),
            )

        # ── Fan-out: agents decide in batches to avoid API rate limits ─────
        BATCH_SIZE = 2  # max concurrent LLM calls per batch
        BATCH_DELAY = 2.0  # seconds between batches

        results = []
        for batch_start in range(0, len(self.actors), BATCH_SIZE):
            batch = self.actors[batch_start:batch_start + BATCH_SIZE]
            order_refs = [
                actor.on_tick.remote(
                    tick=tick,
                    mid_price=self._current_mid,
                    spread=self._current_spread,
                    news_headline=news_headline,
                    news_severity=news_severity,
                )
                for actor in batch
            ]
            batch_results = await asyncio.gather(
                *[ref_to_awaitable(r) for r in order_refs],
                return_exceptions=True,
            )
            results.extend(batch_results)
            if batch_start + BATCH_SIZE < len(self.actors):
                await asyncio.sleep(BATCH_DELAY)

        # ── Collect LLM directives and route through TraderSubAgent ─────────
        directives: list[dict] = []
        exec_refs = []
        for result in results:
            if isinstance(result, Exception):
                logger.debug("Actor error: %s", result)
                continue
            if result is not None:
                directives.append(result)
                agent_id = result.get("agent_id", "")
                trader = self.trader_agents.get(agent_id)
                if trader is not None:
                    ref = trader.execute.remote(
                        result, self._current_mid, tick
                    )
                    exec_refs.append(ref)

        # Await all TraderSubAgent execution results
        exec_results = await asyncio.gather(
            *[ref_to_awaitable(r) for r in exec_refs],
            return_exceptions=True,
        )

        # ── DDQL execution for agents using learned policy ────────────
        ddql_orders: list[dict] = []
        for agent_id, ddql in self._ddql_agents.items():
            # Read actual inventory from actor state (best-effort; default 0)
            actor = self.actor_map.get(agent_id)
            inventory = 0
            if actor is not None:
                try:
                    state_dict = ray.get(actor.get_state.remote(), timeout=0.5)
                    inventory = int(state_dict.get("inventory", 0))
                    cash = float(state_dict.get("cash", 1_000_000.0))
                except Exception:
                    cash = 1_000_000.0
            else:
                cash = 1_000_000.0
            current_state = ddql.encode_state(
                mid_price=self._current_mid,
                spread=self._current_spread,
                inventory=inventory,
                price_history=self._price_history,
                cash=cash,
            )
            # Build and store a transition from the PREVIOUS tick if one exists
            prev_state = self._ddql_prev_states.get(agent_id)
            prev_action = self._ddql_prev_actions.get(agent_id)
            prev_price = self._ddql_prev_prices.get(agent_id)
            if prev_state is not None and prev_action is not None and prev_price is not None:
                # Simple reward: mark-to-market PnL change since last action
                pnl_delta = (self._current_mid - prev_price) * inventory
                spread_captured = self._current_spread * 0.5 if prev_action in (2, 4) else 0.0
                reward = ddql.compute_reward(
                    pnl_delta=pnl_delta,
                    spread_captured=spread_captured,
                    inventory=inventory,
                )
                from ddql_agent import Transition
                ddql.store(Transition(
                    state=prev_state,
                    action=prev_action,
                    reward=reward,
                    next_state=current_state,
                    done=False,
                ))
            # Select action and cache state for next tick
            selected_action = ddql.select_action(current_state)
            self._ddql_prev_states[agent_id] = current_state
            self._ddql_prev_actions[agent_id] = selected_action
            self._ddql_prev_prices[agent_id] = self._current_mid
            order = ddql.act(current_state, self._current_mid, self._current_spread)
            if order is not None:
                order["agent_id"] = agent_id
                order["tick"] = tick
                ddql_orders.append(order)

        # Compute MASS consensus/disagreement signal from ALL agent decisions
        # (including rule-based agents that returned orders)
        mass = compute_mass_signal(directives)

        # Compute spread from agent orders (best bid vs best ask)
        buy_prices = [d["price"] for d in directives
                      if d.get("side") == "buy" or d.get("action") == "buy"]
        sell_prices = [d["price"] for d in directives
                       if d.get("side") == "sell" or d.get("action") == "sell"]
        if buy_prices and sell_prices:
            best_bid = max(buy_prices)
            best_ask = min(sell_prices)
            agent_spread = max(0.0, best_ask - best_bid)
            self._current_spread = agent_spread if agent_spread > 0 else self._current_spread

        orders_submitted = 0
        for result in exec_results:
            if isinstance(result, Exception):
                logger.debug("Trader exec error: %s", result)
                continue
            if result is not None:
                await self._publisher.publish_order(result)
                orders_submitted += 1

        # ── Publish DDQL orders ──────────────────────────────────────────────
        for order in ddql_orders:
            await self._publisher.publish_order(order)
            orders_submitted += 1

        # ── Train DDQL agents (every tick, from replay buffer) ───────────
        for ddql in self._ddql_agents.values():
            ddql.train_step()

        # ── Consume executions from LOB engine ───────────────────────────────
        messages = await self._consumer.poll(timeout_ms=200)

        executions = []
        for msg in messages:
            if msg["topic"] == settings.executions_topic:
                executions.append(msg["value"])
                # Route fill to buyer and seller actors
                buyer_id = msg["value"].get("buyer_id")
                seller_id = msg["value"].get("seller_id")
                if buyer_id in self.actor_map:
                    self.actor_map[buyer_id].on_fill.remote(
                        msg["value"]["price"], msg["value"]["quantity"],
                        "buy", tick
                    )
                if seller_id in self.actor_map:
                    self.actor_map[seller_id].on_fill.remote(
                        msg["value"]["price"], msg["value"]["quantity"],
                        "sell", tick
                    )
            elif msg["topic"] == settings.market_data_topic:
                # Update local market state from LOB snapshot
                self._current_mid = msg["value"].get("mid_price", self._current_mid)
                self._current_spread = msg["value"].get("spread", self._current_spread)
            elif msg["topic"] == settings.shocks_topic:
                shock = msg["value"]
                severity = float(shock.get("severity", 0.5))
                category = shock.get("category", "")
                # Map shock category to SCM variable and intervention value
                if "rate" in category or "monetary" in category or "interest" in category:
                    scm_var = "interest_rate"
                    # severity 0-1 maps to rate range 0.04 - 0.12
                    intervention_val = 0.04 + severity * 0.08
                elif "inflation" in category:
                    scm_var = "inflation"
                    intervention_val = 0.02 + severity * 0.08
                elif "gdp" in category or "recession" in category:
                    scm_var = "gdp_growth"
                    intervention_val = -0.02 + (1 - severity) * 0.05
                elif "sentiment" in category or "crash" in category or "war" in category:
                    scm_var = "market_sentiment"
                    intervention_val = -severity
                elif "liquidity" in category:
                    scm_var = "liquidity"
                    intervention_val = -severity * 0.5
                else:
                    scm_var = "market_sentiment"
                    intervention_val = (0.5 - severity)  # positive = bullish, negative = bearish
                # Duration: shocks persist for ceil(severity * 20) ticks
                import math as _math
                duration = max(1, _math.ceil(severity * 20))
                self._scm.do(scm_var, intervention_val)
                self._active_interventions[scm_var] = duration
                logger.info(
                    "SCM intervention: do(%s=%.4f) for %d ticks (severity=%.2f)",
                    scm_var, intervention_val, duration, severity,
                )

        elapsed = time.monotonic() - t0

        # Accumulate history for SVAR validation
        self._price_history.append(self._current_mid)
        self._volume_history.append(float(orders_submitted))

        # Compute OHLCV from executions + agent order prices for richer data
        exec_prices = [e.get("price", e.get("fill_price", 0.0)) for e in executions]
        # Also include agent order prices for OHLC when no executions
        order_prices = [d.get("price", 0) for d in directives if d.get("price", 0) > 0]
        all_prices = exec_prices if exec_prices else order_prices
        open_price = all_prices[0] if all_prices else self._current_mid
        close_price = all_prices[-1] if all_prices else self._current_mid
        high_price = max(all_prices) if all_prices else self._current_mid
        low_price = min(all_prices) if all_prices else self._current_mid
        # Volume: from executions + count orders submitted as a proxy
        exec_volume = sum(e.get("quantity", e.get("fill_qty", 0)) for e in executions)
        order_volume = sum(d.get("quantity", 1) for d in directives)
        volume = exec_volume if exec_volume > 0 else order_volume

        tick_summary = {
            "tick": tick,
            "open_price": open_price,
            "close_price": close_price,
            "high_price": high_price,
            "low_price": low_price,
            "volume": volume,
            "num_trades": len(executions),
            "book_snapshot": None,  # populated from market_data if available
            "consensus": mass["consensus"],
            "disagreement": mass["disagreement"],
            "mass_signal": mass["mass_signal"],
            # Extra fields for internal use (not part of protobuf TickSummary)
            "orders_submitted": orders_submitted,
            "mid_price": self._current_mid,
            "spread": self._current_spread,
            "elapsed_ms": elapsed * 1000,
            "scm_asset_price": scm_mid,
            "scm_volatility": scm_vol,
        }

        # Publish tick summary as protobuf for WebSocket broadcast
        try:
            proto_bytes = encode_tick_summary(tick_summary)
            await self._publisher._producer.send_and_wait(
                "tick_summary",
                value=proto_bytes,
                key=str(tick),
            )
            logger.info("Published tick_summary for tick %d (%d bytes)", tick, len(proto_bytes))
        except Exception as exc:
            logger.error("Failed to publish tick_summary: %s", exc, exc_info=True)

        return tick_summary

    def validate_against_empirical(
        self, empirical: TimeSeriesData, threshold: float = 0.05
    ) -> dict:
        """Run Guerini-Moneta SVAR validation against empirical data.

        Constructs a TimeSeriesData from accumulated simulation history
        and compares its Granger-causal graph to the empirical one.
        """
        sim_returns = compute_returns(self._price_history)
        sim_vols = compute_realized_vol(sim_returns)

        simulated = TimeSeriesData(
            prices=self._price_history,
            volumes=self._volume_history,
            volatilities=sim_vols,
        )
        return run_svar_validation(simulated, empirical, threshold)

    async def analyze_and_inject_shock(self, event_text: str) -> Optional[dict]:
        """Use the LLM to analyze a news event, then simulate each impacted
        asset through a dedicated agent-driven OrderBook.

        Each asset gets its own pool of heterogeneous agents (same personas as
        the main simulation) that call the LLM, submit orders, and trade in
        a limit order book.  Price emerges from agent interaction, not equations.

        Returns the analysis dict enriched with per-asset simulation results,
        or None on failure.
        """
        import math as _math

        analysis = await analyze_event(event_text, self._llm_call)
        if analysis is None:
            logger.warning("Event analysis failed for: %s", event_text[:80])
            return None

        # Per-asset simulation: agents trade in per-asset LOBs
        sim_results = await simulate_all_impacts(
            analysis,
            event_headline=event_text,
            llm_call_fn=self._llm_call,
            n_ticks=100,
            n_agents=16,
        )
        analysis["simulation_results"] = sim_results

        # Store in Redis for the frontend to fetch
        if self._analysis_redis:
            try:
                await self._analysis_redis.set(
                    "event_analysis:latest",
                    json.dumps(analysis, default=str),
                )
                await self._analysis_redis.lpush(
                    "event_analysis:history",
                    json.dumps(analysis, default=str),
                )
                await self._analysis_redis.ltrim("event_analysis:history", 0, 19)
            except Exception as exc:
                logger.debug("Failed to store analysis in Redis: %s", exc)

        for sim in sim_results:
            sign = "+" if sim["final_effect_pct"] >= 0 else ""
            logger.info(
                "  [LOB] %s: %s → %s%.2f%% (%d trades) base %.2f → final %.2f",
                sim["asset"], sim["direction"],
                sign, sim["final_effect_pct"], sim["total_trades"],
                sim["base_price"], sim["final_price"],
            )

        # Also inject aggregate shock into the main SCM for the tick loop
        shock = to_scm_shock(analysis)
        scm_var = shock["scm_var"]
        severity = shock["severity"]
        intervention_val = shock["intervention_val"]
        duration = max(1, _math.ceil(severity * 20))

        self._scm.do(scm_var, intervention_val)
        self._active_interventions[scm_var] = duration

        logger.info(
            "Event analysis → SCM do(%s=%.4f) for %d ticks | category=%s severity=%.2f | %d assets simulated",
            scm_var, intervention_val, duration,
            shock["category"], severity, len(sim_results),
        )
        return analysis

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        shutdown_refs = [actor.shutdown.remote() for actor in self.actors]
        await asyncio.gather(
            *[ref_to_awaitable(r) for r in shutdown_refs],
            return_exceptions=True,
        )
        if self._publisher:
            await self._publisher.close()
        if self._consumer:
            await self._consumer.close()
        ray.shutdown()
        logger.info("Orchestrator shut down")


# ─── Ray async helper ───────────────────────────────────────────────────────

async def ref_to_awaitable(ref):
    """Convert a Ray ObjectRef to an asyncio-awaitable."""
    return await asyncio.wrap_future(ref.future())


# ─── Main entry point ───────────────────────────────────────────────────────

async def main(total_ticks: int = 100) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    orch = SimulationOrchestrator()
    await orch.initialize()

    # Example event schedule — now analyzed by the LLM for multi-asset impact
    shock_schedule: dict[int, str] = {
        10: (
            "The United States and Israel have launched massive air and missile "
            "strikes against multiple Iranian cities, including the capital, Tehran. "
            'Named "Operation Epic Fury" by the Pentagon and "Lion\'s Roar" by Israel, '
            "the campaign aims to destroy Iran's missile industry and naval capabilities. "
            "President Donald Trump described the operations as an effort to eliminate "
            "imminent threats and prevent Iran from obtaining nuclear weapons."
        ),
        50: "Major tech earnings miss; sector-wide selloff",
        75: "Surprise trade deal announced; risk-on sentiment",
    }

    logger.info("Starting simulation with %d agents (continuous mode)",
                settings.num_agents)

    # ── Redis for simulation control ────────────────────────────────────
    import redis.asyncio as aioredis
    ctrl_redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    await ctrl_redis.set("sim:status", "running")
    await ctrl_redis.delete("sim:stop")

    tick = 0
    while tick < total_ticks:
        # ── Check for stop signal ───────────────────────────────────────
        stop_flag = await ctrl_redis.get("sim:stop")
        if stop_flag == "1":
            logger.info("Stop signal received at tick %d", tick)
            await ctrl_redis.set("sim:status", "stopped")
            # Wait for restart signal
            while True:
                await asyncio.sleep(1)
                stop_flag = await ctrl_redis.get("sim:stop")
                if stop_flag == "0":
                    logger.info("Restart signal received, resuming from tick %d", tick + 1)
                    await ctrl_redis.set("sim:status", "running")
                    break

        tick += 1
        headline = shock_schedule.get(tick)
        severity = 0.0

        # Run LLM event analysis and inject as SCM shock (with timeout so tick loop never hangs)
        if headline:
            try:
                analysis = await asyncio.wait_for(
                    orch.analyze_and_inject_shock(headline),
                    timeout=30.0,  # 30s max for event analysis
                )
            except asyncio.TimeoutError:
                logger.warning("Event analysis timed out for tick %d, skipping", tick)
                analysis = None
            except Exception as exc:
                logger.error("Event analysis failed: %s", exc)
                analysis = None
            if analysis:
                severity = analysis["severity"] / 10.0
                logger.info(
                    "Event impacts (%d assets): %s",
                    len(analysis["impacts"]),
                    ", ".join(
                        f"{imp['asset']}({imp['direction']}/{imp['magnitude']})"
                        for imp in analysis["impacts"][:5]
                    ),
                )

        summary = await orch.run_tick(tick, headline, severity)

        if tick % 10 == 0 or headline:
            flag = " <<SHOCK" if headline else ""
            logger.info(
                "tick=%d orders=%d execs=%d mid=%.2f spread=%.4f %.1fms%s",
                summary["tick"], summary["orders_submitted"],
                summary["num_trades"], summary["mid_price"],
                summary["spread"], summary["elapsed_ms"], flag,
            )

    await orch.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
