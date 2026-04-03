"""BaseActor — Ray Actor implementing the three-tier cognitive architecture.

Each of the 1,000 agents runs as an independent, non-blocking Ray actor.
The actor coordinates:

  Tier 1 (State Store):     Redis — quantitative snapshot only
  Tier 2 (Semantic Memory): Qdrant — RAG retrieval of relevant historical events
  Tier 3 (Episodic Buffer): SLM — rolling summarization of recent trading activity
  Math Shield:              QuantEngine — deterministic Bayesian / BL / VaR computations

The LLM receives a compact, curated prompt assembled from all four sources,
never raw tick data or full ledgers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from typing import Any, Optional

import re

import ray

TRAPPED_TAGS = re.compile(
    r"<\s*/?\s*(?:system|tool|function_call|admin|prompt|instruction)[^>]*>",
    re.IGNORECASE,
)

from memory.state_store import AgentState, StateStore
from memory.semantic_memory import SemanticMemory
from memory.episodic_buffer import EpisodicBuffer, TickObservation
from math_engine.quant_models import (
    QuantEngine, QuantOutput, CPTState, cpt_signal, update_reference_price,
    data_terminal,
)
from config.settings import PERSONA_SOPHISTICATION

logger = logging.getLogger(__name__)


def _tag_trap(source: str, content: str) -> str:
    """Wrap context with source attribution tags for hallucination mitigation."""
    return f"[SOURCE:{source}]\n{content}\n[/SOURCE:{source}]"

# ─── LLM System Prompt Template ─────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """\
You are Agent {agent_id}, a {persona} operating in a simulated equity market.

Your role:
- Analyze the quantitative analysis, your recent history, and any relevant \
news to decide whether to BUY, SELL, or HOLD.
- You do NOT compute math. The quantitative analysis below was computed \
deterministically and is authoritative.
- Focus on strategic reasoning, narrative interpretation of events, and \
risk management.

{shock_behavior_context}

{quant_context}

{portfolio_context}

{episodic_context}

{rag_context}

Current market: mid_price=${mid_price:.2f}, spread=${spread:.4f}
Loss aversion signal (CPT): {loss_aversion_signal:.3f} (positive=gain domain/hold bias, negative=loss domain/sell pressure)
DCF terminal fair value: ${dcf_fair_value:.2f}
Market regime: {market_regime} | Your participation weight: {participation_pct:.0f}%
INSTRUCTION: Only cite numerical values that appear inside a [SOURCE:...] tag above.
Do not invent, extrapolate, or assume numbers. If uncertain about a value, output action=hold.

Respond with ONLY valid JSON:
{{"action": "buy"|"sell"|"hold", "target_price": <float>, "confidence": <float 0.0-1.0>, "reasoning": "<1 sentence>"}}
"""


# ─── Actor Personas — Weighted by Daily Trading Volume ──────────────────────
#
# In a shock simulation, what matters is WHO IS ACTIVELY TRADING, not who
# holds the most assets. Passive index funds hold 50%+ of AUM but contribute
# only ~5% of intraday volume. HFT firms hold minimal overnight inventory
# but generate ~45% of daily volume.
#
# Source framework: SEC Equity Market Structure reports, academic literature
# on Flash Crash dynamics (Kirilenko et al. 2017), and market microstructure.
#
# Each persona has:
#   - volume_weight: fraction of daily trading volume in normal conditions
#   - shock_weight: how their participation changes during high-vol events
#   - shock_behavior: describes their behavioral shift during a shock

PERSONA_POOL = [
    {
        "name": "HFT Market Maker providing continuous liquidity with tight spreads",
        "volume_weight": 0.45,
        "shock_weight_min": 0.10,
        "shock_behavior": "LIQUIDITY PULL",
        "uses_llm": False,  # Pure algorithmic — no LLM needed
        "shock_desc": (
            "You are in LIQUIDITY PULL mode. Volatility has exceeded your risk "
            "threshold. You MUST widen your spreads dramatically (3-5x normal) and "
            "reduce your quoting size. You are pulling liquidity from the book. "
            "Only quote if the spread compensates for inventory risk. Prefer to "
            "HOLD or quote very wide."
        ),
    },
    {
        "name": "Momentum Quantitative Trader executing statistical arbitrage and trend-following",
        "volume_weight": 0.25,
        "shock_weight_min": 0.35,
        "shock_behavior": "VOLATILITY CHASER",
        "uses_llm": False,  # Systematic — follows math signals, not reasoning
        "shock_desc": (
            "You are in VOLATILITY CHASER mode. You detect extreme price velocity "
            "and order book imbalance. Your stop-losses are triggering and your "
            "momentum signals are screaming. You MUST trade aggressively in the "
            "direction of the move. Pile into the momentum — this is your edge. "
            "High confidence, aggressive sizing."
        ),
    },
    {
        "name": "Macro Event-Driven Fund repricing fundamentals based on macro analysis",
        "volume_weight": 0.15,
        "shock_weight_min": 0.30,
        "shock_behavior": "FUNDAMENTAL REPRICER",
        "uses_llm": True,   # Needs LLM to interpret news and set target price
        "shock_desc": (
            "You are in FUNDAMENTAL REPRICER mode. You have processed the news "
            "and calculated a new fundamental target price. You MUST aggressively "
            "dump or accumulate inventory to reach your target regardless of spread "
            "costs. You are a price-insensitive market-taker right now. The spread "
            "is irrelevant — only your fundamental target matters."
        ),
    },
    {
        "name": "Retail Sentiment Trader influenced by social media and crowd psychology",
        "volume_weight": 0.10,
        "shock_weight_min": 0.20,
        "shock_behavior": "DELAYED MEAN-REVERTER",
        "uses_llm": True,   # Needs LLM to model irrational sentiment-driven behavior
        "shock_desc": (
            "You are a delayed participant. You see the crash/spike AFTER the "
            "algorithms have already moved the price. Your instinct is to 'buy "
            "the dip' or 'sell the rip' — you are a mean-reverter providing exit "
            "liquidity for the quant funds. You trade with LOWER confidence and "
            "SMALLER size than the algorithmic agents. You react 1-2 ticks late."
        ),
    },
    {
        "name": "Passive Index Fund Manager executing only for rebalancing with minimal market impact",
        "volume_weight": 0.05,
        "shock_weight_min": 0.05,
        "shock_behavior": "INERTIA",
        "uses_llm": False,  # Always holds — no decision needed
        "shock_desc": (
            "You are in INERTIA mode. You do NOT trade intraday during shocks. "
            "You only execute at market close for index rebalancing. During a shock, "
            "you MUST output action=hold with high confidence. You will only increase "
            "activity if this becomes a multi-day event triggering retail redemptions."
        ),
    },
]

# Flat lists for backward compatibility
PERSONAS = [p["name"] for p in PERSONA_POOL]
PERSONA_VOLUME_WEIGHTS = {p["name"]: p["volume_weight"] for p in PERSONA_POOL}


def assign_persona(agent_index: int, total_agents: int) -> str:
    """Assign a persona based on daily-trading-volume proportional weights."""
    cumulative = 0.0
    position = agent_index / total_agents
    for p in PERSONA_POOL:
        cumulative += p["volume_weight"]
        if position < cumulative:
            return p["name"]
    return PERSONA_POOL[-1]["name"]


def get_persona_config(persona_name: str) -> dict:
    """Look up the full persona config dict by name."""
    for p in PERSONA_POOL:
        if p["name"] == persona_name:
            return p
    return PERSONA_POOL[-1]


# ─── Dynamic Weight Engine ──────────────────────────────────────────────────
# During a shock, HFT pulls liquidity → their volume share drops from 45% to
# ~10%. The 35% vacuum gets redistributed to Momentum (absorbs ~15%),
# Macro (absorbs ~15%), and Retail (absorbs ~5%). This creates a liquidity
# cascade where aggressive market-takers face thin books → exponential impact.

def compute_dynamic_weights(
    base_volatility: float,
    current_volatility: float,
    shock_active: bool = False,
) -> dict[str, float]:
    """Compute participation weights based on current volatility regime.

    Returns a dict mapping persona name → current participation weight.
    The weights always sum to 1.0.

    The transition is smooth: as volatility rises from base to 3x base,
    weights interpolate linearly from normal to shock levels.
    """
    if base_volatility <= 0:
        base_volatility = 0.15

    vol_ratio = current_volatility / base_volatility
    # Shock intensity: 0.0 (normal) to 1.0 (full shock) — kicks in above 1.5x vol
    shock_intensity = max(0.0, min(1.0, (vol_ratio - 1.5) / 1.5))

    if shock_active:
        shock_intensity = max(shock_intensity, 0.5)

    weights = {}
    for p in PERSONA_POOL:
        normal_w = p["volume_weight"]
        shock_w = p["shock_weight_min"]
        # Interpolate between normal and shock weights
        weights[p["name"]] = normal_w + shock_intensity * (shock_w - normal_w)

    # Normalize to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


def get_shock_context(persona_name: str, shock_intensity: float) -> str:
    """Get the shock behavior prompt injection for a persona.

    Returns empty string if shock_intensity < 0.3 (normal regime).
    """
    if shock_intensity < 0.3:
        return ""

    config = get_persona_config(persona_name)
    return (
        f"\n## MARKET REGIME: HIGH VOLATILITY — {config['shock_behavior']}\n"
        f"{config['shock_desc']}\n"
        f"Shock intensity: {shock_intensity:.0%}\n"
    )


# ─── Ray Actor ───────────────────────────────────────────────────────────────

@ray.remote(num_cpus=0.01, max_concurrency=2)  # lightweight: 1000 actors fit in 10 CPUs
class TradingAgentActor:
    """Autonomous trading agent running as a Ray actor.

    Lifecycle per tick:
        1. Load state from Redis (Tier 1)
        2. Run QuantEngine (Math Shield) → deterministic numbers
        3. Summarize if needed (Tier 3 — SLM episodic flush)
        4. Retrieve relevant context via RAG (Tier 2 — if news event)
        5. Assemble prompt → call primary LLM → parse decision
        6. Publish order to Kafka
        7. Save updated state to Redis
    """

    def __init__(
        self,
        agent_id: str,
        redis_url: str,
        qdrant_host: str,
        qdrant_port: int,
        llm_call_fn_ref: Any,       # Ray object ref to the LLM call function
        slm_call_fn_ref: Any,       # Ray object ref to the SLM summarizer
        embed_fn_ref: Any,          # Ray object ref to the embedding function
        persona_index: Optional[int] = None,
        ticks_per_summary: int = 10,
        total_agents: int = 1000,
    ) -> None:
        self.agent_id = agent_id
        if persona_index is not None:
            self.persona = PERSONAS[persona_index % len(PERSONAS)]
        else:
            try:
                idx = int(agent_id.split("-")[1])
            except (IndexError, ValueError):
                idx = hash(agent_id)
            self.persona = assign_persona(idx, total_agents)

        self._persona_config = get_persona_config(self.persona)

        # ── Tier 1: State Store (Redis) ──────────────────────────────────────
        self._state_store = StateStore(redis_url)
        self._state: Optional[AgentState] = None

        # ── Tier 2: Semantic Memory (Qdrant) ─────────────────────────────────
        self._semantic = SemanticMemory(
            host=qdrant_host, port=qdrant_port,
            embed_fn=embed_fn_ref,
        )

        # ── Tier 3: Episodic Buffer (SLM) ───────────────────────────────────
        self._episodic = EpisodicBuffer(
            summarize_fn=slm_call_fn_ref,
            flush_every=ticks_per_summary,
        )

        # ── Math Shield ──────────────────────────────────────────────────────
        self._quant = QuantEngine()

        # ── Agent belief state (fed into Bayesian update) ────────────────────
        self._belief_mu: float = 100.0      # prior mean
        self._belief_sigma: float = 5.0     # prior uncertainty

        # ── CPT state (Cumulative Prospect Theory) ────────────────────────
        self._cpt_state: Optional[CPTState] = None  # initialized on first tick

        # ── Persona sophistication (heterogeneous Bayesian precision) ─────
        self._sophistication = 1.0
        for key, val in PERSONA_SOPHISTICATION.items():
            if key in self.persona:
                self._sophistication = val
                break

        # ── LLM call ─────────────────────────────────────────────────────────
        self._llm_call = llm_call_fn_ref

        # ── Dynamic participation state ──────────────────────────────────────
        self._current_participation: float = self._persona_config["volume_weight"]
        self._shock_intensity: float = 0.0
        self._tick_delay: int = 0  # Retail agents have delayed reaction

        self._initialized = False

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Connect to external stores. Called once at simulation start."""
        await self._state_store.connect()
        await self._semantic.connect()

        # Load or create state
        self._state = await self._state_store.load_state(self.agent_id)
        if self._state is None:
            self._state = AgentState(agent_id=self.agent_id)
            await self._state_store.save_state(self._state)

        self._initialized = True
        logger.info(
            "Agent %s initialized (persona: %s, volume_weight: %.0f%%)",
            self.agent_id, self._persona_config["shock_behavior"],
            self._persona_config["volume_weight"] * 100,
        )

    async def shutdown(self) -> None:
        """Persist final state and close connections."""
        if self._state:
            await self._state_store.save_state(self._state)
        await self._state_store.close()
        await self._semantic.close()

    # ─── Dynamic Weight Update (called by orchestrator each tick) ────────────

    def update_participation(
        self, participation_weight: float, shock_intensity: float
    ) -> None:
        """Update this agent's participation weight and shock intensity."""
        self._current_participation = participation_weight
        self._shock_intensity = shock_intensity

    # ─── Per-Tick Entry Point ────────────────────────────────────────────────

    async def on_tick(
        self,
        tick: int,
        mid_price: float,
        spread: float,
        news_headline: Optional[str] = None,
        news_body: Optional[str] = None,
        news_severity: float = 0.0,
    ) -> Optional[dict]:
        """Process a single simulation tick. Returns an order dict or None."""
        if not self._initialized:
            await self.initialize()

        # ── Participation gate: skip tick probabilistically based on weight ──
        # If this agent's persona has reduced participation, randomly skip ticks.
        # This models HFT pulling liquidity (they simply stop quoting).
        if self._current_participation < self._persona_config["volume_weight"]:
            skip_prob = 1.0 - (
                self._current_participation
                / max(self._persona_config["volume_weight"], 0.01)
            )
            if random.random() < skip_prob:
                return None

        # ── Retail delay: skip first ticks of a shock ────────────────────────
        if "Retail" in self.persona and self._shock_intensity > 0.3:
            if self._tick_delay < 2:
                self._tick_delay += 1
                return None
        else:
            self._tick_delay = 0

        # ── Step 1: Update state from Redis (fast) ───────────────────────────
        self._state.tick = tick

        # ── Step 1b: Initialize CPT on first tick ─────────────────────────
        if self._cpt_state is None:
            self._cpt_state = CPTState(reference_price=mid_price)

        # ── Step 2: Deterministic Math Shield ────────────────────────────────
        belief = self._quant.bayesian_update(
            prior_mu=self._belief_mu,
            prior_sigma=self._belief_sigma,
            signal=mid_price,
            signal_sigma=spread * 5,
            agent_sophistication=self._sophistication,
        )
        self._belief_mu = belief.mu
        self._belief_sigma = belief.sigma

        # DCF fair value from simulated data terminal
        dcf_fv = data_terminal.dcf_from_terminal("SIM", tick)

        # B-L single-asset position sizing
        quant_output = self._quant.compute_full(
            prior_mu=self._belief_mu,
            prior_sigma=self._belief_sigma,
            signal=mid_price,
            signal_sigma=spread * 5,
            current_price=mid_price,
            current_position=self._state.inventory,
        )
        quant_output.dcf_fair_value = dcf_fv

        # Update VaR in state
        self._state.value_at_risk = quant_output.value_at_risk

        # CPT: update reference price + compute loss aversion signal
        self._cpt_state = update_reference_price(self._cpt_state, mid_price)
        cpt_sig = cpt_signal(mid_price, self._cpt_state)

        # ── Step 3: Episodic observation + maybe flush (Tier 3) ──────────────
        obs = TickObservation(
            tick=tick,
            mid_price=mid_price,
            spread=spread,
            agent_filled=False,
            news_headline=news_headline,
        )
        self._episodic.observe(obs)
        await self._episodic.maybe_flush()

        # ── Step 4: RAG retrieval (Tier 2 — only if news event) ──────────────
        rag_context = ""
        if news_headline and news_severity > 0.3:
            try:
                rag_context = await self._semantic.retrieve_as_context(
                    news_headline, top_k=3
                )
            except Exception as exc:
                logger.debug("%s: RAG retrieval failed: %s", self.agent_id, exc)
                rag_context = "Historical context unavailable."

        # ── Step 5: Decide — Rule-based (fast, no API) or LLM (slow, needs API)
        if self._persona_config.get("uses_llm", False):
            # ── LLM path: only Macro and Retail agents ───────────────────────
            shock_ctx = get_shock_context(self.persona, self._shock_intensity)
            regime = "HIGH VOLATILITY" if self._shock_intensity > 0.3 else "NORMAL"

            prompt = AGENT_SYSTEM_PROMPT.format(
                agent_id=self.agent_id,
                persona=self.persona,
                shock_behavior_context=shock_ctx,
                quant_context=_tag_trap("QUANT_ENGINE", quant_output.to_prompt_string()),
                portfolio_context=_tag_trap("REDIS_STATE", self._state.to_prompt_string()),
                episodic_context=_tag_trap("EPISODIC_BUFFER", self._episodic.get_context()),
                rag_context=_tag_trap("QDRANT_RAG", rag_context) if rag_context else "",
                mid_price=mid_price,
                spread=spread,
                loss_aversion_signal=cpt_sig,
                dcf_fair_value=dcf_fv,
                market_regime=regime,
                participation_pct=self._current_participation * 100,
            )

            user_message = f"Tick {tick}."
            if news_headline:
                user_message += f" Breaking: {news_headline}"

            decision = await self._call_llm_safe(prompt, user_message)
        else:
            # ── Rule-based path: HFT, Momentum, Passive (no API call) ────────
            decision = self._rule_based_decide(
                mid_price, spread, dcf_fv, cpt_sig, news_severity,
            )

        if decision is None:
            await self._state_store.save_state(self._state)
            return None

        decision = self.verify_decision(decision, mid_price, spread)
        if decision is None:
            await self._state_store.save_state(self._state)
            return None

        # ── Step 7: Build order for Kafka ────────────────────────────────────
        confidence = decision.get("confidence", 0.5)

        # Use Black-Litterman single-asset for position sizing
        bl_position = self._quant.black_litterman_single_asset(
            belief_mu=self._belief_mu,
            belief_sigma=self._belief_sigma,
            current_price=mid_price,
            confidence=confidence,
        )
        base_qty = max(1, min(abs(bl_position), int(10 * confidence)))

        # Scale quantity by participation weight (dynamic weighting)
        participation_scale = self._current_participation / max(
            self._persona_config["volume_weight"], 0.01
        )
        base_qty = max(1, int(base_qty * participation_scale))

        # Respect drawdown limits
        if self._state.max_drawdown <= self._state.drawdown_limit:
            logger.info("%s: drawdown limit hit, forcing HOLD", self.agent_id)
            await self._state_store.save_state(self._state)
            return None

        # HFT Market Makers widen spread during shock
        target_price = decision.get("target_price", mid_price)
        if "HFT" in self.persona and self._shock_intensity > 0.3:
            spread_multiplier = 1.0 + self._shock_intensity * 4.0  # up to 5x
            if decision["action"] == "buy":
                target_price = mid_price - (spread * spread_multiplier / 2)
            elif decision["action"] == "sell":
                target_price = mid_price + (spread * spread_multiplier / 2)

        order = {
            "agent_id": self.agent_id,
            "action": decision["action"],  # keep "action" for MASS signal
            "side": decision["action"],    # keep "side" for LOB engine
            "price": round(target_price, 2),
            "quantity": base_qty,
            "confidence": decision.get("confidence", 0.5),
            "tick": tick,
            "type": "limit",
            "timestamp_ns": time.time_ns(),
        }

        # Persist updated state
        await self._state_store.save_state(self._state)

        return order

    # ─── Fill notification (called by orchestrator after matching) ───────────

    async def on_fill(
        self,
        price: float,
        quantity: int,
        side: str,
        tick: int,
    ) -> None:
        """Update state after one of our orders is filled."""
        if side == "buy":
            self._state.cash -= price * quantity
            self._state.inventory += quantity
        else:
            self._state.cash += price * quantity
            self._state.inventory -= quantity

        self._state.total_fills += 1
        await self._state_store.save_state(self._state)

        fill_record = {
            "tick": tick, "side": side,
            "price": price, "quantity": quantity,
        }
        await self._state_store.record_fill(self.agent_id, fill_record)

    # ─── Introspection ──────────────────────────────────────────────────────

    async def get_state(self) -> dict:
        """Return current agent state (for telemetry / debugging)."""
        return {
            "agent_id": self.agent_id,
            "persona": self.persona,
            "persona_type": self._persona_config["shock_behavior"],
            "cash": self._state.cash,
            "inventory": self._state.inventory,
            "belief_mu": self._belief_mu,
            "belief_sigma": self._belief_sigma,
            "var": self._state.value_at_risk,
            "total_fills": self._state.total_fills,
            "tick": self._state.tick,
            "cpt_reference_price": self._cpt_state.reference_price if self._cpt_state else None,
            "sophistication": self._sophistication,
            "participation_weight": self._current_participation,
            "shock_intensity": self._shock_intensity,
        }

    def verify_decision(
        self, decision: dict, mid_price: float, spread: float
    ) -> Optional[dict]:
        action = decision.get("action", "hold")
        if action not in ("buy", "sell"):
            return None

        target = decision.get("target_price", mid_price)
        if target <= 0:
            return None

        max_deviation = mid_price * 0.10
        if abs(target - mid_price) > max_deviation:
            decision["target_price"] = mid_price
            logger.debug("%s: target clamped to mid (%.2f)", self.agent_id, mid_price)

        conf = decision.get("confidence", 0.5)
        decision["confidence"] = max(0.0, min(1.0, conf))

        if self._state.max_drawdown <= self._state.drawdown_limit:
            if action == "buy" and self._state.inventory > 0:
                return None
            if action == "sell" and self._state.inventory < 0:
                return None

        return decision

    # ─── Rule-based decision engine (no LLM, no API calls) ────────────────

    def _rule_based_decide(
        self,
        mid_price: float,
        spread: float,
        dcf_fv: float,
        cpt_signal_val: float,
        news_severity: float,
    ) -> Optional[dict]:
        """Deterministic decision logic for non-LLM agents.

        HFT Market Maker:  Quote around mid, tighter in normal, wider in shock.
                           Mean-reverts to fair value. Targets spread capture.
        Momentum Trader:   Follows price trend (belief_mu vs mid_price).
                           Buys if price rising, sells if falling.
        Passive Index:     Always holds. Never trades intraday.
        """
        behavior = self._persona_config["shock_behavior"]

        # ── PASSIVE INDEX: always hold ───────────────────────────────────────
        if behavior == "INERTIA":
            return None

        # ── HFT MARKET MAKER: mean-reversion + spread capture ────────────────
        if behavior == "LIQUIDITY PULL":
            # In shock mode, widen spreads and reduce activity
            if self._shock_intensity > 0.3:
                if spread < mid_price * 0.005 and random.random() < 0.7:
                    return None  # pull liquidity 70% of the time

            # Mean-revert toward belief (Bayesian posterior)
            deviation = (mid_price - self._belief_mu) / max(self._belief_sigma, 0.01)
            # Small random offset for heterogeneity between HFT agents
            jitter = (random.random() - 0.5) * spread * 0.5

            if deviation > 0.3:  # price above fair → sell
                action = "sell"
                # Aggressive: cross the spread to sell NOW
                target = mid_price - spread * 0.1 + jitter
                confidence = min(0.9, 0.4 + abs(deviation) * 0.2)
            elif deviation < -0.3:  # price below fair → buy
                action = "buy"
                # Aggressive: cross the spread to buy NOW
                target = mid_price + spread * 0.1 + jitter
                confidence = min(0.9, 0.4 + abs(deviation) * 0.2)
            else:
                # Near fair value — provide liquidity (alternate sides)
                if self._state.tick % 2 == 0:
                    action = "buy"
                    target = mid_price - spread * 0.2 + jitter
                else:
                    action = "sell"
                    target = mid_price + spread * 0.2 + jitter
                confidence = 0.5

            return {
                "action": action,
                "target_price": round(target, 2),
                "confidence": confidence,
                "reasoning": "algorithmic market-making",
            }

        # ── MOMENTUM TRADER: trend-following ─────────────────────────────────
        if behavior == "VOLATILITY CHASER":
            # Compare belief (moving average) to current price
            momentum = (mid_price - self._belief_mu) / max(self._belief_mu, 0.01)

            # Add noise for heterogeneity
            noise = (random.random() - 0.5) * 0.02
            momentum += noise

            if momentum > 0.005:  # price trending up → buy
                action = "buy"
                # Chase: willing to pay above mid
                target = mid_price + spread * 0.5 * abs(momentum) * 100
                confidence = min(0.95, 0.5 + abs(momentum) * 10)
            elif momentum < -0.005:  # price trending down → sell
                action = "sell"
                target = mid_price - spread * 0.5 * abs(momentum) * 100
                confidence = min(0.95, 0.5 + abs(momentum) * 10)
            else:
                return None  # no clear trend, sit out

            # In shock: amplify (this is their edge — ride the wave)
            if self._shock_intensity > 0.3:
                confidence = min(1.0, confidence * 1.5)

            return {
                "action": action,
                "target_price": round(target, 2),
                "confidence": confidence,
                "reasoning": f"momentum signal {momentum:.4f}",
            }

        # Fallback: hold
        return None

    # ─── Internal: safe LLM call with fallback ──────────────────────────────

    async def _call_llm_safe(
        self, system_prompt: str, user_message: str
    ) -> Optional[dict]:
        """Call the LLM and parse JSON. Returns None on any failure."""
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "15"))

        try:
            raw = await asyncio.wait_for(
                self._llm_call(system_prompt, user_message),
                timeout=timeout,
            )

            text = raw.strip()

            if TRAPPED_TAGS.search(text):
                logger.warning("%s: tag trapping — blocked injected tags", self.agent_id)
                return None

            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            data = json.loads(text)

            # Validate
            action = str(data.get("action", "hold")).lower()
            if action not in ("buy", "sell", "hold"):
                action = "hold"

            return {
                "action": action,
                "target_price": float(data.get("target_price", 0)),
                "confidence": max(0.0, min(1.0, float(data.get("confidence", 0)))),
                "reasoning": str(data.get("reasoning", "")),
            }

        except asyncio.TimeoutError:
            logger.debug("%s: LLM timeout", self.agent_id)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.debug("%s: malformed LLM response: %s", self.agent_id, exc)
        except Exception as exc:
            logger.debug("%s: LLM error: %s", self.agent_id, exc)

        return None
