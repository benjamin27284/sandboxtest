"""Tier 1 — Redis State Store.

Keeps quantitative agent metrics in Redis hashes. Only the current snapshot
is passed to the LLM; the full ledger never touches the prompt.

Keys:
    agent:{agent_id}:state   → Hash with cash, inventory, var, drawdown, pnl
    agent:{agent_id}:fills   → List (capped) of recent fill dicts (JSON)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import redis.asyncio as aioredis


@dataclass
class AgentState:
    """Quantitative snapshot passed to the LLM as structured context."""
    agent_id: str
    cash: float = 100_000.0
    inventory: int = 0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    value_at_risk: float = 0.0
    max_drawdown: float = 0.0
    drawdown_limit: float = -5_000.0
    tick: int = 0
    total_fills: int = 0

    def to_prompt_string(self) -> str:
        """One-line structured summary for the LLM system prompt."""
        return (
            f"Cash=${self.cash:,.2f} | Inventory={self.inventory} units | "
            f"Unrealized P&L=${self.unrealized_pnl:,.2f} | "
            f"Realized P&L=${self.realized_pnl:,.2f} | "
            f"VaR=${self.value_at_risk:,.2f} | "
            f"Drawdown=${self.max_drawdown:,.2f} (limit={self.drawdown_limit:,.2f}) | "
            f"Fills={self.total_fills}"
        )


class StateStore:
    """Async Redis-backed state store for agent quantitative metrics."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis: Optional[aioredis.Redis] = None
        self._url = redis_url

    async def connect(self) -> None:
        self._redis = aioredis.from_url(
            self._url, decode_responses=True, max_connections=100
        )

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()

    def _key(self, agent_id: str) -> str:
        return f"agent:{agent_id}:state"

    async def save_state(self, state: AgentState) -> None:
        """Persist agent state as a Redis hash."""
        data = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in asdict(state).items()}
        await self._redis.hset(self._key(state.agent_id), mapping=data)

    async def load_state(self, agent_id: str) -> Optional[AgentState]:
        """Load agent state from Redis. Returns None if not found."""
        raw = await self._redis.hgetall(self._key(agent_id))
        if not raw:
            return None
        return AgentState(
            agent_id=raw["agent_id"],
            cash=float(raw["cash"]),
            inventory=int(raw["inventory"]),
            unrealized_pnl=float(raw["unrealized_pnl"]),
            realized_pnl=float(raw["realized_pnl"]),
            value_at_risk=float(raw["value_at_risk"]),
            max_drawdown=float(raw["max_drawdown"]),
            drawdown_limit=float(raw["drawdown_limit"]),
            tick=int(raw["tick"]),
            total_fills=int(raw["total_fills"]),
        )

    async def record_fill(self, agent_id: str, fill: dict,
                          max_recent: int = 50) -> None:
        """Append a fill to the agent's recent-fills list (capped)."""
        key = f"agent:{agent_id}:fills"
        await self._redis.lpush(key, json.dumps(fill))
        await self._redis.ltrim(key, 0, max_recent - 1)

    async def get_recent_fills(self, agent_id: str,
                               count: int = 10) -> list[dict]:
        """Retrieve the N most recent fills."""
        key = f"agent:{agent_id}:fills"
        raw = await self._redis.lrange(key, 0, count - 1)
        return [json.loads(r) for r in raw]
