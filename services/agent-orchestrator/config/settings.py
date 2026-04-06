"""Centralized configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    kafka_brokers: str = os.getenv("KAFKA_BROKERS", "localhost:9092")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6334"))

    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    eodhd_api_key: str = os.getenv("EODHD_API_KEY", "")
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    )

    num_agents: int = int(os.getenv("NUM_AGENTS", "1000"))
    ticks_per_summary: int = int(os.getenv("TICKS_PER_SUMMARY", "10"))
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "15"))
    ray_num_cpus: int = int(os.getenv("RAY_NUM_CPUS", "8"))

    primary_model: str = os.getenv("PRIMARY_LLM_MODEL", "qwen-plus")
    slm_model: str = os.getenv("SLM_MODEL", "qwen-turbo")

    orders_topic: str = "orders_submit"
    executions_topic: str = "executions"
    market_data_topic: str = "market_data"
    shocks_topic: str = "exogenous_shocks"

    # Multiple API keys for rate limit distribution (comma-separated)
    # e.g. DASHSCOPE_API_KEYS=sk-key1,sk-key2,sk-key3,sk-key4,sk-key5
    # Each key gets assigned to a subset of agents round-robin.
    # If not set, falls back to single DASHSCOPE_API_KEY for all agents.
    dashscope_api_keys: tuple = ()

    def __post_init__(self):
        keys_str = os.getenv("DASHSCOPE_API_KEYS", "")
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not keys and self.dashscope_api_key:
            keys = [self.dashscope_api_key]
        object.__setattr__(self, "dashscope_api_keys", tuple(keys))


settings = Settings()

# ─── Persona Sophistication Levels ────────────────────────────────────────────
# Higher sophistication = more Bayesian precision (tighter belief updates).
# Weighted by DAILY TRADING VOLUME, not AUM.

PERSONA_SOPHISTICATION: dict[str, float] = {
    "HFT Market Maker": 1.2,                  # 45% vol
    "Momentum Quantitative Trader": 1.1,       # 25% vol
    "Macro Event-Driven Fund": 1.0,            # 15% vol
    "Retail Sentiment Trader": 2.5,            # 10% vol
    "Passive Index Fund Manager": 0.5,         #  5% vol
    # Legacy programmatic archetypes
    "institutional_market_maker": 1.2,
    "fundamental_value_fund": 0.9,
    "quant_momentum_fund": 1.1,
    "retail_sentiment": 2.5,
}
