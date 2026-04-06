"""Tier 3 — Episodic Memory Summarization via a Small Language Model (SLM).

Instead of appending every tick's raw data to the LLM prompt (context bloat),
a fast local SLM compresses the agent's recent trading outcomes and market
trends into a short narrative every N ticks.

The "Cognitive Buffer" holds raw tick observations between summarization
cycles, then flushes them into a compressed summary.

SLM options (local, fast, cheap):
  - Qwen2-0.5B / Qwen2-1.5B via DashScope
  - Phi-3-mini via Ollama
  - Any small model behind an OpenAI-compatible API
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Optional

logger = logging.getLogger(__name__)

SUMMARIZATION_PROMPT = """\
You are a concise financial analyst assistant. Summarize the following raw \
trading observations into a brief narrative (3-5 sentences max). Focus on:
1. Price trend direction and magnitude.
2. Agent's trading activity and P&L movement.
3. Notable market events or volatility changes.

Raw observations:
{observations}

Write a compressed summary:"""


@dataclass
class TickObservation:
    """Raw observation recorded each tick before summarization."""
    tick: int
    mid_price: Optional[float]
    spread: Optional[float]
    agent_filled: bool
    fill_side: Optional[str] = None
    fill_price: Optional[float] = None
    fill_qty: Optional[int] = None
    news_headline: Optional[str] = None

    def to_line(self) -> str:
        parts = [f"t={self.tick} mid={self.mid_price:.2f}" if self.mid_price else f"t={self.tick}"]
        if self.spread is not None:
            parts.append(f"spread={self.spread:.3f}")
        if self.agent_filled:
            parts.append(f"FILL {self.fill_side} {self.fill_qty}@{self.fill_price:.2f}")
        if self.news_headline:
            parts.append(f"NEWS: {self.news_headline}")
        return " | ".join(parts)


class EpisodicBuffer:
    """Rolling cognitive buffer with periodic SLM-driven summarization.

    Usage:
        buffer = EpisodicBuffer(summarize_fn=my_slm_call, flush_every=10)
        buffer.observe(tick_obs)         # call each tick
        await buffer.maybe_flush()       # call each tick; flushes every N
        summary = buffer.get_context()   # inject into LLM prompt
    """

    def __init__(
        self,
        summarize_fn: Callable[[str], Awaitable[str]],
        flush_every: int = 10,
        max_summaries: int = 5,
    ) -> None:
        # The SLM call: async fn(prompt: str) -> str
        self._summarize_fn = summarize_fn
        self._flush_every = flush_every

        # Raw observations since last flush
        self._raw_buffer: list[TickObservation] = []

        # Rolling window of compressed summaries
        self._summaries: deque[str] = deque(maxlen=max_summaries)

        self._ticks_since_flush: int = 0

    def observe(self, obs: TickObservation) -> None:
        """Record a single tick's observation into the raw buffer."""
        self._raw_buffer.append(obs)
        self._ticks_since_flush += 1

    async def maybe_flush(self) -> bool:
        """If enough ticks have accumulated, summarize and flush.

        Returns True if a flush occurred.
        """
        if self._ticks_since_flush < self._flush_every:
            return False

        if not self._raw_buffer:
            return False

        # Build observation text
        obs_text = "\n".join(obs.to_line() for obs in self._raw_buffer)
        prompt = SUMMARIZATION_PROMPT.format(observations=obs_text)

        try:
            summary = await self._summarize_fn(prompt)
            summary = summary.strip()
            if summary:
                self._summaries.append(summary)
                logger.debug("Episodic flush: compressed %d ticks into summary",
                             len(self._raw_buffer))
        except Exception as exc:
            # On SLM failure, create a basic statistical summary as fallback
            logger.warning("SLM summarization failed (%s), using fallback", exc)
            summary = self._fallback_summary()
            self._summaries.append(summary)

        self._raw_buffer.clear()
        self._ticks_since_flush = 0
        return True

    def get_context(self) -> str:
        """Return the compressed episodic context for LLM prompt injection.

        Most recent summary is last (recency bias).
        """
        if not self._summaries:
            return "No trading history yet."

        parts = ["Recent trading history (summarized):"]
        for i, summary in enumerate(self._summaries, 1):
            parts.append(f"  Period {i}: {summary}")
        return "\n".join(parts)

    def _fallback_summary(self) -> str:
        """Deterministic statistical fallback when SLM is unavailable."""
        if not self._raw_buffer:
            return "No observations."

        prices = [o.mid_price for o in self._raw_buffer if o.mid_price is not None]
        fills = [o for o in self._raw_buffer if o.agent_filled]
        news = [o.news_headline for o in self._raw_buffer if o.news_headline]

        parts = []
        if prices:
            p_start, p_end = prices[0], prices[-1]
            change_pct = (p_end - p_start) / p_start * 100 if p_start else 0
            parts.append(
                f"Price moved from {p_start:.2f} to {p_end:.2f} "
                f"({change_pct:+.2f}%) over {len(self._raw_buffer)} ticks."
            )
        parts.append(f"Agent had {len(fills)} fills.")
        if news:
            parts.append(f"News events: {'; '.join(news[:3])}")

        return " ".join(parts)
