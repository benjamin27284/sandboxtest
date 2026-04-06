"""LLM-powered Fundamental Value Fund agent for the LOB simulation.

Uses the Aliyun DashScope API via the OpenAI-compatible chat/completions
endpoint.  API key and base URL are loaded from a .env file.

Dependencies:
  - With `openai` + `python-dotenv` installed → uses the official SDK.
  - Without them → falls back to stdlib `requests` + manual .env parsing.
    This lets the module run in restricted environments (no PyPI access).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agents import BaseAgent
from lob import LimitOrderBook, Order, Side

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loading  — prefer python-dotenv, fall back to manual parsing
# ---------------------------------------------------------------------------

_env_path = Path(__file__).resolve().parent / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(_env_path)
except ImportError:
    # Manual .env parser (KEY=VALUE, no quotes handling needed here)
    if _env_path.exists():
        for line in _env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
_DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

if not _DASHSCOPE_API_KEY:
    raise EnvironmentError(
        "DASHSCOPE_API_KEY not found. "
        "Please set it in .env or as an environment variable."
    )

# ---------------------------------------------------------------------------
# API backend — prefer openai SDK, fall back to requests
# ---------------------------------------------------------------------------

_USE_OPENAI_SDK = False

try:
    from openai import AsyncOpenAI
    _client = AsyncOpenAI(
        api_key=_DASHSCOPE_API_KEY,
        base_url=_DASHSCOPE_BASE_URL,
    )
    _USE_OPENAI_SDK = True
    logger.info("Using openai SDK for DashScope API")
except ImportError:
    import requests
    logger.info("openai SDK not found – using requests fallback")


async def _call_dashscope_sdk(
    model: str,
    messages: list[dict],
    temperature: float,
    timeout: float,
) -> str:
    """Call DashScope via the openai AsyncOpenAI client."""
    response = await asyncio.wait_for(
        _client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        ),
        timeout=timeout,
    )
    return response.choices[0].message.content.strip()


async def _call_dashscope_requests(
    model: str,
    messages: list[dict],
    temperature: float,
    timeout: float,
) -> str:
    """Call DashScope via stdlib requests (sync, wrapped in executor)."""
    url = f"{_DASHSCOPE_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {_DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    loop = asyncio.get_running_loop()
    resp = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: requests.post(
                url, headers=headers, json=payload, timeout=timeout
            ),
        ),
        timeout=timeout + 2,  # small buffer over the requests timeout
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


async def _call_dashscope(
    model: str,
    messages: list[dict],
    temperature: float,
    timeout: float,
) -> str:
    """Dispatch to whichever backend is available."""
    if _USE_OPENAI_SDK:
        return await _call_dashscope_sdk(model, messages, temperature, timeout)
    return await _call_dashscope_requests(model, messages, temperature, timeout)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a Fundamental Value Fund manager operating in a simulated equity market.

Your job:
1. Read the provided **news** about the asset.
2. Compare it with the **current market price**.
3. Estimate whether the asset is overvalued, undervalued, or fairly valued.
4. Decide on an action: buy, sell, or hold.

Decision rules:
- BUY  when your fundamental estimate is ABOVE the current price (undervalued).
- SELL when your fundamental estimate is BELOW the current price (overvalued).
- HOLD when the price is near your estimate or the news is ambiguous.
- Set `target_price` to your best estimate of the asset's fair value.
- Set `confidence` between 0.0 (no conviction) and 1.0 (maximum conviction).

You MUST reply with ONLY a single JSON object—no markdown, no commentary:
{"action": "buy"|"sell"|"hold", "target_price": <float>, "confidence": <float 0.0-1.0>}
"""

# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMDecision:
    action: str   # "buy" | "sell" | "hold"
    target_price: float
    confidence: float


_HOLD_DEFAULT = LLMDecision(action="hold", target_price=0.0, confidence=0.0)

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class LLMFundamentalAgent(BaseAgent):
    """Fundamental-value agent whose trade decisions are generated by an LLM.

    Call `generate_order()` (async) to get an `LLMDecision`, then use
    `act_on_decision()` to translate that into a LOB order.
    """

    def __init__(
        self,
        agent_id: str,
        cash: float = 100_000.0,
        inventory: int = 0,
        model: str = "qwen-plus",
        order_qty: int = 5,
        timeout: float = 15.0,
    ) -> None:
        super().__init__(agent_id, cash, inventory)
        self.model = model
        self.order_qty = order_qty
        self.timeout = timeout

    # -- Core async method ----------------------------------------------------

    async def generate_order(
        self,
        current_price: float,
        news_string: str,
    ) -> LLMDecision:
        """Query the DashScope LLM and return a structured trading decision.

        Falls back to HOLD on timeout, network error, or malformed JSON.
        """
        user_message = (
            f"Current market price: {current_price:.2f}\n"
            f"Latest news:\n{news_string}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            raw = await _call_dashscope(
                model=self.model,
                messages=messages,
                temperature=0.3,
                timeout=self.timeout,
            )

            # Strip markdown code fences if the model wraps its reply
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            data = json.loads(raw)
            return LLMDecision(
                action=str(data["action"]).lower(),
                target_price=float(data["target_price"]),
                confidence=max(0.0, min(1.0, float(data["confidence"]))),
            )

        except asyncio.TimeoutError:
            logger.warning("%s: DashScope API timed out – defaulting to HOLD", self.agent_id)
            return _HOLD_DEFAULT
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("%s: malformed LLM response (%s) – defaulting to HOLD", self.agent_id, exc)
            return _HOLD_DEFAULT
        except Exception as exc:
            logger.warning("%s: unexpected API error (%s) – defaulting to HOLD", self.agent_id, exc)
            return _HOLD_DEFAULT

    # -- Synchronous LOB interface --------------------------------------------

    def act(self, tick: int, lob: LimitOrderBook, last_price: Optional[float]) -> None:
        """Synchronous `act()` required by BaseAgent.

        For the LLM agent, callers should use `act_async()` instead inside
        an async simulation loop so the API call doesn't block other agents.
        This synchronous fallback simply does nothing (hold).
        """
        pass  # Use act_async in an async simulation loop

    async def act_async(
        self,
        tick: int,
        lob: LimitOrderBook,
        last_price: Optional[float],
        news_string: str = "No news available.",
    ) -> Optional[LLMDecision]:
        """Async tick handler: queries the LLM and submits an order."""
        current_price = last_price if last_price is not None else (lob.get_mid_price() or 100.0)
        decision = await self.generate_order(current_price, news_string)
        self.act_on_decision(decision, tick, lob, current_price)
        return decision

    def act_on_decision(
        self,
        decision: LLMDecision,
        tick: int,
        lob: LimitOrderBook,
        current_price: float,
    ) -> None:
        """Translate an LLMDecision into a limit order on the LOB."""
        if decision.action == "hold" or decision.confidence < 0.1:
            return

        # Scale quantity by confidence: more conviction → larger order
        qty = max(1, int(self.order_qty * decision.confidence))

        if decision.action == "buy":
            # Bid at the target price (but no higher than target)
            price = round(min(decision.target_price, current_price * 1.02), 2)
            lob.submit(Order(self.agent_id, Side.BUY, price, qty, tick))
        elif decision.action == "sell":
            # Ask at the target price (but no lower than target)
            price = round(max(decision.target_price, current_price * 0.98), 2)
            lob.submit(Order(self.agent_id, Side.SELL, price, qty, tick))


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

async def _demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    lob = LimitOrderBook()
    lob.submit(Order("SEED", Side.BUY,  price=99.0, quantity=100, tick_submitted=0))
    lob.submit(Order("SEED", Side.SELL, price=101.0, quantity=100, tick_submitted=0))

    agent = LLMFundamentalAgent(agent_id="LLM01")

    news_items = [
        "Company Q3 earnings beat estimates by 15%. Revenue up 8% YoY.",
        "FDA rejects the company's lead drug candidate. Stock downgraded by 3 analysts.",
        "Market conditions are stable. No significant news today.",
    ]

    print(f"\n{'tick':>4}  {'action':>6}  {'target':>8}  {'conf':>5}  "
          f"{'mid':>8}  {'best_bid':>8}  {'best_ask':>8}")
    print("-" * 62)

    for tick, news in enumerate(news_items, start=1):
        mid = lob.get_mid_price()
        decision = await agent.act_async(tick, lob, mid, news)
        lob.match_orders()

        mid_str = f"{lob.get_mid_price():.2f}" if lob.get_mid_price() is not None else "N/A"
        bb_str = f"{lob.best_bid():.2f}" if lob.best_bid() is not None else "N/A"
        ba_str = f"{lob.best_ask():.2f}" if lob.best_ask() is not None else "N/A"

        print(
            f"{tick:4d}  {decision.action:>6}  {decision.target_price:8.2f}  "
            f"{decision.confidence:5.2f}  "
            f"{mid_str:>8}  {bb_str:>8}  {ba_str:>8}"
        )
        print(f"       News: {news[:60]}...")

    print(f"\nAgent state: {agent}")


if __name__ == "__main__":
    asyncio.run(_demo())
