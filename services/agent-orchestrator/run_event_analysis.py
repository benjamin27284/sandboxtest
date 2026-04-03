#!/usr/bin/env python3
"""Standalone runner: analyze a news event via LLM, then simulate each
impacted asset using agent-driven LOB simulation.

Each asset gets 16 heterogeneous agents trading in a dedicated OrderBook
for 100 ticks.  Prices emerge from agent interaction, not equations.

Usage:
    python run_event_analysis.py              # uses DashScope API
    python run_event_analysis.py --offline    # uses cached sample output
"""

import asyncio
import json
import sys
import os

# Ensure we can import project modules
sys.path.insert(0, os.path.dirname(__file__))

from config.settings import settings
from event_impact_analyzer import analyze_event, simulate_all_impacts

EVENT_TEXT = (
    "The United States and Israel have launched massive air and missile "
    "strikes against multiple Iranian cities, including the capital, Tehran. "
    'Named "Operation Epic Fury" by the Pentagon and "Lion\'s Roar" by Israel, '
    "the campaign aims to destroy Iran's missile industry and naval capabilities. "
    "President Donald Trump described the operations as an effort to eliminate "
    "imminent threats and prevent Iran from obtaining nuclear weapons."
)

# ─── Cached sample LLM response (for offline / demo use) ─────────────────

_SAMPLE_LLM_RESPONSE = json.dumps({
    "event": EVENT_TEXT,
    "event_category": "military_conflict",
    "severity": 9,
    "time_horizon": "short_term",
    "impacts": [
        {"asset": "Crude Oil", "ticker": "CL=F", "direction": "up", "magnitude": "high", "confidence": 0.97, "reason": "Direct military strikes on Iran, a major OPEC producer and guardian of the Strait of Hormuz, trigger immediate supply disruption fears and a massive risk premium on global oil prices."},
        {"asset": "Gold", "ticker": "GC=F", "direction": "up", "magnitude": "high", "confidence": 0.96, "reason": "Classic safe-haven surge as a major military escalation in the Middle East drives flight to hard assets amid fears of a broader regional war."},
        {"asset": "Defense", "ticker": "ITA", "direction": "up", "magnitude": "high", "confidence": 0.95, "reason": "Defense contractors rally sharply on expectations of sustained and expanded military operations, accelerated weapons procurement, and replenishment of munitions stockpiles."},
        {"asset": "US Treasury 10Y", "ticker": "TLT", "direction": "up", "magnitude": "medium", "confidence": 0.93, "reason": "Treasury prices rise as investors flee risk assets for the safety of US government bonds, driving yields lower in a classic flight-to-quality response."},
        {"asset": "S&P 500", "ticker": "SPY", "direction": "down", "magnitude": "high", "confidence": 0.92, "reason": "Broad equity selloff triggered by extreme geopolitical uncertainty, potential for war escalation, surging energy costs threatening corporate margins, and a massive risk-off shift."},
        {"asset": "Energy", "ticker": "XLE", "direction": "up", "magnitude": "high", "confidence": 0.91, "reason": "Energy sector stocks surge as oil and gas prices spike, dramatically boosting revenue and earnings expectations for producers and integrated majors."},
        {"asset": "Natural Gas", "ticker": "NG=F", "direction": "up", "magnitude": "medium", "confidence": 0.90, "reason": "Potential disruption of Middle Eastern LNG exports and energy supply chain contagion from the conflict zone lifts natural gas prices globally."},
        {"asset": "Airlines", "ticker": "JETS", "direction": "down", "magnitude": "high", "confidence": 0.90, "reason": "Airlines face a double blow from surging jet fuel costs and collapsed demand as Middle Eastern and adjacent airspace closes and travelers cancel plans."},
        {"asset": "NASDAQ", "ticker": "QQQ", "direction": "down", "magnitude": "high", "confidence": 0.89, "reason": "High-growth technology stocks sell off aggressively as risk appetite collapses and rising energy costs threaten margins across the tech supply chain."},
        {"asset": "USD", "ticker": "DXY", "direction": "up", "magnitude": "medium", "confidence": 0.88, "reason": "The US dollar strengthens as the world's primary reserve currency attracts safe-haven capital flows during a major geopolitical crisis."},
        {"asset": "JPY", "ticker": "JPY=X", "direction": "up", "magnitude": "medium", "confidence": 0.86, "reason": "The Japanese yen appreciates as a traditional safe-haven currency during periods of extreme geopolitical stress and global risk aversion."},
        {"asset": "Silver", "ticker": "SI=F", "direction": "up", "magnitude": "medium", "confidence": 0.84, "reason": "Silver benefits from safe-haven demand alongside gold, though its industrial demand component partially offsets gains due to recession fears."},
        {"asset": "Tourism", "ticker": None, "direction": "down", "magnitude": "medium", "confidence": 0.83, "reason": "Global tourism and hospitality stocks decline as international travel demand drops amid heightened security concerns and regional instability."},
        {"asset": "Bitcoin", "ticker": "BTC-USD", "direction": "down", "magnitude": "medium", "confidence": 0.72, "reason": "Bitcoin initially sells off as a risk-correlated asset in a broad liquidity crunch, despite some narrative support as a non-sovereign store of value."},
        {"asset": "Wheat", "ticker": "ZW=F", "direction": "up", "magnitude": "low", "confidence": 0.68, "reason": "Agricultural commodities see modest upside as Middle East instability raises concerns about trade route disruptions and food security in import-dependent regional economies."},
    ],
})


# ─── LLM call implementations ────────────────────────────────────────────

def _sync_llm_call(system_prompt: str, user_msg: str) -> str:
    import requests
    resp = requests.post(
        f"{settings.dashscope_base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.dashscope_api_key}",
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
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def llm_call(system_prompt: str, user_msg: str) -> str:
    return await asyncio.get_event_loop().run_in_executor(
        None, _sync_llm_call, system_prompt, user_msg,
    )


async def llm_call_offline(_system_prompt: str, _user_msg: str) -> str:
    return _SAMPLE_LLM_RESPONSE


# ─── Display helpers ──────────────────────────────────────────────────────

def _mini_sparkline(trajectory: list[float], base: float, width: int = 20) -> str:
    """Render a tiny ASCII sparkline of the price trajectory."""
    if not trajectory:
        return ""
    pcts = [(p - base) / base * 100 for p in trajectory]
    lo, hi = min(pcts), max(pcts)
    span = hi - lo if hi != lo else 1.0
    chars = " ▁▂▃▄▅▆▇█"
    # Sample `width` evenly spaced points
    step = max(1, len(pcts) // width)
    sampled = pcts[::step][:width]
    return "".join(
        chars[min(len(chars) - 1, int((v - lo) / span * (len(chars) - 1)))]
        for v in sampled
    )


def print_simulation_results(analysis: dict, sim_results: list[dict]) -> None:
    """Pretty-print the full analysis + per-asset simulation output."""

    print("=" * 90)
    print("  EVENT IMPACT ANALYSIS + PER-ASSET SIMULATION")
    print("=" * 90)
    print(f"\n  Event Category : {analysis['event_category']}")
    print(f"  Severity       : {analysis['severity']}/10")
    print(f"  Time Horizon   : {analysis['time_horizon']}")
    print(f"  Assets Impacted: {len(sim_results)}")
    print()

    # ── Per-asset results ────────────────────────────────────────────────
    print("-" * 90)
    print(f"  {'#':<3} {'Asset':<25} {'Dir':>4} {'Mag':>6} {'Conf':>5}"
          f"  {'Base':>10} {'Final':>10} {'Change':>8}  Trajectory")
    print("-" * 90)

    for i, sim in enumerate(sim_results, 1):
        arrow = "UP" if sim["direction"] == "up" else "DN"
        change_pct = sim["final_effect_pct"]
        sign = "+" if change_pct >= 0 else ""
        spark = _mini_sparkline(sim["price_trajectory"], sim["base_price"])

        print(
            f"  {i:<3} {sim['asset']:<25} {arrow:>4} {sim['magnitude']:>6} "
            f"{sim['confidence']:>5.2f}"
            f"  {sim['base_price']:>10,.2f} {sim['final_price']:>10,.2f} "
            f"{sign}{change_pct:>6.2f}%  {spark}"
        )

    # ── Detailed breakdown ───────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  DETAILED SIMULATION BREAKDOWN")
    print("=" * 90)

    for i, sim in enumerate(sim_results, 1):
        change_pct = sim["final_effect_pct"]
        peak_pct = sim["peak_effect_pct"]
        sign = "+" if change_pct >= 0 else ""
        peak_sign = "+" if peak_pct >= 0 else ""

        print(f"\n  [{i}] {sim['asset']} ({sim['ticker'] or 'N/A'})")
        print(f"      LLM Prediction : {sim['direction'].upper()} | "
              f"magnitude={sim['magnitude']} | confidence={sim['confidence']:.2f}")
        print(f"      Simulation     : {sim.get('n_agents', 16)} agents × "
              f"{sim.get('n_ticks', 100)} ticks (agent-driven LOB)")
        print(f"      Base Price     : {sim['base_price']:,.2f}")
        print(f"      Final Price    : {sim['final_price']:,.2f} ({sign}{change_pct:.2f}%)")
        print(f"      Peak Effect    : {peak_sign}{peak_pct:.2f}% at tick {sim['peak_tick']}")
        print(f"      Total Trades   : {sim.get('total_trades', 0)}")
        print(f"      Total Volume   : {sim.get('total_volume', 0)}")
        print(f"      Reason         : {sim['reason']}")

        # Show price trajectory at key ticks
        traj = sim["price_trajectory"]
        ticks_to_show = [0, 9, 24, 49, 74, 99, min(len(traj) - 1, 99)]
        ticks_to_show = sorted(set(t for t in ticks_to_show if t < len(traj)))
        tick_str = "      Trajectory     : " + " → ".join(
            f"t{t}:{traj[t]:,.2f}" for t in ticks_to_show
        )
        print(tick_str)


# ─── Main ─────────────────────────────────────────────────────────────────

async def main():
    offline = "--offline" in sys.argv

    print(f"\n  Model   : {settings.primary_model}")
    print(f"  Endpoint: {settings.dashscope_base_url}")
    if offline:
        print("  Mode    : OFFLINE (cached LLM response)")
    print(f"\n  Event:\n  {EVENT_TEXT}\n")

    # Step 1: LLM analysis
    print("  [1/2] Calling LLM for event analysis...")
    call_fn = llm_call_offline if offline else llm_call
    analysis = await analyze_event(EVENT_TEXT, call_fn, timeout=30)

    if analysis is None:
        print("  ERROR: LLM analysis failed (check API key / connectivity)")
        return

    print(f"  LLM returned {len(analysis['impacts'])} asset impacts.\n")

    # Step 2: Per-asset agent-driven LOB simulation
    print(f"  [2/2] Running agent-driven LOB simulation for each asset "
          f"(100 ticks x 16 agents each)...\n")

    sim_results = await simulate_all_impacts(
        analysis,
        event_headline=EVENT_TEXT,
        llm_call_fn=call_fn,
        n_ticks=100,
        n_agents=16,
    )

    # Display results
    print_simulation_results(analysis, sim_results)

    # Also dump raw JSON for programmatic use
    print("\n" + "=" * 90)
    print("  RAW JSON OUTPUT (for programmatic use)")
    print("=" * 90)
    output = {
        "event": analysis["event"],
        "event_category": analysis["event_category"],
        "severity": analysis["severity"],
        "time_horizon": analysis["time_horizon"],
        "simulation_results": [
            {
                "asset": s["asset"],
                "ticker": s["ticker"],
                "direction": s["direction"],
                "magnitude": s["magnitude"],
                "confidence": s["confidence"],
                "base_price": s["base_price"],
                "final_price": s["final_price"],
                "final_effect_pct": s["final_effect_pct"],
                "peak_effect_pct": s["peak_effect_pct"],
                "peak_tick": s["peak_tick"],
                "total_trades": s.get("total_trades", 0),
                "total_volume": s.get("total_volume", 0),
                "simulation_method": s.get("simulation_method", "agent_lob"),
            }
            for s in sim_results
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
