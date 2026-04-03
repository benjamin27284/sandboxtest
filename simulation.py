"""SimulationEnvironment — orchestrates the multi-agent LOB simulation.

Initialises the LOB, all agent populations, runs discrete ticks with
agent-order shuffling, event injection, and plots the price time-series.

Usage:
    python simulation.py              # 50-tick default
    python simulation.py --ticks 100  # custom tick count
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import random
from typing import Optional

from lob import LimitOrderBook, Order, Side
from agents import BaseAgent, NoiseTrader, MomentumTrader
from llm_agent import LLMFundamentalAgent, LLMDecision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# News event schedule
# ---------------------------------------------------------------------------

NEWS_EVENTS: dict[int, str] = {
    25: (
        "URGENT: Central bank unexpectedly raises interest rates by "
        "150 basis points, citing severe inflation risks."
    ),
}
DEFAULT_NEWS = "No major news."

# ---------------------------------------------------------------------------
# SimulationEnvironment
# ---------------------------------------------------------------------------


class SimulationEnvironment:
    """Orchestrates a discrete-tick multi-agent LOB simulation."""

    def __init__(
        self,
        n_noise: int = 10,
        n_momentum: int = 3,
        n_llm: int = 2,
        seed_price: float = 100.0,
        seed_depth: int = 100,
        random_seed: int = 42,
    ) -> None:
        random.seed(random_seed)

        # --- LOB -------------------------------------------------------------
        self.lob = LimitOrderBook()
        self.lob.submit(
            Order("SEED", Side.BUY, price=seed_price - 1.0,
                  quantity=seed_depth, tick_submitted=0)
        )
        self.lob.submit(
            Order("SEED", Side.SELL, price=seed_price + 1.0,
                  quantity=seed_depth, tick_submitted=0)
        )

        # --- Agents ----------------------------------------------------------
        self.noise_traders = [
            NoiseTrader(agent_id=f"N{i:02d}") for i in range(n_noise)
        ]
        self.momentum_traders = [
            MomentumTrader(agent_id=f"M{i:02d}") for i in range(n_momentum)
        ]
        self.llm_agents = [
            LLMFundamentalAgent(agent_id=f"LLM{i:02d}") for i in range(n_llm)
        ]

        self.all_agents: list[BaseAgent] = (
            self.noise_traders  # type: ignore[assignment]
            + self.momentum_traders
            + self.llm_agents
        )
        self.agents_by_id = {a.agent_id: a for a in self.all_agents}

        # --- Recording -------------------------------------------------------
        self.price_series: list[Optional[float]] = []
        self.mid_series: list[Optional[float]] = []
        self.volume_series: list[int] = []
        self.last_price: Optional[float] = None

    # --------------------------------------------------------------------- #
    #  Main simulation loop                                                   #
    # --------------------------------------------------------------------- #

    async def run_simulation(self, total_ticks: int = 50) -> None:
        """Run the simulation for *total_ticks* discrete ticks."""

        header = (
            f"{'tick':>4}  {'mid':>8}  {'bid':>8}  {'ask':>8}  "
            f"{'fills':>5}  {'last_px':>8}  {'news':>6}"
        )
        print(header)
        print("-" * len(header))

        for tick in range(1, total_ticks + 1):
            news = NEWS_EVENTS.get(tick, DEFAULT_NEWS)
            is_event = tick in NEWS_EVENTS

            # 1. Shuffle agent ordering each tick
            shuffled = list(self.all_agents)
            random.shuffle(shuffled)

            # 2. Each agent observes the market and submits orders
            for agent in shuffled:
                if isinstance(agent, LLMFundamentalAgent):
                    await agent.act_async(
                        tick, self.lob, self.last_price, news
                    )
                else:
                    agent.act(tick, self.lob, self.last_price)

            # 3. Match crossing orders
            executions = self.lob.match_orders()

            # 4. Settle fills
            tick_volume = 0
            for ex in executions:
                price, qty = ex["price"], ex["quantity"]
                buyer = self.agents_by_id.get(ex["buyer_id"])
                seller = self.agents_by_id.get(ex["seller_id"])
                if buyer:
                    buyer.update_on_fill(price, qty, Side.BUY)
                if seller:
                    seller.update_on_fill(price, qty, Side.SELL)
                # Feed execution prices to momentum traders
                for mt in self.momentum_traders:
                    mt.observe_price(price)
                self.last_price = price
                tick_volume += qty

            # 5. Record closing price (last execution price or mid)
            closing = self.last_price if self.last_price is not None else self.lob.get_mid_price()
            self.price_series.append(closing)
            self.mid_series.append(self.lob.get_mid_price())
            self.volume_series.append(tick_volume)

            # 6. Print tick summary
            mid = self.lob.get_mid_price()
            bb = self.lob.best_bid()
            ba = self.lob.best_ask()
            fmt = lambda v: f"{v:.2f}" if v is not None else "N/A"
            event_flag = " <<EVENT" if is_event else ""

            print(
                f"{tick:4d}  {fmt(mid):>8}  {fmt(bb):>8}  {fmt(ba):>8}  "
                f"{len(executions):5d}  {fmt(self.last_price):>8}"
                f"{event_flag}"
            )

        # --- End-of-simulation summary ----------------------------------------
        self._print_summary(total_ticks)

    # --------------------------------------------------------------------- #
    #  Summary & plotting                                                     #
    # --------------------------------------------------------------------- #

    def _print_summary(self, total_ticks: int) -> None:
        total_vol = sum(self.volume_series)
        mark = self.last_price if self.last_price is not None else 100.0
        print(f"\n{'='*60}")
        print(f"Simulation complete: {total_ticks} ticks, "
              f"{total_vol} units traded")
        print(f"{'='*60}")
        print(f"\n  {'ID':<8} {'Type':<12} {'Cash':>12} {'Inv':>6} {'MTM P&L':>12}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*6} {'-'*12}")
        for agent in self.all_agents:
            init_cash = 100_000.0 if isinstance(agent, LLMFundamentalAgent) else 10_000.0
            mtm = agent.cash + agent.inventory * mark - init_cash
            cls = agent.__class__.__name__
            print(
                f"  {agent.agent_id:<8} {cls:<12} "
                f"{agent.cash:12.2f} {agent.inventory:6d} {mtm:12.2f}"
            )

        # Check if LLM agents were active
        llm_inactive = all(
            a.inventory == 0 and a.cash == 100_000.0
            for a in self.llm_agents
        )
        if llm_inactive:
            print("\n  NOTE: LLM agents defaulted to HOLD every tick "
                  "(DashScope API unreachable).")
            print("        With internet access, they will generate "
                  "real buy/sell signals.")

    def plot_price_series(self, save_path: str = "price_series.png") -> None:
        """Plot price time-series with event annotation.

        Uses matplotlib if available; otherwise falls back to an ASCII chart.
        """
        prices = [p if p is not None else float("nan") for p in self.price_series]
        ticks = list(range(1, len(prices) + 1))

        try:
            import matplotlib
            matplotlib.use("Agg")  # headless backend for sandbox
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(14, 8), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )

            # -- Price panel --------------------------------------------------
            ax1.plot(ticks, prices, linewidth=1.5, color="#2563eb", label="Last price")

            mids = [m if m is not None else float("nan") for m in self.mid_series]
            ax1.plot(ticks, mids, linewidth=1.0, color="#94a3b8",
                     linestyle="--", alpha=0.7, label="Mid price")

            for event_tick, event_text in NEWS_EVENTS.items():
                if event_tick <= len(prices):
                    ax1.axvline(x=event_tick, color="#dc2626", linestyle=":",
                                linewidth=1.5, alpha=0.8)
                    ax1.annotate(
                        event_text[:50] + "…",
                        xy=(event_tick, prices[event_tick - 1]),
                        xytext=(event_tick + 2, max(p for p in prices if p == p) - 0.5),
                        fontsize=7,
                        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1),
                        color="#dc2626",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#dc2626"),
                    )

            ax1.set_ylabel("Price")
            ax1.set_title("Multi-Agent LOB Simulation — Price Time-Series")
            ax1.legend(loc="upper left", fontsize=8)
            ax1.grid(True, alpha=0.3)

            # -- Volume panel -------------------------------------------------
            ax2.bar(ticks, self.volume_series, width=0.8, color="#60a5fa", alpha=0.7)
            ax2.set_ylabel("Volume")
            ax2.set_xlabel("Tick")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"\nChart saved to {save_path}")

        except ImportError:
            print("\nmatplotlib not available — falling back to ASCII chart.\n")
            self._ascii_chart(ticks, prices)

    @staticmethod
    def _ascii_chart(
        ticks: list[int],
        prices: list[float],
        width: int = 70,
        height: int = 22,
    ) -> None:
        """Render a minimal ASCII price chart to stdout."""
        clean = [p for p in prices if p == p]  # drop NaN
        if not clean:
            print("  (no price data to plot)")
            return

        lo, hi = min(clean), max(clean)
        spread = hi - lo if hi != lo else 1.0

        # Build character grid
        grid = [[" "] * width for _ in range(height)]

        for i, price in enumerate(prices):
            if price != price:  # NaN
                continue
            col = int(i / max(len(prices) - 1, 1) * (width - 1))
            row = height - 1 - int((price - lo) / spread * (height - 1))
            row = max(0, min(height - 1, row))
            grid[row][col] = "*"

        # Mark event ticks
        for event_tick in NEWS_EVENTS:
            if event_tick <= len(prices):
                col = int((event_tick - 1) / max(len(prices) - 1, 1) * (width - 1))
                for r in range(height):
                    if grid[r][col] == " ":
                        grid[r][col] = "|"

        # Render
        print(f"  Price  {'─' * width}")
        for r in range(height):
            if r == 0:
                label = f"{hi:7.2f}"
            elif r == height - 1:
                label = f"{lo:7.2f}"
            elif r == height // 2:
                label = f"{(hi + lo) / 2:7.2f}"
            else:
                label = "       "
            print(f"  {label} │{''.join(grid[r])}│")

        print(f"         {'─' * width}")
        # X-axis labels
        print(f"         1{' ' * (width - 2 - len(str(len(prices))))}{len(prices)}")
        print(f"         {'Tick →':^{width}}")

        # Legend
        print(f"\n  * = price   | = news event")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent LOB simulation")
    parser.add_argument("--ticks", type=int, default=50, help="Number of ticks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-o", "--output", default="price_series.png",
                        help="Output chart filename")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    # Suppress repetitive LLM-agent API warnings (one summary printed instead)
    logging.getLogger("llm_agent").setLevel(logging.ERROR)

    sim = SimulationEnvironment(random_seed=args.seed)
    await sim.run_simulation(total_ticks=args.ticks)
    sim.plot_price_series(save_path=args.output)


if __name__ == "__main__":
    asyncio.run(main())
