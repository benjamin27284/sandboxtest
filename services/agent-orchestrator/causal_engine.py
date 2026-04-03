"""Structural Causal Model engine implementing Pearl's do-calculus
for the macroeconomic environment.

Features:
  - DAG-based SCM with structural equations per node
  - Pearl's do-operator: force a variable to a value, cutting parent edges
  - EGCIRF: Expected Generalized Counterfactual Impulse Response Function
    via paired Monte Carlo simulations (intervention vs baseline)
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class CausalNode:
    name: str
    equation: Callable[[dict, float], float]  # f(parents_dict, noise) -> float
    current_value: float = 0.0
    parents: list[str] = field(default_factory=list)


class StructuralCausalModel:
    """DAG-based SCM representing the macroeconomic environment.

    Nodes: interest_rate, inflation, gdp_growth, market_sentiment,
           asset_price, liquidity, volatility
    """

    # Topological order for the default DAG
    _TOPO_ORDER = [
        "interest_rate", "inflation", "gdp_growth",
        "market_sentiment", "liquidity", "volatility", "asset_price",
    ]

    def __init__(self) -> None:
        self.nodes: dict[str, CausalNode] = {}
        self._build_dag()
        self._intervention: dict[str, float] = {}

    def _build_dag(self) -> None:
        self.add_node(
            "interest_rate", parents=[],
            equation=lambda p, e: 0.05 + e,
        )
        self.add_node(
            "inflation", parents=["interest_rate"],
            equation=lambda p, e: 0.03 - 0.4 * p["interest_rate"] + e,
        )
        self.add_node(
            "gdp_growth", parents=["interest_rate", "inflation"],
            equation=lambda p, e: (
                0.025 - 0.3 * p["interest_rate"] + 0.1 * p["inflation"] + e
            ),
        )
        self.add_node(
            "market_sentiment", parents=["gdp_growth", "inflation"],
            equation=lambda p, e: (
                0.5 * p["gdp_growth"] - 0.3 * p["inflation"] + e
            ),
        )
        self.add_node(
            "liquidity", parents=["interest_rate", "market_sentiment"],
            equation=lambda p, e: (
                -0.5 * p["interest_rate"] + 0.4 * p["market_sentiment"] + e
            ),
        )
        self.add_node(
            "volatility", parents=["market_sentiment", "liquidity"],
            equation=lambda p, e: (
                0.15 - 0.2 * p["market_sentiment"]
                - 0.1 * p["liquidity"] + abs(e)
            ),
        )
        self.add_node(
            "asset_price",
            parents=["market_sentiment", "liquidity", "volatility"],
            equation=lambda p, e: (
                100 + 20 * p["market_sentiment"]
                + 10 * p["liquidity"] - 5 * p["volatility"] + e
            ),
        )

    def add_node(
        self,
        name: str,
        parents: list[str],
        equation: Callable[[dict, float], float],
    ) -> None:
        self.nodes[name] = CausalNode(
            name=name, equation=equation, parents=parents,
        )

    def do(self, variable: str, value: float) -> None:
        """Pearl's do-operator: forces variable to value, cutting parent edges."""
        self._intervention[variable] = value

    def undo(self, variable: str) -> None:
        self._intervention.pop(variable, None)

    def step(self, noise_scale: float = 0.01) -> dict[str, float]:
        """Evaluate all nodes in topological order.

        Intervened nodes take their fixed value regardless of parents.
        """
        values: dict[str, float] = {}

        for name in self._TOPO_ORDER:
            if name in self._intervention:
                values[name] = self._intervention[name]
            else:
                node = self.nodes[name]
                parents_vals = {p: values[p] for p in node.parents}
                noise = random.gauss(0, noise_scale)
                values[name] = node.equation(parents_vals, noise)
            self.nodes[name].current_value = values[name]

        return values

    def get_state(self) -> dict[str, float]:
        return {n: self.nodes[n].current_value for n in self.nodes}


def compute_egcirf(
    scm_factory: Callable[[], StructuralCausalModel],
    intervention_var: str,
    intervention_value: float,
    target_var: str,
    n_ticks: int = 30,
    n_runs: int = 50,
    noise_scale: float = 0.01,
) -> dict:
    """Expected Generalized Counterfactual Impulse Response Function.

    Runs n_runs paired simulations (with/without do-intervention),
    returns mean causal effect at each tick.
    """
    responses: list[list[float]] = []

    for _ in range(n_runs):
        # Baseline (no intervention)
        scm_base = scm_factory()
        baseline_series: list[float] = []
        for _ in range(n_ticks):
            state = scm_base.step(noise_scale)
            baseline_series.append(state[target_var])

        # Intervention: do(intervention_var = intervention_value)
        scm_int = scm_factory()
        scm_int.do(intervention_var, intervention_value)
        intervention_series: list[float] = []
        for _ in range(n_ticks):
            state = scm_int.step(noise_scale)
            intervention_series.append(state[target_var])

        # Causal effect = intervention - baseline at each tick
        effect = [
            i - b for i, b in zip(intervention_series, baseline_series)
        ]
        responses.append(effect)

    # Average across runs
    mean_egcirf = [
        statistics.mean(r[t] for r in responses) for t in range(n_ticks)
    ]
    std_egcirf = [
        statistics.stdev(r[t] for r in responses) for t in range(n_ticks)
    ]

    peak_effect = max(mean_egcirf, key=abs)

    return {
        "intervention": f"do({intervention_var}={intervention_value})",
        "target": target_var,
        "n_runs": n_runs,
        "mean_response": mean_egcirf,
        "std_response": std_egcirf,
        "peak_effect": peak_effect,
        "peak_tick": mean_egcirf.index(peak_effect),
    }
