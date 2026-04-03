"""Deterministic Math Shield — keeps all quantitative computations outside the LLM.

The LLM never computes Bayesian updates, DCFs, or portfolio optimizations.
This module does the math and passes numerical conclusions to the LLM so
it can form strategic narratives around hard numbers.

Implements:
  - Bayesian belief update (posterior μ given prior + signal)
  - Full Black-Litterman portfolio optimization (Theil's mixed estimation)
  - Single-asset Black-Litterman wrapper for LLM agents
  - Value-at-Risk (historical / parametric)
  - Simple DCF fair-value estimate
  - Simulated financial data terminal (WACC inputs, per-ticker)
  - Cumulative Prospect Theory (Kahneman-Tversky value function)
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class BayesianBelief:
    """Agent's belief about the asset's true value."""
    mu: float         # mean of the belief distribution
    sigma: float      # uncertainty (std dev)

    def to_prompt_string(self) -> str:
        return (
            f"Your current belief: fair value = ${self.mu:.2f} "
            f"(uncertainty ±${self.sigma:.2f}, "
            f"95% CI: ${self.mu - 1.96*self.sigma:.2f}–"
            f"${self.mu + 1.96*self.sigma:.2f})"
        )


@dataclass
class QuantOutput:
    """Packaged numerical conclusions for the LLM to interpret."""
    posterior_belief: BayesianBelief
    optimal_position: float     # target inventory (fractional)
    value_at_risk: float        # $ amount at 95% confidence
    dcf_fair_value: Optional[float] = None
    signal_strength: float = 0.0   # -1.0 (strong sell) to +1.0 (strong buy)

    def to_prompt_string(self) -> str:
        lines = [
            "── Quantitative Analysis (computed, not estimated) ──",
            self.posterior_belief.to_prompt_string(),
            f"Optimal target position: {self.optimal_position:+.1f} units",
            f"Value-at-Risk (95%): ${self.value_at_risk:,.2f}",
            f"Signal strength: {self.signal_strength:+.2f} "
            f"({'bullish' if self.signal_strength > 0.1 else 'bearish' if self.signal_strength < -0.1 else 'neutral'})",
        ]
        if self.dcf_fair_value is not None:
            lines.append(f"DCF fair value estimate: ${self.dcf_fair_value:.2f}")
        return "\n".join(lines)


class QuantEngine:
    """Stateless computation engine — all methods are pure functions."""

    @staticmethod
    def bayesian_update(
        prior_mu: float,
        prior_sigma: float,
        signal: float,
        signal_sigma: float,
        agent_sophistication: float = 1.0,
    ) -> BayesianBelief:
        """Conjugate Gaussian update: posterior given prior + noisy signal.

        agent_sophistication: 1.0 = fully rational (uses true signal_sigma)
                              > 1.0 = retail (overweights signal, reduces effective σ)
                              < 1.0 = overcautious institutional
        """
        effective_signal_sigma = signal_sigma / agent_sophistication
        tau_prior = 1.0 / (prior_sigma ** 2)
        tau_signal = 1.0 / (effective_signal_sigma ** 2)
        posterior_mu = (tau_prior * prior_mu + tau_signal * signal) / (tau_prior + tau_signal)
        posterior_sigma = (1.0 / (tau_prior + tau_signal)) ** 0.5

        return BayesianBelief(mu=posterior_mu, sigma=posterior_sigma)

    @staticmethod
    def black_litterman_position(
        belief: BayesianBelief,
        current_price: float,
        risk_aversion: float = 2.5,
        asset_variance: float = 0.04,
    ) -> float:
        """Simplified single-asset Black-Litterman optimal position.

        w* = (1 / (λ · σ²)) · (μ_post - current_price) / current_price

        Returns target position in units (can be negative for short).
        """
        if current_price <= 0 or asset_variance <= 0:
            return 0.0

        excess_return = (belief.mu - current_price) / current_price
        optimal_weight = excess_return / (risk_aversion * asset_variance)

        # Scale to a reasonable position size (e.g., max ±50 units)
        return max(-50.0, min(50.0, optimal_weight * 100))

    @staticmethod
    def parametric_var(
        position: int,
        current_price: float,
        daily_volatility: float = 0.02,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """Parametric Value-at-Risk.

        VaR = |position| × price × z × σ × √T
        """
        # z-score for confidence level
        z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
        z = z_scores.get(confidence, 1.645)

        exposure = abs(position) * current_price
        return exposure * z * daily_volatility * math.sqrt(horizon_days)

    @staticmethod
    def dcf_fair_value(
        current_earnings: float,
        growth_rate: float = 0.05,
        discount_rate: float = 0.10,
        terminal_growth: float = 0.02,
        years: int = 5,
    ) -> Optional[float]:
        """Simple multi-stage DCF → fair value per share.

        Returns None if discount_rate <= terminal_growth (undefined).
        """
        if discount_rate <= terminal_growth:
            return None

        pv_fcf = 0.0
        fcf = current_earnings
        for t in range(1, years + 1):
            fcf *= (1 + growth_rate)
            pv_fcf += fcf / ((1 + discount_rate) ** t)

        # Terminal value (Gordon Growth Model)
        terminal_fcf = fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / ((1 + discount_rate) ** years)

        return pv_fcf + pv_terminal

    @staticmethod
    def black_litterman(
        market_weights: list[float],
        sigma: list[list[float]],
        pi: list[float],
        P: list[list[float]],
        Q: list[float],
        omega: list[list[float]],
        tau: float = 0.05,
        risk_aversion: float = 3.0,
    ) -> dict:
        """Full Black-Litterman posterior using Theil's mixed estimation.

        Returns posterior mean vector mu_bl, posterior covariance sigma_bl,
        and optimal portfolio weights.
        """
        import numpy as np

        sigma_m = np.array(sigma)
        pi_v = np.array(pi)
        P_m = np.array(P)
        Q_v = np.array(Q)
        omega_m = np.array(omega)

        tau_sigma_inv = np.linalg.inv(tau * sigma_m)
        omega_inv = np.linalg.inv(omega_m)

        M = tau_sigma_inv + P_m.T @ omega_inv @ P_m
        M_inv = np.linalg.inv(M)

        mu_bl = M_inv @ (tau_sigma_inv @ pi_v + P_m.T @ omega_inv @ Q_v)
        sigma_bl = sigma_m + M_inv

        w_star = np.linalg.inv(risk_aversion * sigma_bl) @ mu_bl
        w_star = w_star / np.sum(np.abs(w_star))  # normalize

        return {
            "mu_bl": mu_bl.tolist(),
            "sigma_bl": sigma_bl.tolist(),
            "optimal_weights": w_star.tolist(),
        }

    @classmethod
    def black_litterman_single_asset(
        cls,
        belief_mu: float,
        belief_sigma: float,
        current_price: float,
        market_return: float = 0.07,
        tau: float = 0.05,
        risk_aversion: float = 3.0,
        confidence: float = 0.5,
    ) -> int:
        """Wraps full B-L for single-asset case. Returns integer position in units."""
        import numpy as np

        sigma_sq = belief_sigma ** 2
        pi = [market_return]
        P = [[1.0]]
        Q = [(belief_mu / max(current_price, 0.01)) - 1.0]
        omega = [[(1 - confidence) / max(confidence, 1e-6) * tau * sigma_sq]]

        result = cls.black_litterman(
            [1.0], [[sigma_sq]], pi, P, Q, omega,
            tau=tau, risk_aversion=risk_aversion,
        )
        weight = result["optimal_weights"][0]
        position_value = weight * 100_000  # assume 100k portfolio
        return int(position_value / max(current_price, 0.01))

    @classmethod
    def compute_full(
        cls,
        prior_mu: float,
        prior_sigma: float,
        signal: float,
        signal_sigma: float,
        current_price: float,
        current_position: int,
        daily_volatility: float = 0.02,
    ) -> QuantOutput:
        """Run the full quant pipeline and return packaged results."""
        belief = cls.bayesian_update(prior_mu, prior_sigma, signal, signal_sigma)
        optimal_pos = cls.black_litterman_position(belief, current_price)
        var = cls.parametric_var(current_position, current_price, daily_volatility)

        # Signal strength: normalized divergence of belief from price
        if current_price > 0 and belief.sigma > 0:
            z = (belief.mu - current_price) / belief.sigma
            signal_strength = max(-1.0, min(1.0, z / 3.0))  # scale ±3σ → ±1.0
        else:
            signal_strength = 0.0

        return QuantOutput(
            posterior_belief=belief,
            optimal_position=optimal_pos,
            value_at_risk=var,
            signal_strength=signal_strength,
        )


# ─── Cumulative Prospect Theory ──────────────────────────────────────────────

@dataclass
class CPTState:
    """Kahneman-Tversky prospect theory state for an agent."""
    reference_price: float      # agent's entry price / recent anchor
    lambda_loss: float = 2.25   # loss aversion coefficient
    alpha: float = 0.88         # value function curvature


def cpt_value(gain_or_loss: float, cpt: CPTState) -> float:
    """Kahneman-Tversky value function.

    Returns subjective utility of a gain or loss relative to reference point.
    """
    x = gain_or_loss
    if x >= 0:
        return x ** cpt.alpha
    else:
        return -cpt.lambda_loss * ((-x) ** cpt.alpha)


def cpt_signal(current_price: float, cpt: CPTState) -> float:
    """Returns a [-1, 1] signal.

    Negative = loss aversion pressure to sell.
    Positive = gain domain, willing to hold/buy.
    """
    pnl = current_price - cpt.reference_price
    v = cpt_value(pnl, cpt)
    return max(-1.0, min(1.0, v / (cpt.reference_price + 1e-6)))


def update_reference_price(
    cpt: CPTState, current_price: float, alpha_decay: float = 0.05
) -> CPTState:
    """Reference point drifts toward current price over time (adaptation)."""
    new_ref = (1 - alpha_decay) * cpt.reference_price + alpha_decay * current_price
    return CPTState(new_ref, cpt.lambda_loss, cpt.alpha)


# ─── Simulated Financial Data Terminal ────────────────────────────────────────

class SimulatedDataTerminal:
    """Simulates a financial data terminal.

    Values are seeded per ticker so each agent calling the same ticker gets
    consistent inputs, but different tickers differ. Inputs drift slowly
    each tick (stable per 10-tick window).
    """

    def __init__(self) -> None:
        self._cache: dict = {}

    def get_wacc_inputs(self, ticker: str, tick: int) -> dict:
        rng = _random.Random(hash(ticker) + tick // 10)
        return {
            "risk_free_rate": 0.04 + rng.uniform(-0.005, 0.005),
            "beta": 0.8 + rng.uniform(0, 1.2),
            "equity_risk_premium": 0.05 + rng.uniform(-0.01, 0.01),
            "growth_rate": 0.03 + rng.uniform(-0.02, 0.04),
            "terminal_growth": 0.025,
            "current_earnings": 5.0 + rng.uniform(-2, 5),
            "comparable_ev_ebitda": 12 + rng.uniform(-3, 6),
        }

    def dcf_from_terminal(self, ticker: str, tick: int) -> float:
        inputs = self.get_wacc_inputs(ticker, tick)
        wacc = (inputs["risk_free_rate"]
                + inputs["beta"] * inputs["equity_risk_premium"])
        result = QuantEngine.dcf_fair_value(
            current_earnings=inputs["current_earnings"],
            growth_rate=inputs["growth_rate"],
            discount_rate=wacc,
            terminal_growth=inputs["terminal_growth"],
            years=5,
        )
        return result if result is not None else 100.0


# Module-level singleton
data_terminal = SimulatedDataTerminal()
