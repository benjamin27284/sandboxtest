"""Guerini-Moneta SVAR validation protocol.

Compares the causal structure of simulated vs empirical time-series
by fitting VAR(1) models, extracting Granger-causal graphs, and
computing topological distance (precision / recall / F1) between
the edge sets.

References:
    Guerini & Moneta (2017), "A method for agent-based models validation"
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


# ─── Data container ──────────────────────────────────────────────────────────

@dataclass
class TimeSeriesData:
    prices: list[float]
    volumes: list[float]
    volatilities: list[float]  # realized vol per tick


# ─── Feature extraction ─────────────────────────────────────────────────────

def compute_returns(prices: list[float]) -> list[float]:
    """Log returns.  Clamps non-positive prices to previous value to
    avoid log(0) and preserve index alignment with the price array."""
    returns: list[float] = []
    for i in range(1, len(prices)):
        prev = prices[i - 1] if prices[i - 1] > 0 else prices[i]
        curr = prices[i] if prices[i] > 0 else prev
        if prev > 0 and curr > 0:
            returns.append(math.log(curr / prev))
        else:
            returns.append(0.0)
    return returns


def compute_realized_vol(returns: list[float], window: int = 20) -> list[float]:
    """Rolling realized volatility (sample std dev over *window* returns)."""
    vols: list[float] = []
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        w = returns[start : i + 1]
        vols.append(statistics.stdev(w) if len(w) > 1 else 0.0)
    return vols


# ─── Linear algebra helpers (pure-python, no mandatory numpy) ────────────────

def _dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _transpose(M: list[list[float]]) -> list[list[float]]:
    return [[M[r][c] for r in range(len(M))] for c in range(len(M[0]))]


def _mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    BT = _transpose(B)
    return [[_dot(row, col) for col in BT] for row in A]


def _mat_vec(M: list[list[float]], v: list[float]) -> list[float]:
    return [_dot(row, v) for row in M]


def _pinv(M: list[list[float]]) -> list[list[float]]:
    """Moore-Penrose pseudoinverse.  Prefers numpy; falls back to an
    analytical 2×2 inverse, or identity for larger matrices."""
    try:
        import numpy as np
        return np.linalg.pinv(np.array(M)).tolist()
    except ImportError:
        n = len(M)
        if n == 2:
            a, b = M[0][0], M[0][1]
            c, d = M[1][0], M[1][1]
            det = a * d - b * c
            if abs(det) < 1e-12:
                return [[0.0, 0.0], [0.0, 0.0]]
            return [[d / det, -b / det], [-c / det, a / det]]
        # Last resort: identity (makes coefficient estimates degenerate
        # but avoids a hard crash when numpy is unavailable).
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


# ─── VAR(1) fitting ─────────────────────────────────────────────────────────

def fit_var_lag1(y: list[list[float]]) -> list[list[float]]:
    """Fit VAR(1): Y_t = A · Y_{t-1} + ε via OLS.

    Args:
        y: T observations, each a list of K variables.

    Returns:
        K×K coefficient matrix A where A[i][j] captures the effect of
        variable j at t-1 on variable i at t.
    """
    T = len(y)
    K = len(y[0])

    if T < K + 2:
        return [[0.0] * K for _ in range(K)]

    # Y is the dependent matrix (t=1..T-1), X is the lagged matrix (t=0..T-2)
    Y = [y[t] for t in range(1, T)]
    X = [y[t] for t in range(0, T - 1)]

    # OLS per equation: β_k = (X'X)⁻¹ X' Y_k
    XT = _transpose(X)
    XTX = _mat_mul(XT, X)
    XTX_inv = _pinv(XTX)

    A: list[list[float]] = []
    for k in range(K):
        Yk = [Y[t][k] for t in range(len(Y))]
        XTY_k = [_dot(XT[j], Yk) for j in range(K)]
        a_k = _mat_vec(XTX_inv, XTY_k)
        A.append(a_k)

    return A


# ─── Causal graph extraction ────────────────────────────────────────────────

def extract_causal_graph(
    A: list[list[float]], threshold: float = 0.05
) -> list[tuple[int, int]]:
    """Extract significant directed edges from a VAR coefficient matrix.

    Edge (j → i) exists when |A[i][j]| > threshold, meaning variable j
    at lag 1 Granger-causes variable i.
    """
    edges: list[tuple[int, int]] = []
    for i in range(len(A)):
        for j in range(len(A[i])):
            if i != j and abs(A[i][j]) > threshold:
                edges.append((j, i))
    return edges


# ─── Topological distance ───────────────────────────────────────────────────

def topological_distance(
    edges_sim: list[tuple[int, int]],
    edges_emp: list[tuple[int, int]],
) -> dict:
    """Guerini-Moneta topological comparison.

    Computes precision, recall, and F1 of edge-set similarity between
    the simulated and empirical causal graphs.
    """
    set_sim = set(edges_sim)
    set_emp = set(edges_emp)

    tp = len(set_sim & set_emp)
    fp = len(set_sim - set_emp)
    fn = len(set_emp - set_sim)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shared_edges": sorted(set_sim & set_emp),
        "missing_edges": sorted(set_emp - set_sim),
        "spurious_edges": sorted(set_sim - set_emp),
    }


# ─── Full validation protocol ───────────────────────────────────────────────

def run_svar_validation(
    simulated: TimeSeriesData,
    empirical: TimeSeriesData,
    threshold: float = 0.05,
) -> dict:
    """Full Guerini-Moneta protocol.

    1. Compute log returns and rolling realized vol for both datasets.
    2. Fit VAR(1) on [returns, vol] for each.
    3. Extract Granger-causal graphs.
    4. Compute topological distance (precision / recall / F1).
    """
    sim_ret = compute_returns(simulated.prices)
    sim_vol = compute_realized_vol(sim_ret)
    n = min(len(sim_ret), len(sim_vol))
    Y_sim = [[sim_ret[i], sim_vol[i]] for i in range(n)]

    emp_ret = compute_returns(empirical.prices)
    emp_vol = compute_realized_vol(emp_ret)
    m = min(len(emp_ret), len(emp_vol))
    Y_emp = [[emp_ret[i], emp_vol[i]] for i in range(m)]

    A_sim = fit_var_lag1(Y_sim)
    A_emp = fit_var_lag1(Y_emp)

    edges_sim = extract_causal_graph(A_sim, threshold)
    edges_emp = extract_causal_graph(A_emp, threshold)

    result = topological_distance(edges_sim, edges_emp)
    result["var_matrix_simulated"] = A_sim
    result["var_matrix_empirical"] = A_emp

    return result


def load_empirical_data(
    ticker: str = "SPY.US",
    api_key: str = "",
    period: str = "d",
    n_bars: int = 252,
) -> TimeSeriesData:
    """Load real-world OHLCV data from EODHD and return as TimeSeriesData.
    Uses the EODHD End-of-Day Historical Data API.
    Falls back to a synthetic GBM series if the request fails.
    Args:
        ticker:  EODHD ticker symbol (e.g. "SPY.US", "AAPL.US", "GC.COMM" for Gold).
        api_key: EODHD API key. If empty, reads from EODHD_API_KEY env var / settings.
        period:  "d" = daily, "w" = weekly, "m" = monthly.
        n_bars:  Number of bars to return (most recent).
    """
    import os as _os
    import urllib.request
    import json as _json
    import math as _math
    import random as _random
    if not api_key:
        try:
            from config.settings import settings as _settings
            api_key = _settings.eodhd_api_key
        except Exception:
            api_key = _os.getenv("EODHD_API_KEY", "")
    url = (
        f"https://eodhd.com/api/eod/{ticker}"
        f"?api_token={api_key}&fmt=json&order=a&period={period}"
    )
    prices: list[float] = []
    volumes: list[float] = []
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = _json.loads(resp.read().decode())
        # data is a list of dicts: {date, open, high, low, close, adjusted_close, volume}
        for bar in data[-n_bars:]:
            close = float(bar.get("adjusted_close") or bar.get("close") or 0)
            vol   = float(bar.get("volume") or 0)
            if close > 0:
                prices.append(close)
                volumes.append(vol)
        if len(prices) < 10:
            raise ValueError(f"Insufficient data returned for {ticker}: {len(prices)} bars")
    except Exception as exc:
        # Fallback: generate a realistic GBM price series as stand-in
        import warnings
        warnings.warn(
            f"EODHD fetch failed for {ticker} ({exc}). "
            f"Falling back to synthetic GBM series.",
            RuntimeWarning,
            stacklevel=2,
        )
        rng = _random.Random(42)
        prices = [100.0]
        for _ in range(n_bars - 1):
            r = rng.gauss(0.0003, 0.015)
            prices.append(round(prices[-1] * _math.exp(r), 4))
        volumes = [float(rng.randint(1_000_000, 10_000_000)) for _ in range(n_bars)]
    ret = compute_returns(prices)
    vols = compute_realized_vol(ret)
    return TimeSeriesData(prices=prices, volumes=volumes, volatilities=vols)
