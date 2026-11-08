"""Bookmaker odds → expected goals (plan 06).

1. Remove the margin with Shin's method (more is taken off longshots than
   favourites, matching the favourite-longshot bias).
2. Find the (λ, μ) whose Dixon-Coles grid reproduces the market's home /
   draw / away and over-2.5 probabilities (like penaltyblog's goal_expectancy)."""

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq, minimize

from football_ml.config import MAX_GOALS, MARKET_RATE_START
from football_ml.model.dixon_coles import matrix_from_rates
from football_ml.sources.football_data import MatchOdds


def shin_probabilities(odds: Sequence[float]) -> tuple[float, ...]:
    """Margin-free probabilities from decimal odds (Shin 1993)."""
    if len(odds) < 2 or any(o <= 1.0 for o in odds):
        raise ValueError(f"Need at least two decimal odds above 1.0, got {tuple(odds)}")
    implied = np.array([1 / o for o in odds])
    booksum = implied.sum()
    if booksum <= 1.0 + 1e-12:
        return tuple(float(q) for q in implied / booksum)

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z**2 + 4 * (1 - z) * implied**2 / booksum) - z) / (2 * (1 - z))

    z = brentq(lambda z: probs(z).sum() - 1.0, 0.0, 0.999)
    return tuple(float(p) for p in probs(z))


def over_probability(matrix: np.ndarray, line: float = 2.5) -> float:
    i, j = np.indices(matrix.shape)
    return float(matrix[i + j > line].sum())


def _model_probs(log_rates: np.ndarray, rho: float) -> np.ndarray:
    m = matrix_from_rates(math.exp(log_rates[0]), math.exp(log_rates[1]), rho, MAX_GOALS)
    i, j = np.indices(m.shape)
    return np.array([m[i > j].sum(), m[i == j].sum(), m[i < j].sum(), over_probability(m)])


def market_rates(outcome: tuple[float, float, float], over: float | None, rho: float) -> tuple[float, float]:
    """(λ, μ) matching the 1X2 probabilities and, when given, P(over 2.5)."""
    target = np.array([*outcome, over if over is not None else 0.0])
    used = 4 if over is not None else 3

    def loss(x: np.ndarray) -> float:
        return float(np.sum((_model_probs(x, rho)[:used] - target[:used]) ** 2))

    start = np.log(MARKET_RATE_START)
    result = minimize(loss, start, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-15, "maxiter": 2000})
    if not result.success:
        raise RuntimeError(f"Could not fit market rates: {result.message}")
    return float(math.exp(result.x[0])), float(math.exp(result.x[1]))


def rates_from_odds(odds: MatchOdds, rho: float) -> tuple[float, float]:
    outcome = shin_probabilities((odds.home, odds.draw, odds.away))
    over = None
    if odds.over_2_5 is not None and odds.under_2_5 is not None:
        over = shin_probabilities((odds.over_2_5, odds.under_2_5))[0]
    return market_rates((outcome[0], outcome[1], outcome[2]), over, rho)
