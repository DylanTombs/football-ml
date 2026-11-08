"""Probability scores for one match forecast (plan 06 bench)."""

import math

import numpy as np

_FLOOR = 1e-10
OUTCOMES = ("H", "D", "A")


def outcome_of(home_goals: int, away_goals: int) -> str:
    return "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"


def outcome_probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    i, j = np.indices(matrix.shape)
    return float(matrix[i > j].sum()), float(matrix[i == j].sum()), float(matrix[i < j].sum())


def exact_log_loss(matrix: np.ndarray, home_goals: int, away_goals: int) -> float:
    """−log P(actual score); scores beyond the grid get the floor."""
    rows, cols = matrix.shape
    p = matrix[home_goals, away_goals] if home_goals < rows and away_goals < cols else 0.0
    return -math.log(max(float(p), _FLOOR))


def outcome_log_loss(probs: tuple[float, float, float], outcome: str) -> float:
    return -math.log(max(probs[OUTCOMES.index(outcome)], _FLOOR))


def rps(probs: tuple[float, float, float], outcome: str) -> float:
    """Ranked probability score over the ordered outcomes H < D < A."""
    actual = tuple(1.0 if o == outcome else 0.0 for o in OUTCOMES)
    cum_p, cum_o = np.cumsum(probs)[:-1], np.cumsum(actual)[:-1]
    return float(np.sum((cum_p - cum_o) ** 2) / (len(OUTCOMES) - 1))
