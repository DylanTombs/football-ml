"""Turn a score probability matrix into a headline prediction."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scoreline:
    home_goals: int
    away_goals: int
    probability: float


@dataclass(frozen=True)
class ScorelineSummary:
    home_goals: int
    away_goals: int
    probability: float
    home_win: float
    draw: float
    away_win: float
    expected_home_goals: float
    expected_away_goals: float
    top_scorelines: tuple[Scoreline, ...]


def _outcome_masks(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    i, j = np.indices(shape)
    return {"home_win": i > j, "draw": i == j, "away_win": i < j}


def _headline(matrix: np.ndarray, outcomes: dict[str, float]) -> Scoreline:
    """Most likely scoreline *within* the most likely outcome, so a 56% away
    favourite is shown winning rather than defaulting to the modal 1-1."""
    best_outcome = max(outcomes, key=outcomes.get)
    masked = np.where(_outcome_masks(matrix.shape)[best_outcome], matrix, -1.0)
    i, j = np.unravel_index(int(np.argmax(masked)), matrix.shape)
    return Scoreline(int(i), int(j), float(matrix[i, j]))


def summarise(matrix: np.ndarray, top_n: int) -> ScorelineSummary:
    flat_order = np.argsort(matrix, axis=None)[::-1][:top_n]
    top = tuple(
        Scoreline(int(i), int(j), float(matrix[i, j]))
        for i, j in (np.unravel_index(k, matrix.shape) for k in flat_order)
    )
    outcomes = {name: float(matrix[mask].sum()) for name, mask in _outcome_masks(matrix.shape).items()}
    headline = _headline(matrix, outcomes)
    goals = np.arange(matrix.shape[0])
    return ScorelineSummary(
        home_goals=headline.home_goals,
        away_goals=headline.away_goals,
        probability=headline.probability,
        home_win=outcomes["home_win"],
        draw=outcomes["draw"],
        away_win=outcomes["away_win"],
        expected_home_goals=float(matrix.sum(axis=1) @ goals),
        expected_away_goals=float(matrix.sum(axis=0) @ goals),
        top_scorelines=top,
    )
