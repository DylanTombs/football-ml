"""Expected-points picks: the scoreline and scorers worth the most points under our rules."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

from football_ml.config import (
    MINUTE_HIT_RATE,
    POINTS_CORRECT_RESULT,
    POINTS_EXACT_SCORE,
    POINTS_PER_MINUTE,
    POINTS_PER_SCORER,
)
from football_ml.predict.scorers import Player, scoring_weight


@dataclass(frozen=True)
class Pick:
    home_goals: int
    away_goals: int
    probability: float
    expected_points: float
    is_fallback: bool


@dataclass(frozen=True)
class ScorerPick:
    player: Player
    probability: float  # chance this pick is matched by a real goal


def _result_mask(shape: tuple[int, int], home_goals: int, away_goals: int) -> np.ndarray:
    i, j = np.indices(shape)
    return np.sign(i - j) == np.sign(home_goals - away_goals)


def outcome_expected_points(matrix: np.ndarray, home_goals: int, away_goals: int) -> float:
    """P(result) × result points + P(exact) × the extra for an exact score."""
    p_result = float(matrix[_result_mask(matrix.shape, home_goals, away_goals)].sum())
    p_exact = float(matrix[home_goals, away_goals])
    return POINTS_CORRECT_RESULT * p_result + (POINTS_EXACT_SCORE - POINTS_CORRECT_RESULT) * p_exact


def best_scoreline(
    matrix: np.ndarray,
    home_scorer_points: Sequence[float],
    away_scorer_points: Sequence[float],
    fallback: tuple[int, int],
    floor: float,
) -> Pick:
    """Search every scoreline with probability ≥ `floor`. `*_scorer_points[n]` is
    the expected scorer + minute points of predicting n goals for that side.
    Keeps `fallback` unless something scores strictly more."""
    rows, cols = matrix.shape
    if len(home_scorer_points) < rows or len(away_scorer_points) < cols:
        raise ValueError(f"Need scorer points for 0..{rows - 1} home and 0..{cols - 1} away goals")

    def pick(a: int, b: int, is_fallback: bool) -> Pick:
        value = outcome_expected_points(matrix, a, b) + home_scorer_points[a] + away_scorer_points[b]
        return Pick(a, b, float(matrix[a, b]), value, is_fallback)

    best = pick(*fallback, is_fallback=True)
    for a, b in zip(*np.nonzero(matrix >= floor)):
        candidate = pick(int(a), int(b), is_fallback=False)
        if candidate.expected_points > best.expected_points + 1e-12:
            best = candidate
    return best


def pick_scorers(squad: Sequence[Player], team_rate: float, n: int) -> tuple[ScorerPick, ...]:
    """n picks, greedily by marginal value. A player's goals are Poisson with
    mean (share of eligible scoring weight) × team_rate, so their k-th pick is
    matched with probability P(X ≥ k). The objective is separable and concave,
    so greedy is optimal; ties go to the higher-rated player."""
    eligible = [p for p in squad if p.available and p.minutes > 0 and scoring_weight(p) > 0]
    total = sum(scoring_weight(p) for p in eligible)
    if not eligible or n <= 0 or team_rate <= 0:
        return ()
    rates = {p.id: scoring_weight(p) / total * team_rate for p in eligible}

    picks: tuple[ScorerPick, ...] = ()
    for _ in range(n):
        given = {p.id: sum(s.player.id == p.id for s in picks) for p in eligible}
        chance = {p.id: float(poisson.sf(given[p.id], rates[p.id])) for p in eligible}  # P(X ≥ given+1)
        best = max(eligible, key=lambda p: (chance[p.id], rates[p.id], -p.id))
        picks = (*picks, ScorerPick(best, chance[best.id]))
    return picks


def cumulative_scorer_points(picks: Sequence[ScorerPick], max_goals: int) -> tuple[float, ...]:
    """Expected scorer + minute points of predicting 0..max_goals goals, using
    the first n picks for n goals (goals beyond the picks have no scorer)."""
    per_hit = POINTS_PER_SCORER + POINTS_PER_MINUTE * MINUTE_HIT_RATE
    values = [0.0]
    for n in range(1, max_goals + 1):
        gain = picks[n - 1].probability * per_hit if n <= len(picks) else 0.0
        values.append(values[-1] + gain)
    return tuple(values)
