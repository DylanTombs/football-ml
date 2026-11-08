import math

import numpy as np
import pytest

from football_ml.config import MINUTE_HIT_RATE, POINTS_PER_MINUTE, POINTS_PER_SCORER
from football_ml.predict.picks import (
    best_scoreline,
    cumulative_scorer_points,
    outcome_expected_points,
    pick_scorers,
)
from football_ml.predict.scorers import Player

# rows = home goals, columns = away goals; sums to 1
MATRIX = np.array([
    [0.10, 0.08, 0.02],
    [0.15, 0.12, 0.03],
    [0.20, 0.25, 0.05],
])
NO_SCORERS = (0.0, 0.0, 0.0)

STRIKER = Player(1, "Striker", "Arsenal", xg=5.0, goals=5, minutes=900, available=True)
WINGER = Player(2, "Winger", "Arsenal", xg=1.0, goals=1, minutes=900, available=True)
INJURED = Player(3, "Injured", "Arsenal", xg=9.0, goals=9, minutes=900, available=False)
BENCH = Player(4, "Bench", "Arsenal", xg=0.0, goals=0, minutes=0, available=True)


@pytest.mark.parametrize("score,expected", [
    ((2, 1), 0.60 + 2 * 0.25),   # home win 0.60, exact 0.25
    ((1, 1), 0.27 + 2 * 0.12),   # draw 0.27
    ((0, 2), 0.13 + 2 * 0.02),   # away win 0.13
])
def test_outcome_expected_points(score, expected):
    assert outcome_expected_points(MATRIX, *score) == pytest.approx(expected)


def test_best_scoreline_without_scorers_is_best_outcome_pick():
    pick = best_scoreline(MATRIX, NO_SCORERS, NO_SCORERS, fallback=(2, 0), floor=0.05)
    assert (pick.home_goals, pick.away_goals) == (2, 1)
    assert pick.probability == pytest.approx(0.25)
    assert pick.expected_points == pytest.approx(1.10)


def test_best_scoreline_respects_the_floor():
    rising = (0.0, 1.0, 2.0)
    assert _score(best_scoreline(MATRIX, rising, rising, fallback=(2, 1), floor=0.05)) == (2, 2)
    # 2-2 is exactly 5%: a higher floor rules it out
    assert _score(best_scoreline(MATRIX, rising, rising, fallback=(2, 1), floor=0.06)) == (2, 1)


def test_best_scoreline_keeps_fallback_when_nothing_beats_it():
    pick = best_scoreline(MATRIX, NO_SCORERS, NO_SCORERS, fallback=(2, 1), floor=0.05)
    assert _score(pick) == (2, 1)
    assert pick.is_fallback


def test_best_scoreline_falls_back_when_floor_excludes_everything():
    pick = best_scoreline(MATRIX, NO_SCORERS, NO_SCORERS, fallback=(1, 0), floor=0.5)
    assert _score(pick) == (1, 0)
    assert pick.is_fallback


def test_best_scoreline_needs_a_value_per_goal_count():
    with pytest.raises(ValueError):
        best_scoreline(MATRIX, (0.0,), NO_SCORERS, fallback=(1, 0), floor=0.05)


def test_pick_scorers_uses_exact_repeat_probabilities():
    # weights 5 : 1 of a 1.2-goal team → μ = 1.0 and 0.2
    picks = pick_scorers((STRIKER, WINGER), team_rate=1.2, n=3)
    assert [p.player.id for p in picks] == [1, 1, 2]
    assert picks[0].probability == pytest.approx(1 - math.exp(-1.0))              # P(X ≥ 1)
    assert picks[1].probability == pytest.approx(1 - 2 * math.exp(-1.0))          # P(X ≥ 2) = 0.264 > 0.181
    assert picks[2].probability == pytest.approx(1 - math.exp(-0.2))              # beats P(X ≥ 3) = 0.080


def test_pick_scorers_skips_unavailable_and_unused_players():
    picks = pick_scorers((INJURED, BENCH, WINGER), team_rate=1.0, n=1)
    assert [p.player.id for p in picks] == [2]
    assert picks[0].probability == pytest.approx(1 - math.exp(-1.0))  # all the team's rate


def test_pick_scorers_edge_cases():
    assert pick_scorers((), team_rate=1.5, n=2) == ()
    assert pick_scorers((STRIKER,), team_rate=1.5, n=0) == ()


def test_pick_scorers_is_prefix_consistent():
    squad = (STRIKER, WINGER)
    assert pick_scorers(squad, 2.4, 2) == pick_scorers(squad, 2.4, 5)[:2]


def test_cumulative_scorer_points():
    picks = pick_scorers((STRIKER, WINGER), team_rate=1.2, n=3)
    values = cumulative_scorer_points(picks, max_goals=4)
    per_hit = POINTS_PER_SCORER + POINTS_PER_MINUTE * MINUTE_HIT_RATE
    assert values[0] == 0.0
    assert values[2] == pytest.approx((picks[0].probability + picks[1].probability) * per_hit)
    assert values[4] == values[3]  # only three picks to give out
    assert len(values) == 5


def _score(pick):
    return pick.home_goals, pick.away_goals


def test_pick_scorers_from_given_rates():
    """With per-player rates the squad weights are ignored; players without a rate can't be picked."""
    picks = pick_scorers((STRIKER, WINGER, BENCH), team_rate=9.9, n=2, rates={2: 0.8, 4: 0.1})
    assert [p.player.id for p in picks] == [2, 2]
    assert picks[0].probability == pytest.approx(1 - math.exp(-0.8))
    assert picks[1].probability == pytest.approx(1 - math.exp(-0.8) * (1 + 0.8))
    assert pick_scorers((STRIKER,), team_rate=1.0, n=1, rates={}) == ()
