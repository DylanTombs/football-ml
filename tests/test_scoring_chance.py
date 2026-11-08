import pandas as pd
import pytest

from football_ml.config import OWN_GOAL_SHARE, PENALTY_XG
from football_ml.predict.scorers import Player
from football_ml.predict.scoring_chance import (
    build_scorer_model,
    expected_minutes,
    goal_rates,
    penalty_goals,
    shrunk_npxg_per90,
)
from football_ml.sources.player_history import Appearance
from football_ml.sources.results import ActualGoal, ActualMatch

T0 = pd.Timestamp("2026-08-15T15:00Z")


def _apps(pid, team, rows):
    """rows: (minutes, xg, goals, missed) per consecutive weekly fixture."""
    return tuple(
        Appearance(pid, k + 1, k + 1, T0 + pd.Timedelta(weeks=k), team, m, int(m >= 60), xg, g, missed)
        for k, (m, xg, g, missed) in enumerate(rows)
    )


def _player(pid, team="Arsenal", availability=1.0):
    return Player(pid, f"P{pid}", team, 0.0, 0, 90, True, availability=availability)


def test_shrinkage_pulls_small_samples_to_the_position_rate():
    assert shrunk_npxg_per90(npxg=0.0, minutes=0, position_rate=0.3, prior_minutes=900) == pytest.approx(0.3)
    # 900 minutes of 0.6 per 90 + 900 prior minutes at 0.3 -> halfway
    assert shrunk_npxg_per90(npxg=6.0, minutes=900, position_rate=0.3, prior_minutes=900) == pytest.approx(0.45)


def test_expected_minutes_is_recent_mean_times_availability():
    assert expected_minutes((90, 90, 0, 90, 60), 1.0) == pytest.approx(66.0)
    assert expected_minutes((90, 90), 0.5) == pytest.approx(45.0)
    assert expected_minutes((), 1.0) == 0.0


def test_penalty_goals_counts_only_matched_penalties():
    goals = (
        ActualGoal("Arsenal", 10, "A", None, False, penalty=True),
        ActualGoal("Arsenal", 10, "A", None, False),
        ActualGoal("Arsenal", 30, "O", None, True, penalty=True),  # own goal never counts
    )
    match = ActualMatch(7, 1, "Arsenal", "Hull", 3, 0, goals)
    assert penalty_goals((match,)) == {(7, 10): 1}


def test_model_strips_penalties_and_ignores_matches_after_the_cutoff():
    histories = {
        10: _apps(10, "Arsenal", [(90, 1.0, 1, 0), (90, 0.2, 0, 1), (90, 5.0, 3, 0)]),
    }
    model = build_scorer_model(histories, {10: 4}, {(1, 10): 1}, before=T0 + pd.Timedelta(weeks=2))
    profile = model.profiles[10]
    assert profile.minutes == 180
    # 1.0 - 0.78 (scored pen) + max(0, 0.2 - 0.78) (missed pen) ; the 5.0 xG game is after the cutoff
    assert profile.npxg == pytest.approx(1.0 - PENALTY_XG)
    assert profile.penalty_attempts == 2
    assert profile.recent_minutes == (90, 90)
    assert model.penalty_rates["Arsenal"] == pytest.approx((2 + 0.13 * 20) / (2 + 20))


def test_goal_rates_split_open_play_by_threat_and_minutes():
    histories = {
        10: _apps(10, "Arsenal", [(90, 0.6, 0, 0)] * 10),  # striker, 0.6 per 90
        11: _apps(11, "Arsenal", [(45, 0.15, 0, 0)] * 10),  # sub, 0.3 per 90, 45 mins
        12: _apps(12, "Arsenal", [(90, 0.0, 0, 0)] * 10),  # injured
    }
    model = build_scorer_model(histories, {10: 4, 11: 4, 12: 4}, {}, before=T0 + pd.Timedelta(weeks=20),
                               prior_minutes=0)
    squad = (_player(10), _player(11), _player(12, availability=0.0))
    rates = goal_rates(model, squad, "Arsenal", team_rate=2.0)
    open_play = 2.0 * (1 - OWN_GOAL_SHARE) - model.penalty_rates["Arsenal"] * PENALTY_XG
    # weights: 0.6 * 90 = 54 vs 0.3 * 45 = 13.5
    assert rates[10] == pytest.approx(open_play * 54 / 67.5)
    assert rates[11] == pytest.approx(open_play * 13.5 / 67.5)
    assert 12 not in rates
    assert sum(rates.values()) == pytest.approx(open_play)  # no taker: nobody has taken a penalty


def test_penalty_term_goes_to_first_available_taker():
    histories = {
        10: _apps(10, "Arsenal", [(90, 0.5, 0, 0)] * 10),
        11: _apps(11, "Arsenal", [(90, 0.5, 0, 0)] * 10),
    }
    model = build_scorer_model(histories, {10: 4, 11: 3}, {}, before=T0 + pd.Timedelta(weeks=20),
                               penalty_order={10: 1, 11: 2})
    pen = model.penalty_rates["Arsenal"] * PENALTY_XG
    rates = goal_rates(model, (_player(10), _player(11)), "Arsenal", team_rate=2.0)
    assert rates[10] - rates[11] == pytest.approx(pen)
    rates = goal_rates(model, (_player(10, availability=0.0), _player(11)), "Arsenal", team_rate=2.0)
    assert rates[11] == pytest.approx(2.0 * (1 - OWN_GOAL_SHARE))  # all open play + the penalty


def test_inferred_taker_is_the_player_with_most_attempts():
    histories = {
        10: _apps(10, "Arsenal", [(90, 0.5, 0, 0)] * 4),
        # same open play once two attempts a game (1 scored, 1 missed) are stripped
        11: _apps(11, "Arsenal", [(90, 0.5 + 2 * PENALTY_XG, 1, 1)] * 2 + [(90, 0.5, 0, 0)] * 2),
    }
    model = build_scorer_model(histories, {10: 4, 11: 4}, {(1, 11): 1, (2, 11): 1},
                               before=T0 + pd.Timedelta(weeks=20))
    assert model.profiles[11].npxg == pytest.approx(model.profiles[10].npxg)
    rates = goal_rates(model, (_player(10), _player(11)), "Arsenal", team_rate=2.0)
    assert rates[11] - rates[10] == pytest.approx(model.penalty_rates["Arsenal"] * PENALTY_XG)


def test_unknown_players_and_teams_get_nothing():
    model = build_scorer_model({}, {}, {}, before=T0)
    assert goal_rates(model, (_player(10),), "Arsenal", team_rate=1.5) == {}
