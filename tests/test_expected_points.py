from dataclasses import replace

import pandas as pd
import pytest

from football_ml.config import MIN_SCORELINE_PROBABILITY, REPEAT_PICK_MINUTES
from football_ml.model.dixon_coles import DixonColesParams
from football_ml.predict.scorers import Player
from football_ml.predict.timing import GoalTime
from football_ml.predictors.base import match_json
from football_ml.predictors.context import fixture_context
from football_ml.predictors.dixon_coles import DixonColesPredictor
from football_ml.predictors.expected_points import expected_points_prediction, prediction_for_score
from football_ml.predictors.naive_favourite import NaiveFavouritePredictor
from football_ml.sources.football_data import MatchOdds
from football_ml.sources.fpl import Fixture

PARAMS = DixonColesParams(
    teams=("Arsenal", "Coventry"), attack=(0.5, -0.5), defence=(-0.4, 0.4),
    home_advantage=0.25, rho=-0.05,
)
SAKA = Player(1, "Saka", "Arsenal", xg=4.0, goals=5, minutes=800, available=True)
HAVERTZ = Player(2, "Havertz", "Arsenal", xg=3.0, goals=3, minutes=700, available=True)
SIMMS = Player(3, "Simms", "Coventry", xg=1.0, goals=1, minutes=600, available=True)
DISTRIBUTION = tuple(GoalTime(float(m), f"{m}'") for m in range(1, 91))
FIXTURE = Fixture(id=100, gameweek=10, kickoff=pd.Timestamp("2026-11-07T15:00Z"),
                  home="Arsenal", away="Coventry")
SINGLE = REPEAT_PICK_MINUTES[1][0]


def _context(odds=None):
    ctx = fixture_context(10, FIXTURE, PARAMS, (SAKA, HAVERTZ, SIMMS), DISTRIBUTION, injury_impact=0.0)
    return replace(ctx, odds=odds)


def test_expected_points_pick_is_plausible_and_consistent():
    ctx = _context()
    pred = expected_points_prediction(ctx)
    assert ctx.matrix[pred.home_goals, pred.away_goals] >= MIN_SCORELINE_PROBABILITY
    assert len(pred.goals) == pred.home_goals + pred.away_goals
    for goal in pred.goals:
        expected_team = {SAKA.id, HAVERTZ.id} if goal.side == "home" else {SIMMS.id}
        assert goal.scorer.id in expected_team
    minutes = [g.minute.elapsed() for g in pred.goals]
    assert len(set(minutes)) == len(minutes)


def test_expected_points_predicts_at_least_as_many_goals_as_modal():
    ctx = _context()
    pred = expected_points_prediction(ctx)
    assert pred.home_goals + pred.away_goals >= ctx.baseline.home_goals + ctx.baseline.away_goals


def test_prediction_for_score_uses_fixed_minutes():
    pred = prediction_for_score(_context(), 1, 0)
    (goal,) = pred.goals
    assert (goal.side, goal.scorer.name, goal.minute.label) == ("home", "Saka", f"{SINGLE}'")


def test_prediction_for_score_without_squad_leaves_scorer_unknown():
    ctx = replace(_context(), away_squad=())
    pred = prediction_for_score(ctx, 0, 1)
    assert pred.goals[0].scorer is None


def test_dixon_coles_predictor_strategies():
    ctx = _context()
    assert DixonColesPredictor("modal").predict(ctx) == ctx.baseline
    ep = DixonColesPredictor("expected-points")
    assert ep.predict(ctx) == expected_points_prediction(ctx)
    assert ep.describe() == {"type": "dixon-coles", "pick_strategy": "expected-points"}
    assert DixonColesPredictor("modal", name="dixon-coles-modal").name == "dixon-coles-modal"


def test_dixon_coles_predictor_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        DixonColesPredictor("vibes")


def test_match_json_reports_the_picks_probability():
    ctx = _context()
    pred = expected_points_prediction(ctx)
    data = match_json(ctx, pred)
    assert data["prediction_probability"] == pytest.approx(ctx.matrix[pred.home_goals, pred.away_goals], abs=1e-4)


def test_match_json_probability_is_zero_beyond_the_matrix():
    ctx = _context()
    data = match_json(ctx, replace(ctx.baseline, home_goals=15))
    assert data["prediction_probability"] == 0.0


# --- naive favourite -----------------------------------------------------------

@pytest.mark.parametrize("odds,score", [
    (MatchOdds(home=1.5, draw=4.0, away=6.0, over_2_5=2.2, under_2_5=1.7), (1, 0)),
    (MatchOdds(home=1.5, draw=4.0, away=6.0, over_2_5=1.6, under_2_5=2.4), (2, 1)),
    (MatchOdds(home=4.5, draw=3.8, away=1.8, over_2_5=1.8, under_2_5=2.0), (1, 2)),
    (MatchOdds(home=2.9, draw=2.8, away=2.9, over_2_5=None, under_2_5=None), (1, 0)),  # tie → home, no O/U → 1-0
])
def test_naive_favourite_backs_the_market_favourite(odds, score):
    pred = NaiveFavouritePredictor().predict(_context(odds))
    assert (pred.home_goals, pred.away_goals) == score
    assert not pred.fallback
    assert len(pred.goals) == sum(score)


def test_naive_favourite_without_odds_falls_back_to_baseline():
    ctx = _context()
    pred = NaiveFavouritePredictor().predict(ctx)
    assert pred.fallback
    assert (pred.home_goals, pred.away_goals) == (ctx.baseline.home_goals, ctx.baseline.away_goals)
