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


def _context():
    return fixture_context(10, FIXTURE, PARAMS, (SAKA, HAVERTZ, SIMMS), DISTRIBUTION, injury_impact=0.0)


def test_expected_points_pick_is_plausible_and_consistent():
    ctx = _context()
    pred = expected_points_prediction(ctx, scorer_model="weights")
    assert ctx.matrix[pred.home_goals, pred.away_goals] >= MIN_SCORELINE_PROBABILITY
    assert len(pred.goals) == pred.home_goals + pred.away_goals
    for goal in pred.goals:
        expected_team = {SAKA.id, HAVERTZ.id} if goal.side == "home" else {SIMMS.id}
        assert goal.scorer.id in expected_team
    minutes = [g.minute.elapsed() for g in pred.goals]
    assert len(set(minutes)) == len(minutes)


def test_expected_points_predicts_at_least_as_many_goals_as_modal():
    ctx = _context()
    pred = expected_points_prediction(ctx, scorer_model="weights")
    assert pred.home_goals + pred.away_goals >= ctx.baseline.home_goals + ctx.baseline.away_goals


def test_prediction_for_score_uses_fixed_minutes():
    pred = prediction_for_score(_context(), 1, 0, scorer_model="weights")
    (goal,) = pred.goals
    assert (goal.side, goal.scorer.name, goal.minute.label) == ("home", "Saka", f"{SINGLE}'")


def test_prediction_for_score_without_squad_leaves_scorer_unknown():
    ctx = replace(_context(), away_squad=())
    pred = prediction_for_score(ctx, 0, 1, scorer_model="weights")
    assert pred.goals[0].scorer is None


def test_dixon_coles_predictor_strategies():
    ctx = _context()
    assert DixonColesPredictor("modal").predict(ctx) == ctx.baseline
    ep = DixonColesPredictor("expected-points", scorer_model="weights")
    assert ep.predict(ctx) == expected_points_prediction(ctx, scorer_model="weights")
    assert ep.describe() == {"type": "dixon-coles", "pick_strategy": "expected-points", "scorer_model": "weights"}
    assert DixonColesPredictor("modal", name="dixon-coles-modal").name == "dixon-coles-modal"


def test_dixon_coles_predictor_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        DixonColesPredictor("vibes")


def test_match_json_reports_the_picks_probability():
    ctx = _context()
    pred = expected_points_prediction(ctx, scorer_model="weights")
    data = match_json(ctx, pred)
    assert data["prediction_probability"] == pytest.approx(ctx.matrix[pred.home_goals, pred.away_goals], abs=1e-4)


def test_match_json_probability_is_zero_beyond_the_matrix():
    ctx = _context()
    data = match_json(ctx, replace(ctx.baseline, home_goals=15))
    assert data["prediction_probability"] == 0.0


def _scorer_model():
    from football_ml.predict.scoring_chance import build_scorer_model
    from football_ml.sources.player_history import Appearance

    def apps(pid, team, xg):
        return tuple(Appearance(pid, k, k, pd.Timestamp("2026-08-15T15:00Z") + pd.Timedelta(weeks=k), team,
                                90, 1, xg, 0, 0) for k in range(8))
    # Havertz is the bigger threat per FPL history, though Saka has more season xG in the Player rows
    histories = {1: apps(1, "Arsenal", 0.1), 2: apps(2, "Arsenal", 0.7), 3: apps(3, "Coventry", 0.3)}
    return build_scorer_model(histories, {1: 3, 2: 4, 3: 4}, {}, before=FIXTURE.kickoff, prior_minutes=0)


def test_context_carries_scoring_rates_when_given_a_scorer_model():
    plain = _context()
    assert plain.home_scoring_rates is None
    ctx = fixture_context(10, FIXTURE, PARAMS, (SAKA, HAVERTZ, SIMMS), DISTRIBUTION, injury_impact=0.0,
                          scorer_model=_scorer_model())
    assert ctx.home_scoring_rates[2] > ctx.home_scoring_rates[1]
    assert set(ctx.away_scoring_rates) == {3}
    assert sum(ctx.home_scoring_rates.values()) <= ctx.summary.expected_home_goals


def test_chance_scorer_model_picks_from_the_rates():
    ctx = fixture_context(10, FIXTURE, PARAMS, (SAKA, HAVERTZ, SIMMS), DISTRIBUTION, injury_impact=0.0,
                          scorer_model=_scorer_model())
    weights = DixonColesPredictor("expected-points", scorer_model="weights").predict(ctx)
    chance = DixonColesPredictor("expected-points", scorer_model="chance").predict(ctx)
    first_home = lambda p: next(g.scorer.name for g in p.goals if g.side == "home")  # noqa: E731
    assert first_home(weights) == "Saka"
    assert first_home(chance) == "Havertz"
    assert DixonColesPredictor("expected-points", scorer_model="chance").describe()["scorer_model"] == "chance"


def test_chance_scorer_model_needs_rates():
    with pytest.raises(ValueError, match="scorer model"):
        DixonColesPredictor("expected-points").predict(_context())  # chance is the default
    with pytest.raises(ValueError, match="scorer model"):
        DixonColesPredictor("expected-points", scorer_model="vibes")

