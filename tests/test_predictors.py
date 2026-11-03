import pandas as pd
import pytest

from football_ml.model.dixon_coles import DixonColesParams
from football_ml.predict.scorers import Player
from football_ml.predict.timing import GoalTime, parse_clock
from football_ml.predictors.base import MatchPrediction, PredictedGoal, match_json
from football_ml.predictors.context import fixture_context, recent_form
from football_ml.predictors.dixon_coles import DixonColesPredictor
from football_ml.sources.fpl import Fixture

PARAMS = DixonColesParams(
    teams=("Arsenal", "Coventry"), attack=(0.5, -0.5), defence=(-0.4, 0.4),
    home_advantage=0.25, rho=-0.05,
)
SAKA = Player(1, "Saka", "Arsenal", xg=4.0, goals=5, minutes=800, available=True)
SIMMS = Player(3, "Simms", "Coventry", xg=1.0, goals=1, minutes=600, available=True)
DISTRIBUTION = tuple(GoalTime(float(m), f"{m}'") for m in range(1, 91))
FIXTURE = Fixture(id=100, gameweek=10, kickoff=pd.Timestamp("2026-11-07T15:00Z"),
                  home="Arsenal", away="Coventry")
RESULTS = pd.DataFrame([
    {"date": pd.Timestamp(f"2026-10-{d:02d}"), "home": h, "away": a, "home_goals": hg, "away_goals": ag}
    for d, h, a, hg, ag in [
        (1, "Arsenal", "Leeds", 2, 0), (8, "Hull", "Arsenal", 1, 1), (15, "Arsenal", "Hull", 0, 1),
        (22, "Leeds", "Arsenal", 0, 3), (29, "Arsenal", "Leeds", 1, 0), (30, "Arsenal", "Hull", 4, 0),
    ]
])


def _context():
    return fixture_context(10, FIXTURE, PARAMS, (SAKA, SIMMS), DISTRIBUTION, injury_impact=0.0)


def test_recent_form_is_last_n_before_date_newest_first():
    form = recent_form(RESULTS, "Arsenal", before=pd.Timestamp("2026-10-30"), n=3)
    assert [(f.opponent, f.venue, f.goals_for, f.goals_against) for f in form] == [
        ("Leeds", "H", 1, 0), ("Leeds", "A", 3, 0), ("Hull", "H", 0, 1),
    ]
    assert form[0].result == "W"


def test_recent_form_handles_unknown_team():
    assert recent_form(RESULTS, "Wrexham", before=pd.Timestamp("2026-11-01"), n=5) == ()


def test_context_baseline_matches_dixon_coles_summary():
    ctx = _context()
    assert (ctx.baseline.home_goals, ctx.baseline.away_goals) == (
        ctx.summary.home_goals, ctx.summary.away_goals)
    assert len(ctx.baseline.goals) == ctx.baseline.home_goals + ctx.baseline.away_goals
    assert ctx.home_squad == (SAKA,)


def test_dixon_coles_predictor_returns_the_baseline():
    ctx = _context()
    assert DixonColesPredictor("modal").predict(ctx) == ctx.baseline


def test_match_json_shape():
    ctx = _context()
    prediction = MatchPrediction(
        home_goals=1, away_goals=1,
        goals=(PredictedGoal("away", parse_clock("70'"), SIMMS), PredictedGoal("home", parse_clock("12'"), SAKA)),
        reasoning="tight game", fallback=False,
    )
    data = match_json(ctx, prediction)
    assert data["prediction"]["score"] == {"home": 1, "away": 1}
    assert [g["scorer"] for g in data["prediction"]["goals"]] == ["Saka", "Simms"]  # sorted by minute
    assert data["prediction"]["goals"][0] == {
        "minute": "12'", "sort_key": 12.0, "team": "Arsenal", "scorer": "Saka", "scorer_id": 1}
    assert data["prediction"]["reasoning"] == "tight game"
    assert data["prediction"]["fallback"] is False
    assert data["baseline_score"] == {"home": ctx.summary.home_goals, "away": ctx.summary.away_goals}
    assert set(data) >= {"probabilities", "expected_goals", "top_scorelines", "missing"}


def test_match_json_omits_empty_reasoning():
    ctx = _context()
    assert "reasoning" not in match_json(ctx, ctx.baseline)["prediction"]


def test_goal_without_scorer_serialises_as_unknown():
    ctx = _context()
    prediction = MatchPrediction(1, 0, (PredictedGoal("home", parse_clock("5'"), None),))
    goal = match_json(ctx, prediction)["prediction"]["goals"][0]
    assert goal["scorer"] is None and goal["scorer_id"] is None


def test_predicted_goal_rejects_bad_side():
    with pytest.raises(ValueError):
        PredictedGoal("middle", parse_clock("5'"), None)
