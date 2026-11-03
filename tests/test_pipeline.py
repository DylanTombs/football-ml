import pandas as pd

from football_ml.model.dixon_coles import DixonColesParams
from football_ml.pipeline import predict_fixture
from football_ml.predict.scorers import Player
from football_ml.predict.timing import GoalTime
from football_ml.sources.fpl import Fixture

PARAMS = DixonColesParams(
    teams=("Arsenal", "Coventry"),
    attack=(0.5, -0.5),
    defence=(-0.4, 0.4),
    home_advantage=0.25,
    rho=-0.05,
)
PLAYERS = (
    Player(1, "Saka", "Arsenal", xg=4.0, goals=5, minutes=800, available=True),
    Player(2, "Havertz", "Arsenal", xg=3.0, goals=3, minutes=700, available=True),
    Player(3, "Simms", "Coventry", xg=1.0, goals=1, minutes=600, available=True),
)
DISTRIBUTION = tuple(GoalTime(float(m), f"{m}'") for m in range(1, 91))
FIXTURE = Fixture(id=100, gameweek=10, kickoff=pd.Timestamp("2026-11-07T15:00Z"),
                  home="Arsenal", away="Coventry")


def test_predict_fixture_builds_consistent_timeline():
    pred = predict_fixture(FIXTURE, PARAMS, PLAYERS, DISTRIBUTION)
    score = pred["prediction"]["score"]
    goals = pred["prediction"]["goals"]

    assert pred["home"] == "Arsenal"
    assert len(goals) == score["home"] + score["away"]
    assert sum(g["team"] == "Arsenal" for g in goals) == score["home"]
    keys = [g["sort_key"] for g in goals]
    assert keys == sorted(keys)


def test_predict_fixture_scorers_come_from_right_team():
    pred = predict_fixture(FIXTURE, PARAMS, PLAYERS, DISTRIBUTION)
    for goal in pred["prediction"]["goals"]:
        expected = {"Arsenal": {"Saka", "Havertz"}, "Coventry": {"Simms"}}[goal["team"]]
        assert goal["scorer"] in expected
        assert goal["scorer_id"] in {1, 2, 3}


def test_predict_fixture_probabilities_sum_to_one():
    p = predict_fixture(FIXTURE, PARAMS, PLAYERS, DISTRIBUTION)["probabilities"]
    assert abs(p["home_win"] + p["draw"] + p["away_win"] - 1) < 1e-6


def test_predict_fixture_without_squad_marks_unknown_scorer():
    pred = predict_fixture(FIXTURE, PARAMS, (), DISTRIBUTION)
    for goal in pred["prediction"]["goals"]:
        assert goal["scorer"] is None


SAKA_OUT = (
    Player(1, "Saka", "Arsenal", xg=4.0, goals=5, minutes=800, available=False, availability=0.0),
    *PLAYERS[1:],
)


def test_predict_fixture_lists_missing_players():
    pred = predict_fixture(FIXTURE, PARAMS, SAKA_OUT, DISTRIBUTION, injury_impact=0.5)
    assert pred["missing"]["home"] == [{"name": "Saka", "xg_share": round(4 / 7, 3)}]
    assert pred["missing"]["away"] == []


def test_injury_lowers_expected_goals():
    full = predict_fixture(FIXTURE, PARAMS, PLAYERS, DISTRIBUTION, injury_impact=0.5)
    hurt = predict_fixture(FIXTURE, PARAMS, SAKA_OUT, DISTRIBUTION, injury_impact=0.5)
    assert hurt["expected_goals"]["home"] < full["expected_goals"]["home"]
    assert hurt["expected_goals"]["away"] == full["expected_goals"]["away"]


def test_zero_injury_impact_matches_unadjusted_model():
    full = predict_fixture(FIXTURE, PARAMS, PLAYERS, DISTRIBUTION, injury_impact=0.0)
    hurt = predict_fixture(FIXTURE, PARAMS, SAKA_OUT, DISTRIBUTION, injury_impact=0.0)
    assert hurt["probabilities"] == full["probabilities"]
