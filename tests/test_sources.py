import pytest

from football_ml.sources.espn import parse_goal_clocks
from football_ml.sources.football_data import parse_results
from football_ml.sources.fpl import parse_fixtures, parse_players

CSV = (
    "﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    "E0,21/08/2026,20:00,Arsenal,Coventry,3,0,H\n"
    "E0,22/08/2026,15:00,Hull,Man United,1,1,D\n"
    "E0,23/08/2026,15:00,Spurs,,,,\n"
)


def test_parse_results_normalises_columns_and_drops_incomplete_rows():
    df = parse_results(CSV)
    assert list(df.columns) == ["date", "home", "away", "home_goals", "away_goals"]
    assert len(df) == 2
    assert df.iloc[0]["home"] == "Arsenal"
    assert df.iloc[0]["home_goals"] == 3
    assert df.iloc[0]["date"].month == 8


def test_parse_results_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        parse_results("Div,Date\nE0,21/08/2026\n")


BOOTSTRAP = {
    "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Spurs"}],
    "elements": [
        {"id": 10, "web_name": "Saka", "team": 1, "expected_goals": "2.5", "goals_scored": 3,
         "minutes": 700, "status": "a", "chance_of_playing_next_round": None},
        {"id": 11, "web_name": "Hurt", "team": 2, "expected_goals": "1.0", "goals_scored": 1,
         "minutes": 300, "status": "i", "chance_of_playing_next_round": 0},
        {"id": 12, "web_name": "Doubt", "team": 2, "expected_goals": "1.0", "goals_scored": 1,
         "minutes": 300, "status": "d", "chance_of_playing_next_round": 75},
    ],
}


def test_parse_players_maps_names_and_availability():
    players = {p.name: p for p in parse_players(BOOTSTRAP)}
    assert players["Saka"].team == "Arsenal"
    assert players["Saka"].xg == 2.5
    assert players["Saka"].available
    assert not players["Hurt"].available
    assert players["Doubt"].available  # 75% meets threshold
    assert players["Doubt"].team == "Tottenham"  # FPL name mapped to football-data name


def test_parse_fixtures_filters_gameweek():
    raw = [
        {"id": 100, "event": 10, "team_h": 1, "team_a": 2, "kickoff_time": "2026-11-07T15:00:00Z"},
        {"id": 101, "event": 11, "team_h": 2, "team_a": 1, "kickoff_time": "2026-11-14T15:00:00Z"},
    ]
    fixtures = parse_fixtures(raw, BOOTSTRAP, gameweek=10)
    assert len(fixtures) == 1
    assert (fixtures[0].home, fixtures[0].away) == ("Arsenal", "Tottenham")


def test_parse_goal_clocks_only_scoring_plays():
    payload = {"events": [{"competitions": [{"details": [
        {"scoringPlay": True, "clock": {"displayValue": "15'"}},
        {"scoringPlay": False, "clock": {"displayValue": "27'"}},
        {"scoringPlay": True, "clock": {"displayValue": "90'+3'"}},
    ]}]}]}
    assert parse_goal_clocks(payload) == ("15'", "90'+3'")


def test_parse_goal_clocks_handles_no_events():
    assert parse_goal_clocks({}) == ()
