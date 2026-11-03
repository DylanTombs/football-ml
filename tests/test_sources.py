import pytest

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
        {"id": 13, "web_name": "Loanee", "team": 2, "expected_goals": "0.5", "goals_scored": 0,
         "minutes": 90, "status": "u", "chance_of_playing_next_round": 0},
        {"id": 14, "web_name": "Banned", "team": 2, "expected_goals": "0.5", "goals_scored": 0,
         "minutes": 90, "status": "s", "chance_of_playing_next_round": None},
        {"id": 15, "web_name": "Fit", "team": 2, "expected_goals": "0.5", "goals_scored": 0,
         "minutes": 90, "status": "a", "chance_of_playing_next_round": 100},
    ],
}


def test_parse_players_availability_fraction():
    players = {p.name: p for p in parse_players(BOOTSTRAP)}
    assert players["Saka"].availability == 1.0
    assert players["Fit"].availability == 1.0
    assert players["Hurt"].availability == 0.0
    assert players["Doubt"].availability == 0.75
    assert players["Banned"].availability == 0.0  # suspended with no chance given
    assert players["Loanee"].departed
    assert not players["Hurt"].departed


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


def test_fetch_results_combines_leagues_and_seasons(monkeypatch):
    from football_ml.sources import football_data

    requested = []

    def fake_fetch(url, cache_name, max_age_hours):
        requested.append(cache_name)
        return CSV

    monkeypatch.setattr(football_data, "fetch_text", fake_fetch)
    df = football_data.fetch_results(("2526", "2627"), leagues=("E0", "E1"))
    assert sorted(requested) == sorted([
        "football_data_E0_2526.csv", "football_data_E1_2526.csv",
        "football_data_E0_2627.csv", "football_data_E1_2627.csv",
    ])
    assert len(df) == 8
    assert df["date"].is_monotonic_increasing


# --- odds (plan 04) ---------------------------------------------------------------

import pandas as pd  # noqa: E402

from football_ml.sources.football_data import MatchOdds, odds_for, parse_odds  # noqa: E402

ODDS_CSV = (
    "﻿Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5,AvgCH,AvgCD,AvgCA\n"
    "E0,21/08/2026,Arsenal,Coventry,3,0,1.30,5.50,9.00,1.70,2.20,1.20,6.00,12.0\n"
    "E0,22/08/2026,Hull,Man United,1,1,4.00,3.60,1.90,,,4.2,3.5,1.8\n"
    "E0,23/08/2026,Spurs,Leeds,,,,,,,,,,\n"
)


def test_parse_odds_keeps_pre_closing_columns_only():
    df = parse_odds(ODDS_CSV)
    assert len(df) == 2  # no 1X2 odds → dropped
    first = odds_for(df, "Arsenal", "Coventry", pd.Timestamp("2026-08-21T19:00Z"))
    assert first == MatchOdds(home=1.30, draw=5.50, away=9.00, over_2_5=1.70, under_2_5=2.20)


def test_odds_for_missing_over_under_and_unknown_fixture():
    df = parse_odds(ODDS_CSV)
    hull = odds_for(df, "Hull", "Man United", pd.Timestamp("2026-08-22T14:00Z"))
    assert hull.over_2_5 is None and hull.over_probability is None
    assert odds_for(df, "Arsenal", "Coventry", pd.Timestamp("2027-02-01T15:00Z")) is None  # return fixture
    assert odds_for(df, "Leeds", "Hull", pd.Timestamp("2026-08-21T15:00Z")) is None


def test_match_odds_favourite_and_over_probability():
    odds = MatchOdds(home=1.30, draw=5.50, away=9.00, over_2_5=1.70, under_2_5=2.20)
    assert odds.favourite == "home"
    assert odds.over_probability == pytest.approx((1 / 1.7) / (1 / 1.7 + 1 / 2.2))
    assert MatchOdds(3.0, 3.0, 2.0, None, None).favourite == "away"


def test_parse_odds_without_odds_columns_is_empty():
    assert parse_odds(CSV).empty
