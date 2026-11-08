import json

import pandas as pd
import pytest

from football_ml.sources import player_history
from football_ml.sources.http import DataSourceError
from football_ml.sources.player_history import Appearance, fixture_teams, parse_history

BOOTSTRAP = {
    "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Spurs"}],
    "elements": [
        {"id": 10, "team": 1, "minutes": 450},
        {"id": 11, "team": 1, "minutes": 0},
        {"id": 20, "team": 2, "minutes": 90},
    ],
}
RAW_FIXTURES = [
    {"id": 1, "event": 1, "team_h": 1, "team_a": 2},
    {"id": 2, "event": 2, "team_h": 2, "team_a": 1},
]


def _row(fixture, was_home, minutes=90, xg="0.45", goals=1, missed=0, starts=1, rnd=1):
    return {
        "element": 10, "fixture": fixture, "was_home": was_home, "round": rnd,
        "kickoff_time": f"2026-08-{fixture + 10}T15:00:00Z", "minutes": minutes, "starts": starts,
        "expected_goals": xg, "goals_scored": goals, "penalties_missed": missed,
    }


def test_fixture_teams_uses_canonical_names():
    assert fixture_teams(RAW_FIXTURES, BOOTSTRAP) == {1: ("Arsenal", "Tottenham"), 2: ("Tottenham", "Arsenal")}


def test_parse_history_takes_the_team_from_the_fixture():
    payload = {"history": [_row(1, True), _row(2, False, minutes=30, xg="0.10", goals=0, missed=1, starts=0, rnd=2)]}
    first, second = parse_history(payload, 10, fixture_teams(RAW_FIXTURES, BOOTSTRAP))
    assert first == Appearance(10, 1, 1, pd.Timestamp("2026-08-11T15:00Z"), "Arsenal", 90, 1, 0.45, 1, 0)
    assert (second.team, second.minutes, second.starts, second.penalties_missed) == ("Arsenal", 30, 0, 1)


def test_parse_history_skips_unknown_fixtures():
    payload = {"history": [_row(99, True)]}
    assert parse_history(payload, 10, fixture_teams(RAW_FIXTURES, BOOTSTRAP)) == ()


def test_parse_history_rejects_malformed_rows():
    bad = {"history": [{"fixture": 1, "was_home": True}]}
    with pytest.raises(ValueError, match="player 10"):
        parse_history(bad, 10, fixture_teams(RAW_FIXTURES, BOOTSTRAP))


def test_fetch_histories_only_for_players_with_minutes(monkeypatch):
    requested = []

    def fake_fetch(url, cache_name, max_age_hours):
        requested.append(cache_name)
        if cache_name.endswith("/20.json"):
            raise DataSourceError("boom")
        return json.dumps({"history": [_row(1, True)]})

    monkeypatch.setattr(player_history, "fetch_text", fake_fetch)
    histories = player_history.fetch_player_histories(BOOTSTRAP, RAW_FIXTURES)
    assert requested == ["fpl_players/10.json", "fpl_players/20.json"]
    assert set(histories) == {10}  # 11 never played; 20 failed and is logged, not fatal
    assert histories[10][0].xg == 0.45
