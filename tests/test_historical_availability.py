import pytest

import pandas as pd

from football_ml.historical_availability import players_as_of, players_before, team_gameweeks
from football_ml.predict.scorers import Player
from football_ml.sources.fpl import parse_gameweek_minutes

BOOTSTRAP = {"teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Hull"}]}

PLAYERS = (
    Player(1, "Saka", "Arsenal", xg=4.0, goals=4, minutes=600, available=False, availability=0.0),
    Player(2, "Havertz", "Arsenal", xg=2.0, goals=2, minutes=600, available=True),
    Player(3, "NewSigning", "Arsenal", xg=1.0, goals=1, minutes=90, available=True),
    Player(4, "Loanee", "Arsenal", xg=0.5, goals=0, minutes=90, available=False,
           availability=0.0, departed=True),
)

# Saka plays GW1-2 then misses GW3; NewSigning only debuts in GW3; Arsenal blank in GW4.
MINUTES = {
    1: {1: 90, 2: 90, 4: 90},
    2: {1: 90, 2: 0, 4: 90},
    3: {1: 0, 2: 90, 3: 90},
    5: {1: 90, 2: 90, 3: 90},
}


def _fixture(gw, home, away, finished=True):
    return {"event": gw, "team_h": home, "team_a": away, "finished": finished}


RAW_FIXTURES = [
    _fixture(1, 1, 2), _fixture(2, 2, 1), _fixture(3, 1, 2),
    _fixture(5, 2, 1), _fixture(6, 1, 2, finished=False),
]


def test_parse_gameweek_minutes():
    live = {"elements": [{"id": 1, "stats": {"minutes": 90}}, {"id": 2, "stats": {"minutes": 0}}]}
    assert parse_gameweek_minutes(live) == {1: 90, 2: 0}


def test_parse_gameweek_minutes_rejects_bad_payload():
    with pytest.raises(ValueError):
        parse_gameweek_minutes({"nope": []})


def test_team_gameweeks_only_counts_finished_matches():
    assert team_gameweeks(RAW_FIXTURES, BOOTSTRAP)["Arsenal"] == (1, 2, 3, 5)


def _avail(gameweek, proxy):
    teams = team_gameweeks(RAW_FIXTURES, BOOTSTRAP)
    return {p.name: p.availability for p in players_as_of(PLAYERS, MINUTES, teams, gameweek, proxy)}


def test_lagged_marks_players_who_missed_the_previous_game():
    # GW3 looks at GW2: Havertz played GW1, then 0 minutes in GW2 → missing
    assert _avail(3, "lagged") == {"Saka": 1.0, "Havertz": 0.0, "NewSigning": 1.0, "Loanee": 1.0}


def test_lagged_skips_blank_gameweeks():
    # GW5 looks at GW3 (Arsenal blank in GW4): Saka missed it; NewSigning debuted so isn't missing
    avail = _avail(5, "lagged")
    assert avail["Saka"] == 0.0
    assert avail["Havertz"] == 1.0
    assert avail["NewSigning"] == 1.0


def test_oracle_uses_the_gameweek_itself():
    assert _avail(3, "oracle")["Saka"] == 0.0
    assert _avail(2, "oracle")["Havertz"] == 0.0


def test_first_gameweek_has_everyone_available():
    assert set(_avail(1, "lagged").values()) == {1.0}
    assert set(_avail(1, "oracle").values()) == {1.0}


def test_historical_players_are_not_treated_as_departed():
    teams = team_gameweeks(RAW_FIXTURES, BOOTSTRAP)
    assert not any(p.departed for p in players_as_of(PLAYERS, MINUTES, teams, 3, "lagged"))


def test_scorer_eligibility_is_left_alone():
    teams = team_gameweeks(RAW_FIXTURES, BOOTSTRAP)
    result = {p.name: p.available for p in players_as_of(PLAYERS, MINUTES, teams, 3, "lagged")}
    assert result == {p.name: p.available for p in PLAYERS}


def test_live_proxy_returns_players_unchanged():
    teams = team_gameweeks(RAW_FIXTURES, BOOTSTRAP)
    assert players_as_of(PLAYERS, MINUTES, teams, 3, "live") == PLAYERS


def test_unknown_proxy_is_rejected():
    with pytest.raises(ValueError, match="proxy"):
        players_as_of(PLAYERS, MINUTES, {}, 3, "crystal-ball")


def test_fetch_gameweek_minutes_caches_each_finished_gameweek(monkeypatch):
    from football_ml.sources import fpl
    calls = []

    def fake_fetch(url, cache_name, max_age_hours):
        calls.append((url, cache_name, max_age_hours))
        return '{"elements": [{"id": 7, "stats": {"minutes": 45}}]}'

    monkeypatch.setattr(fpl, "fetch_text", fake_fetch)
    assert fpl.fetch_gameweek_minutes((1, 2)) == {1: {7: 45}, 2: {7: 45}}
    assert calls[0][1:] == ("fpl_live/gw01.json", None)
    assert "event/2/live" in calls[1][0]


def test_fetch_gameweek_minutes_wraps_bad_json(monkeypatch):
    from football_ml.sources import fpl
    from football_ml.sources.http import DataSourceError
    monkeypatch.setattr(fpl, "fetch_text", lambda *a, **k: "<html>")
    with pytest.raises(DataSourceError):
        fpl.fetch_gameweek_minutes((1,))


def test_finished_gameweeks():
    from football_ml.sources.fpl import finished_gameweeks
    bootstrap = {"events": [{"id": 1, "finished": True}, {"id": 2, "finished": False}]}
    assert finished_gameweeks(bootstrap) == (1,)


def test_players_before_rebuilds_season_stats_from_history():
    from football_ml.sources.player_history import Appearance
    t0 = pd.Timestamp("2026-08-15T15:00Z")
    apps = tuple(Appearance(7, k, k, t0 + pd.Timedelta(weeks=k), "Arsenal", 90, 1, 0.4, 1, 0) for k in range(4))
    live = Player(7, "Saka", "Arsenal", xg=9.9, goals=9, minutes=999, available=False, availability=1.0)
    out = Player(8, "Out", "Arsenal", xg=1.0, goals=1, minutes=90, available=True, availability=0.0)
    saka, gone = players_before((live, out), {7: apps}, before=t0 + pd.Timedelta(weeks=2))
    assert (saka.xg, saka.goals, saka.minutes) == (pytest.approx(0.8), 2, 180)
    assert saka.available  # eligibility follows the as-of availability, not today's FPL flag
    assert (gone.minutes, gone.available) == (0, False)
