import json

import pandas as pd

from football_ml.sources import espn_schedule
from football_ml.sources.espn_schedule import ScheduledMatch, fetch_team_schedules, parse_schedule, parse_team_ids
from football_ml.sources.http import DataSourceError

TEAMS = {"sports": [{"leagues": [{"teams": [
    {"team": {"id": "359", "displayName": "Arsenal"}},
    {"team": {"id": "367", "displayName": "Tottenham Hotspur"}},
]}]}]}


def _event(date, slug, name="x"):
    return {"date": date, "name": name, "league": {"slug": slug, "name": name}}


PAST = {"events": [
    _event("2026-11-07T15:00Z", "eng.1"),
    _event("2026-11-04T20:00Z", "uefa.champions"),
    _event("2026-08-01T18:00Z", "club.friendly"),
]}
UPCOMING = {"events": [
    _event("2026-11-21T17:30Z", "eng.1"),
    _event("2026-11-24T20:00Z", "uefa.champions"),
]}


def test_team_ids_use_canonical_names():
    assert parse_team_ids(TEAMS) == {"Arsenal": "359", "Tottenham": "367"}


def test_schedule_keeps_competitive_matches_with_their_competition():
    matches = parse_schedule(PAST, "Arsenal")
    assert matches == (
        ScheduledMatch("Arsenal", pd.Timestamp("2026-11-07T15:00Z"), "eng.1"),
        ScheduledMatch("Arsenal", pd.Timestamp("2026-11-04T20:00Z"), "uefa.champions"),
    )


def test_schedule_skips_events_without_a_date():
    assert parse_schedule({"events": [{"league": {"slug": "eng.1"}}]}, "Arsenal") == ()


def _fake_fetch(responses, calls):
    def fake(url, cache_name, max_age_hours):
        calls.append(url)
        for marker, body in responses.items():
            if marker in url:
                if isinstance(body, Exception):
                    raise body
                return json.dumps(body)
        raise AssertionError(f"unexpected url {url}")
    return fake


def test_fetch_combines_past_and_upcoming_for_each_requested_team(monkeypatch):
    calls = []
    monkeypatch.setattr(espn_schedule, "fetch_text", _fake_fetch({
        "eng.1/teams?": TEAMS, "359/schedule?fixture=true": UPCOMING, "359/schedule?": PAST,
    }, calls))
    matches = fetch_team_schedules(("Arsenal",))
    assert [m.kickoff.day for m in matches] == [4, 7, 21, 24]
    assert {m.team for m in matches} == {"Arsenal"}
    assert len(calls) == 3


def test_fetch_skips_a_team_espn_does_not_know_and_a_failed_request(monkeypatch, caplog):
    monkeypatch.setattr(espn_schedule, "fetch_text", _fake_fetch({
        "eng.1/teams?": TEAMS, "359/schedule?fixture=true": DataSourceError("down"), "359/schedule?": PAST,
    }, []))
    matches = fetch_team_schedules(("Arsenal", "Wrexham"))
    assert [m.kickoff.day for m in matches] == [4, 7]
    assert "Wrexham" in caplog.text and "down" in caplog.text


def test_fetch_returns_nothing_when_the_team_list_fails(monkeypatch, caplog):
    monkeypatch.setattr(espn_schedule, "fetch_text", _fake_fetch({"eng.1/teams?": DataSourceError("down")}, []))
    assert fetch_team_schedules(("Arsenal",)) == ()
    assert "team list" in caplog.text
