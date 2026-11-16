"""Every team's matches in all competitions (league, cups, Europe) from ESPN team schedules.
Used for rest days and fixture congestion; no LLM involved."""

import json
import logging
from dataclasses import dataclass

import pandas as pd

from football_ml.config import (
    ESPN_FRIENDLY_MARKER,
    ESPN_TEAM_FIXTURES_URL,
    ESPN_TEAM_SCHEDULE_URL,
    ESPN_TEAMS_URL,
    ESPN_TO_CANONICAL_TEAM,
    LIVE_CACHE_MAX_AGE_HOURS,
)
from football_ml.sources.http import DataSourceError, fetch_text

log = logging.getLogger(__name__)

_FETCH_ERRORS = (DataSourceError, json.JSONDecodeError, KeyError, IndexError, TypeError)


@dataclass(frozen=True)
class ScheduledMatch:
    team: str
    kickoff: pd.Timestamp
    competition: str  # ESPN league slug, e.g. "eng.1", "uefa.champions", "eng.league_cup"


def parse_team_ids(payload: dict) -> dict[str, str]:
    teams = payload["sports"][0]["leagues"][0]["teams"]
    return {
        ESPN_TO_CANONICAL_TEAM.get(t["team"]["displayName"], t["team"]["displayName"]): t["team"]["id"]
        for t in teams
    }


def parse_schedule(payload: dict, team: str) -> tuple[ScheduledMatch, ...]:
    matches = []
    for event in payload.get("events", []):
        slug = (event.get("league") or {}).get("slug") or ""
        if not event.get("date") or ESPN_FRIENDLY_MARKER in slug:
            continue
        matches.append(ScheduledMatch(team, pd.Timestamp(event["date"]), slug))
    return tuple(matches)


def _team_schedule(team: str, team_id: str) -> tuple[ScheduledMatch, ...]:
    matches: list[ScheduledMatch] = []
    for url, kind in ((ESPN_TEAM_SCHEDULE_URL, "past"), (ESPN_TEAM_FIXTURES_URL, "upcoming")):
        try:
            text = fetch_text(url.format(team_id=team_id), cache_name=f"espn_schedule/{team_id}-{kind}.json",
                              max_age_hours=LIVE_CACHE_MAX_AGE_HOURS)
            matches.extend(parse_schedule(json.loads(text), team))
        except _FETCH_ERRORS as exc:
            log.warning("Skipping ESPN %s schedule for %s: %s", kind, team, exc)
    return tuple(matches)


def fetch_team_schedules(teams: tuple[str, ...]) -> tuple[ScheduledMatch, ...]:
    """All competitive matches, past and upcoming, for each canonical team name given."""
    try:
        ids = parse_team_ids(json.loads(fetch_text(ESPN_TEAMS_URL, cache_name="espn_schedule/teams.json",
                                                   max_age_hours=LIVE_CACHE_MAX_AGE_HOURS)))
    except _FETCH_ERRORS as exc:
        log.warning("ESPN team list unavailable, no congestion facts: %s", exc)
        return ()
    matches: set[ScheduledMatch] = set()
    for team in teams:
        if team not in ids:
            log.warning("ESPN has no team id for %s; no congestion facts for it", team)
            continue
        matches.update(_team_schedule(team, ids[team]))
    return tuple(sorted(matches, key=lambda m: (m.team, m.kickoff)))
