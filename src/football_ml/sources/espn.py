"""Goal events (team, scorer, clock) from ESPN's public scoreboard, one request per day."""

import json
import logging
from dataclasses import dataclass

import pandas as pd

from football_ml.config import ESPN_SCOREBOARD_URL, ESPN_TO_CANONICAL_TEAM
from football_ml.sources.http import DataSourceError, fetch_text

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GoalEvent:
    kickoff: pd.Timestamp
    home: str
    away: str
    team: str  # team credited with the goal
    scorer: str | None
    clock: str  # raw ESPN clock, e.g. "90'+3'"
    own_goal: bool
    penalty: bool = False


def _canonical(name: str) -> str:
    return ESPN_TO_CANONICAL_TEAM.get(name, name)


def _events_for(event: dict) -> tuple[GoalEvent, ...]:
    goals: list[GoalEvent] = []
    for competition in event.get("competitions", []):
        sides = {c["homeAway"]: c["team"] for c in competition.get("competitors", [])}
        if {"home", "away"} - set(sides):
            continue
        team_by_id = {t["id"]: _canonical(t["displayName"]) for t in sides.values()}
        for detail in competition.get("details", []):
            clock = detail.get("clock", {}).get("displayValue")
            team_id = detail.get("team", {}).get("id")
            if not (detail.get("scoringPlay") and clock and team_id in team_by_id):
                continue
            athletes = detail.get("athletesInvolved") or []
            goals.append(GoalEvent(
                kickoff=pd.Timestamp(event["date"]),
                home=team_by_id[sides["home"]["id"]],
                away=team_by_id[sides["away"]["id"]],
                team=team_by_id[team_id],
                scorer=athletes[0].get("displayName") if athletes else None,
                clock=clock,
                own_goal=bool(detail.get("ownGoal")),
                penalty=bool(detail.get("penaltyKick")),
            ))
    return tuple(goals)


def parse_goal_events(payload: dict) -> tuple[GoalEvent, ...]:
    return tuple(g for event in payload.get("events", []) for g in _events_for(event))


def fetch_goal_events(start: pd.Timestamp, end: pd.Timestamp) -> tuple[GoalEvent, ...]:
    """Goal events for every day in [start, end). Past days are cached forever."""
    today = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
    events: list[GoalEvent] = []
    for day in pd.date_range(_naive(start).normalize(), _naive(end).normalize(), inclusive="left"):
        stamp = day.strftime("%Y%m%d")
        try:
            text = fetch_text(
                ESPN_SCOREBOARD_URL.format(date=stamp),
                cache_name=f"espn/{stamp}.json",
                max_age_hours=None if day < today - pd.Timedelta(days=1) else 1,
            )
            events.extend(parse_goal_events(json.loads(text)))
        except (DataSourceError, json.JSONDecodeError, KeyError) as exc:
            log.warning("Skipping ESPN day %s: %s", stamp, exc)
    return tuple(events)


def _naive(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_convert(None) if ts.tzinfo else ts
