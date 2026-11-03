"""Fixtures, squads and player availability from the Fantasy Premier League API."""

import json
from dataclasses import dataclass

import pandas as pd

from football_ml.config import (
    FPL_BOOTSTRAP_URL,
    FPL_FIXTURES_URL,
    FPL_TO_CANONICAL_TEAM,
    LIVE_CACHE_MAX_AGE_HOURS,
    MIN_CHANCE_OF_PLAYING,
)
from football_ml.predict.scorers import Player
from football_ml.sources.http import DataSourceError, fetch_text


@dataclass(frozen=True)
class Fixture:
    id: int
    gameweek: int
    kickoff: pd.Timestamp
    home: str
    away: str


def team_names(bootstrap: dict) -> dict[int, str]:
    return {
        t["id"]: FPL_TO_CANONICAL_TEAM.get(t["name"], t["name"])
        for t in bootstrap["teams"]
    }


def _is_available(element: dict) -> bool:
    chance = element.get("chance_of_playing_next_round")
    if chance is None:
        return element.get("status") == "a"
    return chance >= MIN_CHANCE_OF_PLAYING


def parse_players(bootstrap: dict) -> tuple[Player, ...]:
    teams = team_names(bootstrap)
    return tuple(
        Player(
            id=e["id"],
            name=e["web_name"],
            team=teams[e["team"]],
            xg=float(e.get("expected_goals") or 0.0),
            goals=int(e.get("goals_scored") or 0),
            minutes=int(e.get("minutes") or 0),
            available=_is_available(e),
            full_name=f"{e.get('first_name', '')} {e.get('second_name', '')}".strip(),
        )
        for e in bootstrap["elements"]
    )


def parse_fixtures(raw: list[dict], bootstrap: dict, gameweek: int) -> tuple[Fixture, ...]:
    teams = team_names(bootstrap)
    return tuple(
        Fixture(
            id=f["id"],
            gameweek=f["event"],
            kickoff=pd.Timestamp(f["kickoff_time"]),
            home=teams[f["team_h"]],
            away=teams[f["team_a"]],
        )
        for f in raw
        if f.get("event") == gameweek and f.get("kickoff_time")
    )


def next_gameweek(bootstrap: dict) -> int:
    upcoming = [e["id"] for e in bootstrap["events"] if e.get("is_next")]
    if not upcoming:
        raise DataSourceError("FPL reports no upcoming gameweek (season over?)")
    return upcoming[0]


def season_start(bootstrap: dict) -> pd.Timestamp:
    return min(pd.Timestamp(e["deadline_time"]) for e in bootstrap["events"])


def _fetch_json(url: str, cache_name: str):
    text = fetch_text(url, cache_name=cache_name, max_age_hours=LIVE_CACHE_MAX_AGE_HOURS)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise DataSourceError(f"FPL returned invalid JSON from {url}") from exc


def fetch_bootstrap() -> dict:
    return _fetch_json(FPL_BOOTSTRAP_URL, "fpl_bootstrap.json")


def fetch_raw_fixtures() -> list[dict]:
    return _fetch_json(FPL_FIXTURES_URL, "fpl_fixtures.json")
