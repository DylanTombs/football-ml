"""Fixtures, squads and player availability from the Fantasy Premier League API."""

import json
from dataclasses import dataclass

import pandas as pd

from football_ml.config import (
    FPL_BOOTSTRAP_URL,
    FPL_DEPARTED_STATUSES,
    FPL_EVENT_LIVE_URL,
    FPL_FIXTURES_URL,
    FPL_OUT_STATUSES,
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


def _availability(element: dict) -> float:
    """Chance of playing next round as 0..1; injured/suspended with no chance given → 0."""
    chance = element.get("chance_of_playing_next_round")
    if chance is not None:
        return max(0.0, min(1.0, float(chance) / 100))
    return 0.0 if element.get("status") in FPL_OUT_STATUSES else 1.0


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
            availability=_availability(e),
            departed=e.get("status") in FPL_DEPARTED_STATUSES,
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


def parse_gameweek_minutes(live: dict) -> dict[int, int]:
    """Minutes per player id from an `event/{gw}/live/` payload."""
    try:
        return {int(e["id"]): int(e["stats"]["minutes"]) for e in live["elements"]}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unexpected FPL live payload: {exc!r}") from exc


def fetch_gameweek_minutes(gameweeks: tuple[int, ...]) -> dict[int, dict[int, int]]:
    """Per-gameweek minutes for finished gameweeks (cached for good)."""
    minutes = {}
    for gw in gameweeks:
        url = FPL_EVENT_LIVE_URL.format(gameweek=gw)
        text = fetch_text(url, cache_name=f"fpl_live/gw{gw:02d}.json", max_age_hours=None)
        try:
            minutes[gw] = parse_gameweek_minutes(json.loads(text))
        except (json.JSONDecodeError, ValueError) as exc:
            raise DataSourceError(f"FPL returned unusable data from {url}: {exc}") from exc
    return minutes


def finished_gameweeks(bootstrap: dict) -> tuple[int, ...]:
    return tuple(e["id"] for e in bootstrap["events"] if e.get("finished"))
