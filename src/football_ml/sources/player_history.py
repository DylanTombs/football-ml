"""Per-player, per-match FPL history (`element-summary/{id}`): xG, minutes,
starts and missed penalties. Rows for played fixtures never change, so one
fetch serves every past gameweek; callers filter by kickoff to avoid leaks."""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from football_ml.config import FPL_PLAYER_URL, PLAYER_HISTORY_MAX_AGE_HOURS
from football_ml.sources.fpl import team_names
from football_ml.sources.http import DataSourceError, fetch_text

log = logging.getLogger(__name__)

FixtureTeams = Mapping[int, tuple[str, str]]  # fixture id -> (home, away)


@dataclass(frozen=True)
class Appearance:
    """One player's row for one of his team's fixtures (minutes may be 0)."""
    player_id: int
    fixture_id: int
    gameweek: int
    kickoff: pd.Timestamp
    team: str
    minutes: int
    starts: int
    xg: float  # includes penalties (Opta)
    goals: int
    penalties_missed: int


def fixture_teams(raw_fixtures: list[dict], bootstrap: dict) -> dict[int, tuple[str, str]]:
    names = team_names(bootstrap)
    return {f["id"]: (names[f["team_h"]], names[f["team_a"]]) for f in raw_fixtures}


def _appearance(row: dict, player_id: int, teams: FixtureTeams) -> Appearance | None:
    fixture = int(row["fixture"])
    if fixture not in teams:
        log.warning("Player %d: fixture %d not in the FPL fixture list; skipped", player_id, fixture)
        return None
    home, away = teams[fixture]
    return Appearance(
        player_id=player_id,
        fixture_id=fixture,
        gameweek=int(row["round"]),
        kickoff=pd.Timestamp(row["kickoff_time"]),
        team=home if row["was_home"] else away,
        minutes=int(row["minutes"]),
        starts=int(row["starts"]),
        xg=float(row["expected_goals"] or 0.0),
        goals=int(row["goals_scored"]),
        penalties_missed=int(row["penalties_missed"]),
    )


def parse_history(payload: dict, player_id: int, teams: FixtureTeams) -> tuple[Appearance, ...]:
    try:
        parsed = tuple(_appearance(row, player_id, teams) for row in payload["history"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unexpected FPL history for player {player_id}: {exc!r}") from exc
    return tuple(a for a in parsed if a is not None)


def fetch_player_histories(bootstrap: dict, raw_fixtures: list[dict]) -> dict[int, tuple[Appearance, ...]]:
    """Histories for every player with league minutes this season. A player
    whose history can't be fetched is logged and left out (he then has no
    recent minutes, so he's never picked as a scorer)."""
    teams = fixture_teams(raw_fixtures, bootstrap)
    ids = [e["id"] for e in bootstrap["elements"] if int(e.get("minutes") or 0) > 0]
    histories: dict[int, tuple[Appearance, ...]] = {}
    failed: list[int] = []
    for player_id in ids:
        url = FPL_PLAYER_URL.format(player_id=player_id)
        try:
            text = fetch_text(url, cache_name=f"fpl_players/{player_id}.json",
                              max_age_hours=PLAYER_HISTORY_MAX_AGE_HOURS)
            histories = {**histories, player_id: parse_history(json.loads(text), player_id, teams)}
        except (DataSourceError, json.JSONDecodeError, ValueError) as exc:
            log.warning("No FPL history for player %d: %s", player_id, exc)
            failed = [*failed, player_id]
    if failed:
        log.warning("FPL history missing for %d of %d players", len(failed), len(ids))
    return histories
