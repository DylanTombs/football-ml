"""Upcoming pre-closing odds from football-data's fixtures.csv, and the copy
saved with each frozen prediction (the file is overwritten every round)."""

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from football_ml.config import (
    FOOTBALL_DATA_FIXTURES_URL,
    ODDS_LEAGUE,
    ODDS_MATCH_DAYS,
    ODDS_SNAPSHOTS_DIR,
    UPCOMING_ODDS_MAX_AGE_HOURS,
)
from football_ml.snapshots import gameweek_deadline
from football_ml.sources.football_data import parse_odds
from football_ml.sources.fpl import Fixture
from football_ml.sources.http import DataSourceError, fetch_text

log = logging.getLogger(__name__)


def fetch_upcoming_odds() -> pd.DataFrame:
    """Premier League rows of fixtures.csv. Without it predictions fall back to
    the model alone, so a failure is logged rather than raised."""
    try:
        text = fetch_text(FOOTBALL_DATA_FIXTURES_URL, cache_name="football_data_fixtures.csv",
                          max_age_hours=UPCOMING_ODDS_MAX_AGE_HOURS)
        return parse_odds(text, league=ODDS_LEAGUE)
    except (DataSourceError, ValueError, KeyError) as exc:
        log.warning("No upcoming odds (%s); fixtures without odds use the model alone", exc)
        return parse_odds("Date,HomeTeam,AwayTeam\n")


def odds_for_fixtures(odds: pd.DataFrame, fixtures: Sequence[Fixture]) -> pd.DataFrame:
    """Rows matching one of `fixtures` (teams + date within ODDS_MATCH_DAYS),
    the same rule `odds_for` uses. Drops a previous round still in the file."""
    def matches(row) -> bool:
        return any(
            row["home"] == f.home and row["away"] == f.away
            and abs(row["date"] - f.kickoff.tz_convert(None).normalize()) <= pd.Timedelta(days=ODDS_MATCH_DAYS)
            for f in fixtures
        )
    return odds[odds.apply(matches, axis=1)] if not odds.empty else odds


def write_odds_snapshot(
    odds: pd.DataFrame, bootstrap: dict, fixtures: Sequence[Fixture], now: pd.Timestamp,
    odds_dir: Path = ODDS_SNAPSHOTS_DIR,
) -> Path | None:
    """Save the gameweek's odds next to its prediction, before the deadline only."""
    gameweek = fixtures[0].gameweek
    deadline = gameweek_deadline(bootstrap, gameweek)
    if now >= deadline:
        log.warning("GW%d deadline (%s) has passed; not saving odds", gameweek, deadline)
        return None
    odds = odds_for_fixtures(odds, fixtures)
    if odds.empty:
        log.warning("No upcoming odds to save for GW%d", gameweek)
        return None
    path = odds_dir / f"gw{gameweek:02d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    odds.to_csv(path, index=False)
    return path
