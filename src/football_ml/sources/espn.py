"""Goal event clocks from ESPN's public scoreboard (one request per day)."""

import json
import logging

import pandas as pd

from football_ml.config import ESPN_SCOREBOARD_URL
from football_ml.sources.http import DataSourceError, fetch_text

log = logging.getLogger(__name__)


def parse_goal_clocks(payload: dict) -> tuple[str, ...]:
    """Raw clock strings (e.g. "90'+3'") for every goal in a scoreboard payload."""
    return tuple(
        detail["clock"]["displayValue"]
        for event in payload.get("events", [])
        for competition in event.get("competitions", [])
        for detail in competition.get("details", [])
        if detail.get("scoringPlay") and detail.get("clock", {}).get("displayValue")
    )


def fetch_goal_clocks(start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, ...]:
    """Goal clocks for every PL match day between start and end (exclusive)."""
    today = pd.Timestamp.now(tz="UTC").normalize()
    clocks: list[str] = []
    for day in pd.date_range(start.normalize(), end.normalize(), inclusive="left"):
        stamp = day.strftime("%Y%m%d")
        try:
            text = fetch_text(
                ESPN_SCOREBOARD_URL.format(date=stamp),
                cache_name=f"espn/{stamp}.json",
                max_age_hours=None if day < today - pd.Timedelta(days=1) else 1,
            )
            clocks.extend(parse_goal_clocks(json.loads(text)))
        except (DataSourceError, json.JSONDecodeError) as exc:
            log.warning("Skipping ESPN day %s: %s", stamp, exc)
    return tuple(clocks)
