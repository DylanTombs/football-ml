"""Historical English league results from football-data.co.uk."""

import io

import pandas as pd

from football_ml.config import (
    CURRENT_SEASON,
    FOOTBALL_DATA_URL,
    LIVE_CACHE_MAX_AGE_HOURS,
    TRAINING_LEAGUES,
)
from football_ml.sources.http import fetch_text

_COLUMNS = {
    "Date": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
}


def parse_results(csv_text: str) -> pd.DataFrame:
    """Parse a football-data CSV into date/home/away/home_goals/away_goals."""
    raw = pd.read_csv(io.StringIO(csv_text.lstrip("﻿")))
    missing = set(_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"football-data CSV missing columns: {sorted(missing)}")

    df = raw[list(_COLUMNS)].rename(columns=_COLUMNS).dropna()
    return df.assign(
        date=pd.to_datetime(df["date"], dayfirst=True, format="mixed"),
        home_goals=df["home_goals"].astype(int),
        away_goals=df["away_goals"].astype(int),
    ).reset_index(drop=True)


def fetch_results(seasons: tuple[str, ...], leagues: tuple[str, ...] = TRAINING_LEAGUES) -> pd.DataFrame:
    """Fetch and combine results for the given season codes (e.g. '2526') and leagues."""
    frames = [
        parse_results(
            fetch_text(
                FOOTBALL_DATA_URL.format(season=season, league=league),
                cache_name=f"football_data_{league}_{season}.csv",
                max_age_hours=LIVE_CACHE_MAX_AGE_HOURS if season == CURRENT_SEASON else None,
            )
        )
        for season in seasons
        for league in leagues
    ]
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
