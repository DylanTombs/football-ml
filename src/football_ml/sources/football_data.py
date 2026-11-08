"""Historical English league results from football-data.co.uk."""

import io
import math
from dataclasses import dataclass

import pandas as pd

from football_ml.config import (
    CURRENT_SEASON,
    FOOTBALL_DATA_URL,
    LIVE_CACHE_MAX_AGE_HOURS,
    ODDS_COLUMNS,
    ODDS_LEAGUE,
    ODDS_MATCH_DAYS,
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
    raw = _read_csv(csv_text)
    missing = set(_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"football-data CSV missing columns: {sorted(missing)}")

    df = raw[list(_COLUMNS)].rename(columns=_COLUMNS).dropna()
    return df.assign(
        date=pd.to_datetime(df["date"], dayfirst=True, format="mixed"),
        home_goals=df["home_goals"].astype(int),
        away_goals=df["away_goals"].astype(int),
    ).reset_index(drop=True)


# A UTF-8 byte-order mark, as text or as it reads when decoded as Latin-1
# (football-data labels its files ISO-8859-1).
_BOMS = ("\ufeff", "\u00ef\u00bb\u00bf")


def _read_csv(csv_text: str) -> pd.DataFrame:
    for bom in _BOMS:
        csv_text = csv_text.removeprefix(bom)
    return pd.read_csv(io.StringIO(csv_text))


def _fetch_csv(season: str, league: str) -> str:
    return fetch_text(
        FOOTBALL_DATA_URL.format(season=season, league=league),
        cache_name=f"football_data_{league}_{season}.csv",
        max_age_hours=LIVE_CACHE_MAX_AGE_HOURS if season == CURRENT_SEASON else None,
    )


def fetch_results(seasons: tuple[str, ...], leagues: tuple[str, ...] = TRAINING_LEAGUES) -> pd.DataFrame:
    """Fetch and combine results for the given season codes (e.g. '2526') and
    leagues, tagged with `league` and `season`."""
    frames = [
        parse_results(_fetch_csv(season, league)).assign(league=league, season=season)
        for season in seasons for league in leagues
    ]
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


# --- Odds ----------------------------------------------------------------------

@dataclass(frozen=True)
class MatchOdds:
    """Decimal market-average odds before closing."""
    home: float
    draw: float
    away: float
    over_2_5: float | None
    under_2_5: float | None

    @property
    def favourite(self) -> str:
        """"home" or "away" (never a draw); level odds go to the home side."""
        return "home" if self.home <= self.away else "away"

    @property
    def over_probability(self) -> float | None:
        """P(over 2.5 goals) with the margin normalised away."""
        if self.over_2_5 is None or self.under_2_5 is None:
            return None
        over, under = 1 / self.over_2_5, 1 / self.under_2_5
        return over / (over + under)


_ODDS_KEYS = ("Date", "HomeTeam", "AwayTeam")
_ODDS_FRAME = ("date", "home", "away", *ODDS_COLUMNS.values())


def parse_odds(csv_text: str, league: str | None = None) -> pd.DataFrame:
    """date/home/away + pre-closing odds. Rows without 1X2 odds are dropped; a
    CSV without odds columns gives an empty frame. `league` keeps one `Div`
    (for multi-league files such as fixtures.csv)."""
    raw = _read_csv(csv_text)
    if league is not None and "Div" in raw.columns:
        raw = raw[raw["Div"] == league]
    if not set(_ODDS_KEYS) | {"AvgH", "AvgD", "AvgA"} <= set(raw.columns):
        return pd.DataFrame(columns=list(_ODDS_FRAME))
    present = [c for c in ODDS_COLUMNS if c in raw.columns]
    df = raw[[*_ODDS_KEYS, *present]].rename(
        columns={"Date": "date", "HomeTeam": "home", "AwayTeam": "away", **ODDS_COLUMNS}
    )
    df = df.dropna(subset=["home", "away", "odds_home", "odds_draw", "odds_away"])
    df = df.assign(**{c: float("nan") for c in _ODDS_FRAME if c not in df.columns})
    return df.assign(date=pd.to_datetime(df["date"], dayfirst=True, format="mixed"))[list(_ODDS_FRAME)].reset_index(drop=True)


def fetch_odds(seasons: tuple[str, ...], league: str = ODDS_LEAGUE) -> pd.DataFrame:
    frames = [parse_odds(_fetch_csv(season, league)) for season in seasons]
    return pd.concat(frames, ignore_index=True)


def _optional(value) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def odds_for(odds: pd.DataFrame, home: str, away: str, kickoff: pd.Timestamp) -> MatchOdds | None:
    """The odds row for this fixture, matched on teams and a kickoff within
    ODDS_MATCH_DAYS of the row's date; None if there isn't one."""
    if odds is None or odds.empty:
        return None
    day = (kickoff.tz_convert(None) if kickoff.tzinfo else kickoff).normalize()
    rows = odds[(odds["home"] == home) & (odds["away"] == away)
                & ((odds["date"] - day).abs() <= pd.Timedelta(days=ODDS_MATCH_DAYS))]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return MatchOdds(
        home=float(row["odds_home"]), draw=float(row["odds_draw"]), away=float(row["odds_away"]),
        over_2_5=_optional(row["odds_over"]), under_2_5=_optional(row["odds_under"]),
    )
