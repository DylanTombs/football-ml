"""FPL news items from a pre-deadline snapshot: the only text the news extractor reads."""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from football_ml.config import FPL_DEPARTED_STATUSES
from football_ml.sources.fpl import team_names


@dataclass(frozen=True)
class NewsItem:
    player_id: int
    name: str
    team: str
    status: str
    chance_next_round: int | None
    news: str
    news_added: date | None


def _news_date(raw: str | None) -> date | None:
    return pd.Timestamp(raw).date() if raw else None


def news_items(snapshot: dict) -> tuple[NewsItem, ...]:
    """Players with news who are still at the club (departures are already final in FPL)."""
    teams = team_names(snapshot)
    return tuple(
        NewsItem(
            player_id=e["id"],
            name=e["web_name"],
            team=teams[e["team"]],
            status=e.get("status") or "",
            chance_next_round=e.get("chance_of_playing_next_round"),
            news=e["news"].strip(),
            news_added=_news_date(e.get("news_added")),
        )
        for e in snapshot["elements"]
        if (e.get("news") or "").strip() and e.get("status") not in FPL_DEPARTED_STATUSES
    )
