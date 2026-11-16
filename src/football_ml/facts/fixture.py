"""Per-fixture facts computed without an LLM: rest days and the next game (any competition),
derbies, and new managers."""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from football_ml.config import DERBIES, NEW_MANAGER_WINDOW_DAYS, SAME_MATCH_HOURS
from football_ml.sources.espn_schedule import ScheduledMatch
from football_ml.sources.fpl import Fixture

_SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class Congestion:
    rest_days: float | None
    previous_competition: str | None
    days_to_next: float | None
    next_competition: str | None


@dataclass(frozen=True)
class ManagerChange:
    team: str
    appointed: date
    manager: str


@dataclass(frozen=True)
class NewManager:
    manager: str
    days_in_charge: int


@dataclass(frozen=True)
class FixtureFacts:
    fixture_id: int
    derby: bool
    home_congestion: Congestion
    away_congestion: Congestion
    home_new_manager: NewManager | None
    away_new_manager: NewManager | None


def _days(delta: pd.Timedelta) -> float:
    return round(delta.total_seconds() / _SECONDS_PER_DAY, 1)


def congestion(schedule: tuple[ScheduledMatch, ...], team: str, kickoff: pd.Timestamp) -> Congestion:
    margin = pd.Timedelta(hours=SAME_MATCH_HOURS)
    own = [m for m in schedule if m.team == team]
    before = max((m for m in own if m.kickoff < kickoff - margin), key=lambda m: m.kickoff, default=None)
    after = min((m for m in own if m.kickoff > kickoff + margin), key=lambda m: m.kickoff, default=None)
    return Congestion(
        rest_days=_days(kickoff - before.kickoff) if before else None,
        previous_competition=before.competition if before else None,
        days_to_next=_days(after.kickoff - kickoff) if after else None,
        next_competition=after.competition if after else None,
    )


def is_derby(home: str, away: str) -> bool:
    return frozenset((home, away)) in DERBIES


def manager_changes(raw: tuple[tuple[str, str, str], ...]) -> tuple[ManagerChange, ...]:
    return tuple(ManagerChange(team, date.fromisoformat(appointed), manager) for team, appointed, manager in raw)


def new_manager(changes: tuple[ManagerChange, ...], team: str, kickoff: pd.Timestamp) -> NewManager | None:
    match_day = kickoff.date()
    latest = max((c for c in changes if c.team == team and c.appointed <= match_day),
                 key=lambda c: c.appointed, default=None)
    if latest is None:
        return None
    days = (match_day - latest.appointed).days
    return NewManager(latest.manager, days) if days <= NEW_MANAGER_WINDOW_DAYS else None


def fixture_facts(fixture: Fixture, schedule: tuple[ScheduledMatch, ...],
                  changes: tuple[ManagerChange, ...]) -> FixtureFacts:
    return FixtureFacts(
        fixture_id=fixture.id,
        derby=is_derby(fixture.home, fixture.away),
        home_congestion=congestion(schedule, fixture.home, fixture.kickoff),
        away_congestion=congestion(schedule, fixture.away, fixture.kickoff),
        home_new_manager=new_manager(changes, fixture.home, fixture.kickoff),
        away_new_manager=new_manager(changes, fixture.away, fixture.kickoff),
    )
