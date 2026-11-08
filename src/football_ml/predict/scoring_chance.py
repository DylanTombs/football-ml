"""Plan 05 scorer model: each player's expected goals in a fixture.

    μ = team open-play rate × (shrunk npxG per 90 × expected minutes) / Σ squad
        + team penalty rate × PENALTY_XG × taker's expected minutes / 90

Open-play rate = team λ less own goals and penalties. Shares are normalised
over the available squad, so a missing player's threat passes to whoever
plays instead (λ itself is left to the team model). No IO here."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from football_ml.config import (
    EXPECTED_MINUTES_MATCHES,
    FALLBACK_NPXG_PER90,
    OWN_GOAL_SHARE,
    PENALTY_PRIOR_MATCHES,
    PENALTY_XG,
    POSITION_MIN_MINUTES,
    PRIOR_PENALTY_ATTEMPTS_PER_MATCH,
    SCORER_PRIOR_MINUTES,
)
from football_ml.predict.scorers import Player
from football_ml.sources.player_history import Appearance
from football_ml.sources.results import ActualMatch

PenaltyGoals = Mapping[tuple[int, int], int]  # (fixture id, player id) -> penalty goals


@dataclass(frozen=True)
class ScorerProfile:
    player_id: int
    position: int  # FPL element_type
    minutes: int
    npxg: float
    recent_minutes: tuple[int, ...]  # the team's last N matches, oldest first
    penalty_attempts: int


@dataclass(frozen=True)
class ScorerModel:
    """Everything known about scorers before one gameweek."""
    profiles: Mapping[int, ScorerProfile]
    position_rates: Mapping[int, float]  # npxG per 90
    penalty_rates: Mapping[str, float]  # attempts per match, shrunk
    penalty_order: Mapping[int, int]  # player id -> FPL order (live only)
    prior_minutes: float


def penalty_goals(matches: Sequence[ActualMatch]) -> dict[tuple[int, int], int]:
    counts = Counter(
        (m.fixture_id, g.scorer_id) for m in matches for g in m.goals
        if g.penalty and not g.own_goal and g.scorer_id is not None
    )
    return dict(counts)


def shrunk_npxg_per90(npxg: float, minutes: float, position_rate: float, prior_minutes: float) -> float:
    return 90 * (npxg + position_rate * prior_minutes / 90) / (minutes + prior_minutes) if minutes + prior_minutes else 0.0


def expected_minutes(recent_minutes: Sequence[int], availability: float) -> float:
    return sum(recent_minutes) / len(recent_minutes) * availability if recent_minutes else 0.0


def _profile(apps: Sequence[Appearance], position: int, pens: PenaltyGoals, recent: int) -> ScorerProfile:
    attempts = [pens.get((a.fixture_id, a.player_id), 0) + a.penalties_missed for a in apps]
    return ScorerProfile(
        player_id=apps[0].player_id,
        position=position,
        minutes=sum(a.minutes for a in apps),
        npxg=sum(max(0.0, a.xg - PENALTY_XG * n) for a, n in zip(apps, attempts)),
        recent_minutes=tuple(a.minutes for a in apps[-recent:]),
        penalty_attempts=sum(attempts),
    )


def _position_rates(profiles: Sequence[ScorerProfile]) -> dict[int, float]:
    def rate(position: int) -> float:
        group = [p for p in profiles if p.position == position]
        minutes = sum(p.minutes for p in group)
        return 90 * sum(p.npxg for p in group) / minutes if minutes >= POSITION_MIN_MINUTES else FALLBACK_NPXG_PER90[position]
    return {position: rate(position) for position in FALLBACK_NPXG_PER90}


def _penalty_rates(apps: Sequence[Appearance], pens: PenaltyGoals) -> dict[str, float]:
    teams = {a.team for a in apps}
    prior = PRIOR_PENALTY_ATTEMPTS_PER_MATCH * PENALTY_PRIOR_MATCHES

    def rate(team: str) -> float:
        rows = [a for a in apps if a.team == team]
        attempts = sum(pens.get((a.fixture_id, a.player_id), 0) + a.penalties_missed for a in rows)
        matches = len({a.fixture_id for a in rows})
        return (attempts + prior) / (matches + PENALTY_PRIOR_MATCHES)
    return {team: rate(team) for team in teams}


def build_scorer_model(
    histories: Mapping[int, Sequence[Appearance]],
    positions: Mapping[int, int],
    pens: PenaltyGoals,
    before: pd.Timestamp,
    penalty_order: Mapping[int, int] | None = None,
    prior_minutes: float = SCORER_PRIOR_MINUTES,
    recent: int = EXPECTED_MINUTES_MATCHES,
) -> ScorerModel:
    """Only appearances that kicked off before `before` count."""
    past = {
        pid: tuple(sorted((a for a in apps if a.kickoff < before), key=lambda a: a.kickoff))
        for pid, apps in histories.items()
    }
    profiles = {
        pid: _profile(apps, positions.get(pid, 3), pens, recent) for pid, apps in past.items() if apps
    }
    all_apps = tuple(a for apps in past.values() for a in apps)
    return ScorerModel(
        profiles=profiles,
        position_rates=_position_rates(tuple(profiles.values())),
        penalty_rates=_penalty_rates(all_apps, pens),
        penalty_order=dict(penalty_order or {}),
        prior_minutes=prior_minutes,
    )


def _taker(model: ScorerModel, minutes: Mapping[int, float]) -> int | None:
    """First-choice taker who is expected to play: FPL order when known,
    otherwise the player with the most attempts so far."""
    candidates = [pid for pid in minutes if minutes[pid] > 0]
    if model.penalty_order:
        ordered = sorted((pid for pid in candidates if pid in model.penalty_order), key=model.penalty_order.get)
    else:
        ordered = sorted(
            (pid for pid in candidates if model.profiles[pid].penalty_attempts > 0),
            key=lambda pid: (-model.profiles[pid].penalty_attempts, -model.profiles[pid].minutes),
        )
    return ordered[0] if ordered else None


def goal_rates(model: ScorerModel, squad: Sequence[Player], team: str, team_rate: float) -> dict[int, float]:
    """Expected goals per player for one fixture; players with no chance are left out."""
    known = [p for p in squad if p.id in model.profiles and not p.departed]
    minutes = {p.id: expected_minutes(model.profiles[p.id].recent_minutes, p.availability) for p in known}
    threat = {
        pid: shrunk_npxg_per90(prof.npxg, prof.minutes, model.position_rates[prof.position], model.prior_minutes) * minutes[pid]
        for pid, prof in ((pid, model.profiles[pid]) for pid in minutes)
    }
    total = sum(threat.values())
    penalty = model.penalty_rates.get(team, PRIOR_PENALTY_ATTEMPTS_PER_MATCH) * PENALTY_XG
    open_play = max(0.0, team_rate * (1 - OWN_GOAL_SHARE) - penalty)
    rates = {pid: open_play * t / total for pid, t in threat.items() if t > 0} if total > 0 else {}
    taker = _taker(model, minutes)
    if taker is not None:
        rates = {**rates, taker: rates.get(taker, 0.0) + penalty * min(1.0, minutes[taker] / 90)}
    return rates
