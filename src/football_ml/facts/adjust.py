"""Facts → multipliers on each side's expected goals, from fixed small priors (config).
Capped per side, then re-centred over the gameweek so the adjustment can't drain or add
goals overall (plan 02's mistake). No IO."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta

from football_ml.config import (
    CONGESTION_AHEAD_DAYS,
    DEFENDER_POSITIONS,
    EUROPEAN_COMPETITION_PREFIX,
    FACT_EFFECTS,
    KEY_ATTACKER_XG_SHARE,
    MAX_ADJUSTMENT,
    NEWS_RECENT_DAYS,
    REGULAR_MINUTES_SHARE,
    SHORT_REST_DAYS,
)
from football_ml.facts.extractor import PlayerFact
from football_ml.facts.fixture import Congestion, FixtureFacts, NewManager
from football_ml.predict.scorers import Player
from football_ml.predictors.base import SIDES, FixtureContext

_OTHER = {"home": "away", "away": "home"}


@dataclass(frozen=True)
class Adjustment:
    side: str  # whose expected goals change
    kind: str  # key of FACT_EFFECTS
    effect: float
    detail: str


@dataclass(frozen=True)
class FixtureAdjustment:
    fixture_id: int
    home_multiplier: float
    away_multiplier: float
    adjustments: tuple[Adjustment, ...]


def _is_recent(fact: PlayerFact, as_of: date) -> bool:
    return fact.news_added is not None and fact.news_added >= as_of - timedelta(days=NEWS_RECENT_DAYS)


def _chance_missing(fact: PlayerFact, player: Player, match_day: date) -> float:
    if fact.expected_return is not None and fact.expected_return <= match_day:
        return 0.0
    return 1.0 - player.availability


def _player_adjustments(fact: PlayerFact, player: Player, squad: tuple[Player, ...], side: str,
                        positions: Mapping[int, int], match_day: date) -> list[Adjustment]:
    members = [p for p in squad if not p.departed]
    team_xg = sum(p.xg for p in members)
    key = team_xg > 0 and player.xg / team_xg >= KEY_ATTACKER_XG_SHARE
    if fact.reason == "returned":
        return [Adjustment(side, "key_attacker_back", FACT_EFFECTS["key_attacker_back"],
                           f"{player.name} back")] if key else []
    missing = _chance_missing(fact, player, match_day)
    if missing <= 0:
        return []
    detail = f"{player.name} ({fact.reason}, {missing:.0%} to miss)"
    out = [Adjustment(side, "key_attacker_out", FACT_EFFECTS["key_attacker_out"] * missing, detail)] if key else []
    most_minutes = max((p.minutes for p in members), default=0)
    regular = most_minutes > 0 and player.minutes >= REGULAR_MINUTES_SHARE * most_minutes
    if positions.get(player.id) in DEFENDER_POSITIONS and regular:
        out.append(Adjustment(_OTHER[side], "defender_out", FACT_EFFECTS["defender_out"] * missing, detail))
    return out


def _congestion_adjustments(congestion: Congestion, side: str) -> list[Adjustment]:
    out = []
    if congestion.rest_days is not None and congestion.rest_days < SHORT_REST_DAYS:
        out.append(Adjustment(side, "short_rest", FACT_EFFECTS["short_rest"],
                              f"{congestion.rest_days} days after {congestion.previous_competition}"))
    if (congestion.days_to_next is not None and congestion.days_to_next <= CONGESTION_AHEAD_DAYS
            and (congestion.next_competition or "").startswith(EUROPEAN_COMPETITION_PREFIX)):
        out.append(Adjustment(side, "europe_next", FACT_EFFECTS["europe_next"],
                              f"{congestion.next_competition} in {congestion.days_to_next} days"))
    return out


def _manager_adjustments(manager: NewManager | None, side: str) -> list[Adjustment]:
    if manager is None:
        return []
    return [Adjustment(side, "new_manager", FACT_EFFECTS["new_manager"],
                       f"{manager.manager}, {manager.days_in_charge} days in charge")]


def _fixture_fact_adjustments(facts: FixtureFacts) -> list[Adjustment]:
    out = [Adjustment(side, "derby", FACT_EFFECTS["derby"], "derby") for side in SIDES] if facts.derby else []
    for side, congestion, manager in (("home", facts.home_congestion, facts.home_new_manager),
                                      ("away", facts.away_congestion, facts.away_new_manager)):
        out += _congestion_adjustments(congestion, side) + _manager_adjustments(manager, side)
    return out


def fixture_adjustments(ctx: FixtureContext, news: tuple[PlayerFact, ...], facts: FixtureFacts | None,
                        positions: Mapping[int, int], as_of: date) -> tuple[Adjustment, ...]:
    match_day = ctx.fixture.kickoff.date()
    out: list[Adjustment] = []
    for side, team, squad in (("home", ctx.fixture.home, ctx.home_squad), ("away", ctx.fixture.away, ctx.away_squad)):
        by_id = {p.id: p for p in squad}
        for fact in news:
            if fact.team == team and fact.player_id in by_id and _is_recent(fact, as_of):
                out += _player_adjustments(fact, by_id[fact.player_id], squad, side, positions, match_day)
    if facts is not None:
        out += _fixture_fact_adjustments(facts)
    return tuple(out)


def _clip(value: float) -> float:
    return min(1 + MAX_ADJUSTMENT, max(1 - MAX_ADJUSTMENT, value))


def side_multiplier(adjustments: tuple[Adjustment, ...], side: str) -> float:
    product = 1.0
    for a in adjustments:
        if a.side == side:
            product *= 1 + a.effect
    return _clip(product)


def gameweek_adjustments(contexts: tuple[FixtureContext, ...], news: tuple[PlayerFact, ...],
                         facts: Mapping[int, FixtureFacts], positions: Mapping[int, int],
                         as_of: date) -> dict[int, FixtureAdjustment]:
    per_fixture = {c.fixture.id: fixture_adjustments(c, news, facts.get(c.fixture.id), positions, as_of)
                   for c in contexts}
    raw = {fid: (side_multiplier(a, "home"), side_multiplier(a, "away")) for fid, a in per_fixture.items()}
    before = sum(c.rates[0] + c.rates[1] for c in contexts)
    after = sum(c.rates[0] * raw[c.fixture.id][0] + c.rates[1] * raw[c.fixture.id][1] for c in contexts)
    scale = before / after if after > 0 else 1.0
    return {
        fid: FixtureAdjustment(fid, _clip(home * scale), _clip(away * scale), per_fixture[fid])
        for fid, (home, away) in raw.items()
    }
