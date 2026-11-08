"""Actual match results: FPL gives the score and scorer ids, ESPN gives the minutes."""

import logging
import unicodedata
from dataclasses import dataclass

from football_ml.predict.scorers import Player
from football_ml.predict.timing import GoalTime, parse_clock
from football_ml.sources.espn import GoalEvent
from football_ml.sources.fpl import parse_players, team_names

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActualGoal:
    team: str  # team credited with the goal
    scorer_id: int | None  # for own goals: the player who put it in his own net
    scorer_name: str
    minute: GoalTime | None
    own_goal: bool


@dataclass(frozen=True)
class ActualMatch:
    fixture_id: int
    gameweek: int
    home: str
    away: str
    home_goals: int
    away_goals: int
    goals: tuple[ActualGoal, ...]


_SPECIAL_LETTERS = str.maketrans({"ø": "o", "Ø": "O", "ß": "ss", "æ": "ae", "đ": "d", "ł": "l"})


def _normalise(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name.translate(_SPECIAL_LETTERS))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()


def _tokens(name: str) -> set[str]:
    return {t.strip(".") for t in _normalise(name).replace("-", " ").split() if len(t.strip(".")) > 1}


def names_match(espn_name: str, player: Player) -> bool:
    """Loose match of an ESPN display name to an FPL player. Only used against
    the scorers of one team in one match, so false positives are unlikely."""
    espn = _normalise(espn_name)
    if espn in {_normalise(player.name), _normalise(player.full_name)}:
        return True
    espn_tokens = _tokens(espn_name)
    full_tokens = _tokens(player.full_name)
    web_tokens = _tokens(player.name)
    surname = espn.split(" ")[-1].strip(".")
    return (
        (bool(full_tokens) and espn_tokens <= full_tokens)  # "Yeremy Pino" ⊂ full legal name
        or surname in full_tokens | web_tokens               # "M. Ødegaard"
        or espn.replace(" ", "").endswith(_normalise(player.name).replace(" ", ""))  # "Da Silva"
    )


def _stat(fixture: dict, identifier: str, side: str) -> list[tuple[int, int]]:
    for stat in fixture.get("stats", []):
        if stat.get("identifier") == identifier:
            return [(s["element"], s["value"]) for s in stat.get(side, [])]
    return []


def _goals_for_side(fixture, side, team, opponent, players, espn_goals) -> tuple[ActualGoal, ...]:
    """Normal goals scored by `side`, plus own goals by the *other* side's players."""
    other = "a" if side == "h" else "h"
    unused = [g for g in espn_goals if g.team == team and not g.own_goal]
    goals: list[ActualGoal] = []
    for player_id, count in _stat(fixture, "goals_scored", side):
        player = players.get(player_id)
        for _ in range(count):
            match = next((g for g in unused if player and g.scorer and names_match(g.scorer, player)), None)
            unused = [g for g in unused if g is not match]
            goals.append(ActualGoal(team, player_id, player.name if player else str(player_id),
                                    parse_clock(match.clock) if match else None, False))
    own_goal_clocks = [g.clock for g in espn_goals if g.team == team and g.own_goal]
    for player_id, count in _stat(fixture, "own_goals", other):
        player = players.get(player_id)
        for _ in range(count):
            clock = own_goal_clocks.pop(0) if own_goal_clocks else None
            goals.append(ActualGoal(team, player_id, player.name if player else str(player_id),
                                    parse_clock(clock) if clock else None, True))
    return tuple(goals)


def build_actual_matches(
    raw_fixtures: list[dict],
    bootstrap: dict,
    espn_events: tuple[GoalEvent, ...],
    gameweek: int,
) -> tuple[ActualMatch, ...]:
    """Finished (incl. provisionally) matches of a gameweek. A home/away pairing is unique per season,
    so ESPN events are joined on (home, away)."""
    teams = team_names(bootstrap)
    players = {p.id: p for p in parse_players(bootstrap)}
    matches = []
    for f in raw_fixtures:
        # `finished` waits for bonus points; the score is final at `finished_provisional`.
        if f.get("event") != gameweek or not (f.get("finished") or f.get("finished_provisional")):
            continue
        home, away = teams[f["team_h"]], teams[f["team_a"]]
        espn_goals = [e for e in espn_events if (e.home, e.away) == (home, away)]
        goals = (
            *_goals_for_side(f, "h", home, away, players, espn_goals),
            *_goals_for_side(f, "a", away, home, players, espn_goals),
        )
        if any(g.minute is None for g in goals):
            log.info("Missing goal minutes for %s v %s", home, away)
        matches.append(ActualMatch(f["id"], gameweek, home, away,
                                   int(f["team_h_score"]), int(f["team_a_score"]),
                                   tuple(sorted(goals, key=lambda g: g.minute.sort_key if g.minute else 999))))
    return tuple(matches)
