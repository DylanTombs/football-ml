"""Pick who scores the predicted goals."""

from dataclasses import dataclass

from football_ml.config import SCORER_REPEAT_DECAY, SCORER_XG_WEIGHT


@dataclass(frozen=True)
class Player:
    id: int
    name: str
    team: str
    xg: float
    goals: int
    minutes: int
    available: bool


def scoring_weight(player: Player) -> float:
    """Season threat: blend of expected goals and actual goals."""
    return SCORER_XG_WEIGHT * player.xg + (1 - SCORER_XG_WEIGHT) * player.goals


def assign_scorers(
    squad: tuple[Player, ...],
    n_goals: int,
    repeat_decay: float = SCORER_REPEAT_DECAY,
) -> tuple[Player, ...]:
    """Greedy most-likely allocation: each goal goes to the player with the
    highest weight / (1 + decay * goals already given to them)."""
    eligible = tuple(
        p for p in squad if p.available and p.minutes > 0 and scoring_weight(p) > 0
    )
    if not eligible:
        return ()

    scorers: tuple[Player, ...] = ()
    for _ in range(n_goals):
        tally = {p.id: sum(s.id == p.id for s in scorers) for p in eligible}
        best = max(eligible, key=lambda p: scoring_weight(p) / (1 + repeat_decay * tally[p.id]))
        scorers = (*scorers, best)
    return scorers
