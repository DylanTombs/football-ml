"""How much of a team's attacking threat is missing through injury or suspension."""

from dataclasses import dataclass

from football_ml.predict.scorers import Player


@dataclass(frozen=True)
class MissingPlayer:
    name: str
    xg_share: float  # share of team xG lost: xg × (1 − availability) / team xG


@dataclass(frozen=True)
class TeamAvailability:
    missing_share: float
    missing: tuple[MissingPlayer, ...]


_FULL_SQUAD = TeamAvailability(missing_share=0.0, missing=())


def team_availability(squad: tuple[Player, ...]) -> TeamAvailability:
    members = tuple(p for p in squad if not p.departed)
    team_xg = sum(p.xg for p in members)
    if team_xg <= 0:
        return _FULL_SQUAD

    shares = tuple(
        MissingPlayer(p.name, p.xg * (1 - p.availability) / team_xg)
        for p in members
        if p.availability < 1 and p.xg > 0
    )
    return TeamAvailability(
        missing_share=min(1.0, sum(m.xg_share for m in shares)),  # guard float drift
        missing=tuple(sorted(shares, key=lambda m: m.xg_share, reverse=True)),
    )


def adjust_rate(rate: float, missing_share: float, impact: float) -> float:
    """λ_adj = λ × (1 − impact × missing_share)."""
    if not 0 <= missing_share <= 1:
        raise ValueError(f"missing_share must be in [0, 1], got {missing_share}")
    if not 0 <= impact <= 1:
        raise ValueError(f"injury impact must be in [0, 1], got {impact}")
    return rate * (1 - impact * missing_share)
