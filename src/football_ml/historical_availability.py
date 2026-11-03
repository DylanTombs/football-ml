"""Who was missing in a past gameweek, rebuilt from FPL minutes.

FPL keeps no history of injury news, so the backtest infers absences from
minutes played. See AVAILABILITY_PROXIES in config for the options."""

from collections.abc import Mapping
from dataclasses import replace

from football_ml.config import AVAILABILITY_PROXIES
from football_ml.predict.scorers import Player
from football_ml.sources.fpl import team_names

Minutes = Mapping[int, Mapping[int, int]]  # gameweek -> player id -> minutes


def team_gameweeks(raw_fixtures: list[dict], bootstrap: dict) -> dict[str, tuple[int, ...]]:
    """Gameweeks in which each team has a finished match, ascending."""
    names = team_names(bootstrap)
    played: dict[str, set[int]] = {}
    for f in raw_fixtures:
        if not f.get("finished") or f.get("event") is None:
            continue
        for side in ("team_h", "team_a"):
            played.setdefault(names[f[side]], set()).add(f["event"])
    return {team: tuple(sorted(gws)) for team, gws in played.items()}


def _reference_gameweek(team_gws: tuple[int, ...], gameweek: int, proxy: str) -> int | None:
    if proxy == "oracle":
        return gameweek if gameweek in team_gws else None
    earlier = [gw for gw in team_gws if gw < gameweek]
    return earlier[-1] if earlier else None


def _was_missing(player: Player, minutes: Minutes, team_gws: tuple[int, ...], reference: int) -> bool:
    """0 minutes in the reference gameweek, having played for the team before it."""
    played_before = sum(minutes.get(gw, {}).get(player.id, 0) for gw in team_gws if gw < reference)
    return played_before > 0 and minutes.get(reference, {}).get(player.id, 0) == 0


def players_as_of(
    players: tuple[Player, ...],
    minutes: Minutes,
    teams: Mapping[str, tuple[int, ...]],
    gameweek: int,
    proxy: str,
) -> tuple[Player, ...]:
    """Copies of `players` with availability as it stood before `gameweek`.
    Only `availability` changes; scorer eligibility (`available`) is left alone
    so scorer points stay comparable with the baseline."""
    if proxy not in AVAILABILITY_PROXIES:
        raise ValueError(f"Unknown availability proxy {proxy!r}; choose from {AVAILABILITY_PROXIES}")
    if proxy == "live":
        return players

    def as_of(player: Player) -> Player:
        team_gws = teams.get(player.team, ())
        reference = _reference_gameweek(team_gws, gameweek, proxy)
        missing = reference is not None and _was_missing(player, minutes, team_gws, reference)
        return replace(player, availability=0.0 if missing else 1.0, departed=False)

    return tuple(as_of(p) for p in players)
