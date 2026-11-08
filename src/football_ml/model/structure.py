"""Plan 07 extra parameters for Dixon-Coles, as design-matrix columns.

Each extra θ_k adds X_λ[i, k] θ_k to log λ_i and X_μ[i, k] θ_k to log μ_i, with
its own ridge penalty:
  per_league   league:L  scoring level of league L vs the reference (λ and μ)
               home:L    home advantage of league L vs the reference (λ only)
  season_home  home_season:S  home advantage in season S vs the shared one (λ only)
  season_level season:S  scoring level of season S (λ and μ)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_ml.config import REFERENCE_LEAGUE, SEASON_HOME_L2, SEASON_LEVEL_L2, SHOTS_WEIGHT


@dataclass(frozen=True)
class Structure:
    per_league: bool = False
    season_home: bool = False
    season_home_l2: float = SEASON_HOME_L2
    season_level: bool = False
    season_level_l2: float = SEASON_LEVEL_L2
    shots_weight: float = 0.0  # blend with a shots-on-target model (model/shots.py)


@dataclass(frozen=True)
class Design:
    names: tuple[str, ...]
    x_lam: np.ndarray  # matches × extras
    x_mu: np.ndarray
    penalties: np.ndarray


def label(structure: Structure) -> str:
    """Short name for reports and bench files; "base" for plain Dixon-Coles."""
    parts = (["per_league"] if structure.per_league else []) + (
        [f"season_home{structure.season_home_l2}"] if structure.season_home else []) + (
        [f"season_level{structure.season_level_l2}"] if structure.season_level else []) + (
        [f"shots{structure.shots_weight}"] if structure.shots_weight else [])
    return "-".join(parts) or "base"


# Plan 07 shadow variants: every predict also writes dixon-coles-<name> with the
# same picks and scorers, so only the team model differs. Promote one only
# after a few weeks of forward results (decided 2026-11-08).
VARIANTS: dict[str, Structure] = {
    "per-league": Structure(per_league=True),
    "season-home": Structure(season_home=True),
    "season-level": Structure(season_level=True),
    "shots": Structure(shots_weight=SHOTS_WEIGHT),
}


def _require(played: pd.DataFrame, column: str) -> None:
    if column not in played.columns:
        raise ValueError(f"This model structure needs a '{column}' column in the results")


def design(played: pd.DataFrame, structure: Structure) -> Design:
    columns: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    if structure.per_league:
        _require(played, "league")
        for league in sorted(set(played["league"]) - {REFERENCE_LEAGUE}):
            in_league = (played["league"] == league).to_numpy(dtype=float)
            columns += [(f"league:{league}", in_league, in_league, 0.0),
                        (f"home:{league}", in_league, np.zeros_like(in_league), 0.0)]
    if structure.season_home:
        _require(played, "season")
        for season in sorted(set(played["season"])):
            in_season = (played["season"] == season).to_numpy(dtype=float)
            columns.append((f"home_season:{season}", in_season, np.zeros_like(in_season), structure.season_home_l2))
    if structure.season_level:
        _require(played, "season")
        for season in sorted(set(played["season"])):
            in_season = (played["season"] == season).to_numpy(dtype=float)
            columns.append((f"season:{season}", in_season, in_season, structure.season_level_l2))
    m = len(played)
    if not columns:
        return Design((), np.zeros((m, 0)), np.zeros((m, 0)), np.zeros(0))
    names, x_lam, x_mu, pens = zip(*columns)
    return Design(tuple(names), np.column_stack(x_lam), np.column_stack(x_mu), np.array(pens))


def extra_terms(extras: dict[str, float], league: str, season: str | None) -> tuple[float, float]:
    """(added to log λ, added to log μ) for a match. An unknown season uses the latest one fitted."""
    seasons = sorted({k.split(":", 1)[1] for k in extras if k.startswith(("home_season:", "season:"))})
    if seasons and (season is None or season not in seasons):
        season = seasons[-1]
    level = extras.get(f"league:{league}", 0.0) + extras.get(f"season:{season}", 0.0)
    home = extras.get(f"home:{league}", 0.0) + extras.get(f"home_season:{season}", 0.0)
    return level + home, level
