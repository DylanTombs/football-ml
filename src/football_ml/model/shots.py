"""Plan 07 shots variant: blend the goals model with the same model fitted on
shots on target, which are less noisy than goals (Wheatcroft 2020).

    λ_shots = expected shots on target × goals per shot on target (recent, weighted)
    log λ   = (1 − w) log λ_goals + w log λ_shots

ρ (the low-score correction) always comes from the goals model."""

import math
from dataclasses import dataclass, replace

import pandas as pd

from football_ml.config import REFERENCE_LEAGUE
from football_ml.model.dixon_coles import DixonColesParams, fit, time_weights
from football_ml.model.structure import Structure

SHOT_COLUMNS = ("home_sot", "away_sot")


@dataclass(frozen=True)
class TeamModel:
    """Same interface as DixonColesParams for the rest of the pipeline."""
    goals: DixonColesParams
    shots: DixonColesParams | None = None
    conversion: float = 0.0  # goals per shot on target
    shots_weight: float = 0.0

    @property
    def rho(self) -> float:
        return self.goals.rho

    @property
    def home_advantage(self) -> float:
        return self.goals.home_advantage

    @property
    def extras(self) -> tuple[tuple[str, float], ...]:
        return self.goals.extras

    def expected_goals(
        self, home: str, away: str, league: str = REFERENCE_LEAGUE, season: str | None = None,
    ) -> tuple[float, float]:
        goal_rates = self.goals.expected_goals(home, away, league, season)
        if self.shots is None or not self.shots_weight:
            return goal_rates
        try:
            shot_rates = self.shots.expected_goals(home, away, league, season)
        except KeyError:  # no shots data for a team: the goals model alone
            return goal_rates
        w = self.shots_weight
        return tuple(math.exp((1 - w) * math.log(g) + w * math.log(self.conversion * s))
                     for g, s in zip(goal_rates, shot_rates))


def _conversion(results: pd.DataFrame, cutoff: pd.Timestamp, xi: float) -> float:
    played = results[results["date"] < cutoff].dropna(subset=list(SHOT_COLUMNS))
    weights = time_weights(played["date"], cutoff, xi)
    goals = played["home_goals"] + played["away_goals"]
    shots = played["home_sot"] + played["away_sot"]
    return float((weights * goals).sum() / (weights * shots).sum())


def fit_team_model(
    results: pd.DataFrame, cutoff: pd.Timestamp, xi: float, l2: float, structure: Structure,
) -> TeamModel:
    goals = fit(results, reference_date=cutoff, xi=xi, l2=l2, structure=structure)
    if not structure.shots_weight:
        return TeamModel(goals)
    missing = [c for c in SHOT_COLUMNS if c not in results.columns or results[c].isna().all()]
    if missing:
        raise ValueError(f"The shots variant needs shots on target ({', '.join(missing)}) in the results")
    shots = fit(results, reference_date=cutoff, xi=xi, l2=l2, structure=replace(structure, shots_weight=0.0),
                goal_columns=SHOT_COLUMNS, rho_bounds=(0.0, 0.0))
    return TeamModel(goals, shots, _conversion(results, cutoff, xi), structure.shots_weight)
