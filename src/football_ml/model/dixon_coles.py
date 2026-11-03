"""Dixon-Coles (1997) bivariate Poisson model with time decay.

home goals ~ Poisson(exp(attack_h + defence_a + home_advantage))
away goals ~ Poisson(exp(attack_a + defence_h))
with the tau correction on 0-0, 1-0, 0-1, 1-1 controlled by rho.
Higher `defence` = leakier defence.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from football_ml.config import RATING_L2_PENALTY, RHO_BOUNDS, TIME_DECAY_XI

_EPS = 1e-10


@dataclass(frozen=True)
class DixonColesParams:
    teams: tuple[str, ...]
    attack: tuple[float, ...]
    defence: tuple[float, ...]
    home_advantage: float
    rho: float

    def _index(self, team: str) -> int:
        try:
            return self.teams.index(team)
        except ValueError:
            raise KeyError(f"Team '{team}' not in model; known: {sorted(self.teams)}") from None

    def attack_of(self, team: str) -> float:
        return self.attack[self._index(team)]

    def defence_of(self, team: str) -> float:
        return self.defence[self._index(team)]

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        lam = np.exp(self.attack_of(home) + self.defence_of(away) + self.home_advantage)
        mu = np.exp(self.attack_of(away) + self.defence_of(home))
        return float(lam), float(mu)


def _tau(x, y, lam, mu, rho):
    return np.select(
        [(x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)],
        [1 - lam * mu * rho, 1 + lam * rho, 1 + mu * rho, np.full_like(lam, 1 - rho)],
        default=1.0,
    )


def _unpack(theta: np.ndarray, n: int):
    raw_attack = theta[:n]
    attack = raw_attack - raw_attack.mean()  # sum-to-zero identifiability
    return attack, theta[n : 2 * n], theta[2 * n], theta[2 * n + 1]


def _objective(theta, n, h_idx, a_idx, hg, ag, weights, l2):
    attack, defence, home, rho = _unpack(theta, n)
    lam = np.exp(attack[h_idx] + defence[a_idx] + home)
    mu = np.exp(attack[a_idx] + defence[h_idx])
    tau = np.clip(_tau(hg, ag, lam, mu, rho), _EPS, None)
    log_lik = np.log(tau) + poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
    penalty = l2 * (np.sum(attack**2) + np.sum(defence**2))
    return -np.sum(weights * log_lik) + penalty


def time_weights(dates: pd.Series, reference_date: pd.Timestamp, xi: float) -> np.ndarray:
    days_ago = (reference_date - dates).dt.days.to_numpy()
    return np.exp(-xi * np.clip(days_ago, 0, None))


def fit(
    results: pd.DataFrame,
    reference_date: pd.Timestamp,
    xi: float = TIME_DECAY_XI,
    l2: float = RATING_L2_PENALTY,
) -> DixonColesParams:
    """Fit on matches played before `reference_date`."""
    played = results[results["date"] < reference_date]
    if played.empty:
        raise ValueError("Cannot fit Dixon-Coles: no matches before reference date")

    teams = tuple(sorted(set(played["home"]) | set(played["away"])))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    args = (
        n,
        played["home"].map(index).to_numpy(),
        played["away"].map(index).to_numpy(),
        played["home_goals"].to_numpy(),
        played["away_goals"].to_numpy(),
        time_weights(played["date"], reference_date, xi),
        l2,
    )
    theta0 = np.concatenate([np.zeros(2 * n), [0.25, 0.0]])
    bounds = [(None, None)] * (2 * n + 1) + [RHO_BOUNDS]
    result = minimize(_objective, theta0, args=args, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"Dixon-Coles optimisation failed: {result.message}")

    attack, defence, home, rho = _unpack(result.x, n)
    return DixonColesParams(
        teams=teams,
        attack=tuple(float(a) for a in attack),
        defence=tuple(float(d) for d in defence),
        home_advantage=float(home),
        rho=float(rho),
    )


def matrix_from_rates(lam: float, mu: float, rho: float, max_goals: int) -> np.ndarray:
    """P(home scores i, away scores j) for i, j in 0..max_goals, given the two
    expected-goal rates. Kept separate so rates can be adjusted (e.g. injuries)."""
    goals = np.arange(max_goals + 1)
    independent = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    x, y = np.meshgrid(goals, goals, indexing="ij")
    correction = _tau(x, y, np.full(x.shape, lam), np.full(x.shape, mu), rho)
    matrix = np.clip(independent * correction, 0.0, None)
    return matrix / matrix.sum()  # renormalise mass lost by truncating at max_goals


def score_matrix(params: DixonColesParams, home: str, away: str, max_goals: int) -> np.ndarray:
    lam, mu = params.expected_goals(home, away)
    return matrix_from_rates(lam, mu, params.rho, max_goals)
