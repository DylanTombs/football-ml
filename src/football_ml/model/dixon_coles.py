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

from football_ml.config import RATING_L2_PENALTY, REFERENCE_LEAGUE, RHO_BOUNDS, TIME_DECAY_XI
from football_ml.model.structure import Structure, design, extra_terms

_EPS = 1e-10


@dataclass(frozen=True)
class DixonColesParams:
    teams: tuple[str, ...]
    attack: tuple[float, ...]
    defence: tuple[float, ...]
    home_advantage: float
    rho: float
    extras: tuple[tuple[str, float], ...] = ()  # plan 07 structure terms, by name

    def _index(self, team: str) -> int:
        try:
            return self.teams.index(team)
        except ValueError:
            raise KeyError(f"Team '{team}' not in model; known: {sorted(self.teams)}") from None

    def attack_of(self, team: str) -> float:
        return self.attack[self._index(team)]

    def defence_of(self, team: str) -> float:
        return self.defence[self._index(team)]

    def expected_goals(
        self, home: str, away: str, league: str = REFERENCE_LEAGUE, season: str | None = None,
    ) -> tuple[float, float]:
        extra_lam, extra_mu = extra_terms(dict(self.extras), league, season)
        lam = np.exp(self.attack_of(home) + self.defence_of(away) + self.home_advantage + extra_lam)
        mu = np.exp(self.attack_of(away) + self.defence_of(home) + extra_mu)
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
    return attack, theta[n : 2 * n], theta[2 * n], theta[2 * n + 1], theta[2 * n + 2:]


def _rates(theta, n, h_idx, a_idx, x_lam, x_mu):
    attack, defence, home, rho, extra = _unpack(theta, n)
    log_lam = attack[h_idx] + defence[a_idx] + home
    log_mu = attack[a_idx] + defence[h_idx]
    if extra.size:
        log_lam, log_mu = log_lam + x_lam @ extra, log_mu + x_mu @ extra
    return attack, defence, rho, extra, np.exp(log_lam), np.exp(log_mu)


def _objective(theta, n, h_idx, a_idx, hg, ag, weights, l2, x_lam=None, x_mu=None, pen=None):
    attack, defence, rho, extra, lam, mu = _rates(theta, n, h_idx, a_idx, x_lam, x_mu)
    tau = np.clip(_tau(hg, ag, lam, mu, rho), _EPS, None)
    log_lik = np.log(tau) + poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
    penalty = l2 * (np.sum(attack**2) + np.sum(defence**2)) + (np.sum(pen * extra**2) if extra.size else 0.0)
    return -np.sum(weights * log_lik) + penalty


def _tau_gradients(x, y, lam, mu, rho, tau):
    """d log τ / d log λ, d log τ / d log μ and d log τ / dρ per match."""
    zero_zero, zero_one, one_zero, one_one = (x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)
    d_lam = np.select([zero_zero, zero_one], [-lam * mu * rho / tau, lam * rho / tau], default=0.0)
    d_mu = np.select([zero_zero, one_zero], [-lam * mu * rho / tau, mu * rho / tau], default=0.0)
    d_rho = np.select([zero_zero, zero_one, one_zero, one_one], [-lam * mu / tau, lam / tau, mu / tau, -1 / tau], default=0.0)
    return d_lam, d_mu, d_rho


def _gradient(theta, n, h_idx, a_idx, hg, ag, weights, l2, x_lam=None, x_mu=None, pen=None):
    """Exact gradient of `_objective`; makes each fit tens of times faster."""
    attack, defence, rho, extra, lam, mu = _rates(theta, n, h_idx, a_idx, x_lam, x_mu)
    tau = np.clip(_tau(hg, ag, lam, mu, rho), _EPS, None)
    t_lam, t_mu, t_rho = _tau_gradients(hg, ag, lam, mu, rho, tau)
    g_lam = -weights * (hg - lam + t_lam)  # d(-weighted log-lik) / d log λ
    g_mu = -weights * (ag - mu + t_mu)
    g_attack = np.bincount(h_idx, g_lam, n) + np.bincount(a_idx, g_mu, n) + 2 * l2 * attack
    g_defence = np.bincount(a_idx, g_lam, n) + np.bincount(h_idx, g_mu, n) + 2 * l2 * defence
    return np.concatenate([
        g_attack - g_attack.mean(),  # through the sum-to-zero centring
        g_defence,
        [g_lam.sum(), -np.sum(weights * t_rho)],
        (x_lam.T @ g_lam + x_mu.T @ g_mu + 2 * pen * extra) if extra.size else [],
    ])


def time_weights(dates: pd.Series, reference_date: pd.Timestamp, xi: float) -> np.ndarray:
    days_ago = (reference_date - dates).dt.days.to_numpy()
    return np.exp(-xi * np.clip(days_ago, 0, None))


def fit(
    results: pd.DataFrame,
    reference_date: pd.Timestamp,
    xi: float = TIME_DECAY_XI,
    l2: float = RATING_L2_PENALTY,
    structure: Structure = Structure(),
    goal_columns: tuple[str, str] = ("home_goals", "away_goals"),
    rho_bounds: tuple[float, float] = RHO_BOUNDS,
) -> DixonColesParams:
    """Fit on matches played before `reference_date`. `goal_columns` and
    `rho_bounds` let the same model be fitted to other counts (plan 07 shots,
    with ρ fixed at 0)."""
    played = results[results["date"] < reference_date].dropna(subset=list(goal_columns))
    if played.empty:
        raise ValueError("Cannot fit Dixon-Coles: no matches before reference date")

    teams = tuple(sorted(set(played["home"]) | set(played["away"])))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    extras = design(played, structure)
    args = (
        n,
        played["home"].map(index).to_numpy(),
        played["away"].map(index).to_numpy(),
        played[goal_columns[0]].to_numpy(),
        played[goal_columns[1]].to_numpy(),
        time_weights(played["date"], reference_date, xi),
        l2,
        extras.x_lam,
        extras.x_mu,
        extras.penalties,
    )
    theta0 = np.concatenate([np.zeros(2 * n), [0.25, 0.0], np.zeros(len(extras.names))])
    bounds = [(None, None)] * (2 * n + 1) + [rho_bounds] + [(None, None)] * len(extras.names)
    result = minimize(_objective, theta0, args=args, jac=_gradient, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"Dixon-Coles optimisation failed: {result.message}")

    attack, defence, home, rho, extra = _unpack(result.x, n)
    return DixonColesParams(
        teams=teams,
        attack=tuple(float(a) for a in attack),
        defence=tuple(float(d) for d in defence),
        home_advantage=float(home),
        rho=float(rho),
        extras=tuple((name, float(v)) for name, v in zip(extras.names, extra)),
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
