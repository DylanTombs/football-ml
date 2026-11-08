from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from football_ml.model.dixon_coles import fit, matrix_from_rates, score_matrix


def _synthetic_results(n_rounds: int = 12) -> pd.DataFrame:
    """Strong beats everyone, Weak loses to everyone, Mid teams draw."""
    teams = ["Strong", "MidA", "MidB", "Weak"]
    scores = {
        ("Strong", "Weak"): (4, 0), ("Weak", "Strong"): (0, 3),
        ("Strong", "MidA"): (2, 0), ("MidA", "Strong"): (1, 2),
        ("Strong", "MidB"): (3, 1), ("MidB", "Strong"): (0, 2),
        ("MidA", "Weak"): (2, 0), ("Weak", "MidA"): (1, 1),
        ("MidB", "Weak"): (2, 1), ("Weak", "MidB"): (0, 1),
        ("MidA", "MidB"): (1, 1), ("MidB", "MidA"): (1, 1),
    }
    start = date(2025, 8, 1)
    rows = [
        {"date": pd.Timestamp(start + timedelta(days=7 * r)), "home": h, "away": a,
         "home_goals": hg, "away_goals": ag}
        for r in range(n_rounds)
        for (h, a), (hg, ag) in scores.items()
        if h in teams and a in teams
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def params():
    return fit(_synthetic_results(), reference_date=pd.Timestamp("2026-01-01"))


def test_fit_ranks_strong_team_above_weak(params):
    assert params.attack_of("Strong") > params.attack_of("Weak")
    assert params.defence_of("Strong") < params.defence_of("Weak")


def test_fit_learns_positive_home_advantage(params):
    assert params.home_advantage > 0


def test_attack_ratings_are_centred(params):
    assert abs(sum(params.attack)) < 1e-6


def test_score_matrix_is_a_probability_distribution(params):
    matrix = score_matrix(params, "Strong", "Weak", max_goals=10)
    assert matrix.shape == (11, 11)
    assert np.all(matrix >= 0)
    assert matrix.sum() == pytest.approx(1.0, abs=1e-3)


def test_score_matrix_favours_strong_team(params):
    matrix = score_matrix(params, "Strong", "Weak", max_goals=10)
    home_win = np.tril(matrix, -1).sum()
    away_win = np.triu(matrix, 1).sum()
    assert home_win > away_win


def test_unknown_team_raises_clear_error(params):
    with pytest.raises(KeyError, match="Atlantis"):
        score_matrix(params, "Atlantis", "Weak", max_goals=10)


def test_fit_rejects_empty_results():
    empty = pd.DataFrame(columns=["date", "home", "away", "home_goals", "away_goals"])
    with pytest.raises(ValueError, match="no matches"):
        fit(empty, reference_date=pd.Timestamp("2026-01-01"))


def test_score_matrix_equals_matrix_from_its_rates(params):
    lam, mu = params.expected_goals("Strong", "Weak")
    expected = matrix_from_rates(lam, mu, params.rho, max_goals=10)
    np.testing.assert_allclose(score_matrix(params, "Strong", "Weak", max_goals=10), expected)


def test_matrix_from_lower_rate_shifts_mass_to_fewer_goals():
    high = matrix_from_rates(2.0, 1.0, 0.0, max_goals=10)
    low = matrix_from_rates(1.0, 1.0, 0.0, max_goals=10)
    goals = np.arange(11)
    assert (low.sum(axis=1) @ goals) < (high.sum(axis=1) @ goals)
    assert low.sum() == pytest.approx(1.0)


def test_matrix_from_zero_rate_is_valid():
    matrix = matrix_from_rates(0.0, 1.2, -0.05, max_goals=10)
    assert np.all(matrix >= 0)
    assert matrix[1:, :].sum() == pytest.approx(0.0)


def _problem(seed=0, n_teams=6, n_matches=200):
    rng = np.random.default_rng(seed)
    h = rng.integers(0, n_teams, n_matches)
    a = (h + rng.integers(1, n_teams, n_matches)) % n_teams
    hg, ag = rng.poisson(1.5, n_matches), rng.poisson(1.1, n_matches)
    w = rng.uniform(0.2, 1.0, n_matches)
    return n_teams, h, a, hg, ag, w


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_analytic_gradient_matches_finite_differences(seed):
    from scipy.optimize import approx_fprime
    from football_ml.model.dixon_coles import _gradient, _objective
    n, h, a, hg, ag, w = _problem(seed)
    rng = np.random.default_rng(seed + 10)
    theta = np.concatenate([rng.normal(0, 0.3, 2 * n), [0.25, -0.08]])
    args = (n, h, a, hg, ag, w, 0.5)
    numeric = approx_fprime(theta, _objective, 1e-7, *args)
    np.testing.assert_allclose(_gradient(theta, *args), numeric, rtol=1e-4, atol=1e-4)


def test_fit_with_gradient_reaches_the_same_optimum():
    """The exact gradient only makes fitting faster; the ratings stay the same."""
    from scipy.optimize import minimize
    from football_ml.model.dixon_coles import _objective
    results = _synthetic_results()
    fast = fit(results, pd.Timestamp(results["date"].max()) + pd.Timedelta(days=1))
    teams = fast.teams
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    ref = pd.Timestamp(results["date"].max()) + pd.Timedelta(days=1)
    from football_ml.model.dixon_coles import time_weights
    from football_ml.config import RATING_L2_PENALTY, TIME_DECAY_XI
    args = (n, results["home"].map(index).to_numpy(), results["away"].map(index).to_numpy(),
            results["home_goals"].to_numpy(), results["away_goals"].to_numpy(),
            time_weights(results["date"], ref, TIME_DECAY_XI), RATING_L2_PENALTY)
    slow = minimize(_objective, np.concatenate([np.zeros(2 * n), [0.25, 0.0]]), args=args, method="L-BFGS-B",
                    bounds=[(None, None)] * (2 * n + 1) + [(-0.2, 0.2)])
    assert _objective(np.concatenate([fast.attack, fast.defence, [fast.home_advantage, fast.rho]]), *args) \
        <= slow.fun + 1e-6
