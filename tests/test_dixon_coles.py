from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from football_ml.model.dixon_coles import fit, score_matrix


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
