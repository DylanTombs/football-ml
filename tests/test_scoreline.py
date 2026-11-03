import numpy as np
import pytest

from football_ml.predict.scoreline import summarise


def _matrix() -> np.ndarray:
    m = np.zeros((4, 4))
    m[1, 0] = 0.30  # 1-0
    m[2, 1] = 0.20  # 2-1
    m[1, 1] = 0.25  # 1-1
    m[0, 2] = 0.25  # 0-2
    return m


def test_summarise_picks_most_likely_scoreline():
    s = summarise(_matrix(), top_n=2)
    assert (s.home_goals, s.away_goals) == (1, 0)
    assert s.probability == pytest.approx(0.30)


def test_headline_score_matches_most_likely_outcome():
    # 1-1 is the single likeliest score, but a home win is the likeliest outcome
    m = np.zeros((4, 4))
    m[1, 1] = 0.30
    m[1, 0] = 0.25
    m[2, 1] = 0.20
    m[2, 0] = 0.10
    m[0, 1] = 0.15
    s = summarise(m, top_n=3)
    assert (s.home_goals, s.away_goals) == (1, 0)
    assert s.top_scorelines[0].home_goals == 1 and s.top_scorelines[0].away_goals == 1


def test_summarise_outcome_probabilities():
    s = summarise(_matrix(), top_n=2)
    assert s.home_win == pytest.approx(0.50)
    assert s.draw == pytest.approx(0.25)
    assert s.away_win == pytest.approx(0.25)


def test_summarise_top_scorelines_sorted():
    s = summarise(_matrix(), top_n=3)
    assert [(t.home_goals, t.away_goals) for t in s.top_scorelines][0] == (1, 0)
    probs = [t.probability for t in s.top_scorelines]
    assert probs == sorted(probs, reverse=True)
    assert len(probs) == 3


def test_summarise_expected_goals():
    s = summarise(_matrix(), top_n=1)
    assert s.expected_home_goals == pytest.approx(0.3 + 0.4 + 0.25)
    assert s.expected_away_goals == pytest.approx(0.2 + 0.25 + 0.5)
