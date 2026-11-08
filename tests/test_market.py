import pytest

from football_ml.config import MAX_GOALS
from football_ml.evaluate.forecast import outcome_probabilities
from football_ml.model.dixon_coles import matrix_from_rates
from football_ml.model.market import market_rates, over_probability, shin_probabilities
from football_ml.sources.football_data import MatchOdds


def test_shin_without_margin_returns_the_implied_probabilities():
    assert shin_probabilities((2.0, 4.0, 4.0)) == pytest.approx((0.5, 0.25, 0.25))


def test_shin_removes_more_margin_from_longshots():
    odds = (1.30, 5.50, 9.00)
    shin = shin_probabilities(odds)
    implied = [1 / o for o in odds]
    proportional = [q / sum(implied) for q in implied]
    assert sum(shin) == pytest.approx(1.0)
    assert shin[2] < proportional[2] and shin[0] > proportional[0]


def test_shin_rejects_bad_odds():
    with pytest.raises(ValueError):
        shin_probabilities((1.0, 3.0, 4.0))
    with pytest.raises(ValueError):
        shin_probabilities((2.0,))


@pytest.mark.parametrize("lam,mu,rho", [(1.6, 1.1, -0.05), (2.7, 0.5, -0.04), (0.9, 1.4, 0.0)])
def test_market_rates_recover_the_rates_behind_the_probabilities(lam, mu, rho):
    m = matrix_from_rates(lam, mu, rho, MAX_GOALS)
    home, draw, away = outcome_probabilities(m)
    fitted = market_rates((home, draw, away), over_probability(m), rho)
    assert fitted == pytest.approx((lam, mu), rel=1e-3)


def test_market_rates_without_over_under_still_fit_the_1x2():
    m = matrix_from_rates(1.8, 0.9, -0.05, MAX_GOALS)
    lam, mu = market_rates(outcome_probabilities(m), None, -0.05)
    fitted = outcome_probabilities(matrix_from_rates(lam, mu, -0.05, MAX_GOALS))
    assert fitted == pytest.approx(outcome_probabilities(m), abs=1e-3)


def test_rates_from_odds_uses_shin_on_both_markets():
    from football_ml.model.market import rates_from_odds
    odds = MatchOdds(home=1.50, draw=4.40, away=6.50, over_2_5=1.65, under_2_5=2.30)
    lam, mu = rates_from_odds(odds, rho=-0.04)
    assert lam > 1.5 > mu
    assert rates_from_odds(MatchOdds(1.5, 4.4, 6.5, None, None), rho=-0.04)[0] > mu
