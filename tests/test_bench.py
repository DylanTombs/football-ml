import math

import numpy as np
import pandas as pd
import pytest

from football_ml.bench import Forecast, forecast_metrics, walk_forward, within_one_se
from football_ml.evaluate.forecast import exact_log_loss, outcome_log_loss, outcome_probabilities, rps


def test_outcome_probabilities_and_rps():
    m = np.array([[0.1, 0.1], [0.5, 0.3]])  # home win 0.5, draw 0.4, away 0.1
    assert outcome_probabilities(m) == pytest.approx((0.5, 0.4, 0.1))
    assert rps((0.5, 0.4, 0.1), "H") == pytest.approx(((0.5 - 1) ** 2 + (0.9 - 1) ** 2) / 2)
    assert rps((1.0, 0.0, 0.0), "H") == 0.0
    assert outcome_log_loss((0.5, 0.4, 0.1), "D") == pytest.approx(-math.log(0.4))


def test_exact_log_loss_floors_impossible_scores():
    m = np.array([[0.25, 0.25], [0.5, 0.0]])
    assert exact_log_loss(m, 1, 0) == pytest.approx(-math.log(0.5))
    assert exact_log_loss(m, 1, 1) == pytest.approx(-math.log(1e-10))
    assert exact_log_loss(m, 7, 0) == pytest.approx(-math.log(1e-10))  # beyond the grid


def _season(code, start, teams=("Strong", "MidA", "MidB", "Weak"), league="E0", rounds=10):
    rows = []
    for r in range(rounds):
        for i, h in enumerate(teams):
            for a in teams[i + 1:]:
                home, away = (h, a) if r % 2 == 0 else (a, h)
                strong = 3 if home == "Strong" else 1
                rows.append({"date": pd.Timestamp(start) + pd.Timedelta(days=7 * r), "home": home, "away": away,
                             "home_goals": strong, "away_goals": 1 if away == "Strong" else 0,
                             "league": league, "season": code})
    return rows


RESULTS = pd.DataFrame(_season("2021", "2020-09-05") + _season("2122", "2021-08-14")
                       + _season("2122", "2021-08-15", teams=("ChampA", "ChampB"), league="E1"))


def test_walk_forward_forecasts_only_test_season_league_matches():
    forecasts = walk_forward(RESULTS, ("2122",), xi=0.0019, l2=0.5)
    assert len(forecasts) == 60  # 6 pairings × 10 rounds, E0 only
    assert all(f.season == "2122" for f in forecasts)
    strong_home = [f for f in forecasts if f.home == "Strong"]
    assert all(f.lam > f.mu for f in strong_home)


def test_walk_forward_never_sees_the_future():
    first = walk_forward(RESULTS, ("2122",), xi=0.0019, l2=0.5)
    changed = RESULTS.copy()
    late = changed["date"] >= pd.Timestamp("2021-10-01")
    changed.loc[late, "home_goals"] = 9
    second = walk_forward(changed, ("2122",), xi=0.0019, l2=0.5)
    before = [(a.lam, b.lam) for a, b in zip(first, second) if a.date < pd.Timestamp("2021-09-27")]
    assert before and all(x == y for x, y in before)


def test_forecast_metrics_reports_scores_and_calibration():
    f = Forecast(pd.Timestamp("2021-08-14"), "2122", "A", "B", 1, 0, lam=1.5, mu=1.0, rho=0.0)
    metrics = forecast_metrics((f, f))
    assert metrics["matches"] == 2
    assert metrics["predicted_goals"] == pytest.approx(5.0, rel=1e-3)
    assert metrics["actual_goals"] == 2
    assert 0 < metrics["predicted_draws"] < 2 and metrics["actual_draws"] == 0
    assert metrics["exact_log_loss"] > metrics["outcome_log_loss"] > 0
    assert metrics["outcome_points"] in (0, 2, 6)  # modal pick, twice


def test_market_blend_moves_rates_towards_the_market():
    f = Forecast(pd.Timestamp("2021-08-14"), "2122", "A", "B", 1, 0, lam=1.0, mu=1.0, rho=0.0,
                 market_lam=2.0, market_mu=0.5)
    assert f.blended(1.0) == (1.0, 1.0)
    assert f.blended(0.25) == pytest.approx((1.75, 0.625))
    no_market = Forecast(pd.Timestamp("2021-08-14"), "2122", "A", "B", 1, 0, lam=1.0, mu=1.0, rho=0.0)
    assert no_market.blended(0.0) == (1.0, 1.0)  # falls back to the model


def test_within_one_se_picks_ties_with_the_best():
    best = np.array([1.0, 1.0, 1.0, 1.0])
    close = np.array([1.01, 0.99, 1.02, 1.0])
    worse = np.array([1.5, 1.6, 1.4, 1.5])
    assert within_one_se({"best": best, "close": close, "worse": worse}) == {"best", "close"}


def test_tune_grid_keeps_current_settings_when_within_one_se():
    from football_ml.bench import tune
    table = tune(RESULTS, ("2122",), xis=(0.0019, 0.004), l2s=(0.5, 2.0), current=(0.0019, 0.5))
    assert len(table["rows"]) == 4
    assert {(r["xi"], r["l2"]) for r in table["rows"]} == {(0.0019, 0.5), (0.0019, 2.0), (0.004, 0.5), (0.004, 2.0)}
    best = min(table["rows"], key=lambda r: r["exact_log_loss"])
    assert (table["best"]["xi"], table["best"]["l2"]) == (best["xi"], best["l2"])
    chosen = (table["chosen"]["xi"], table["chosen"]["l2"])
    assert chosen == (0.0019, 0.5) if table["rows"][0]["within_one_se"] else chosen == (best["xi"], best["l2"])


def test_attach_market_rates_matches_odds_by_teams_and_date():
    from football_ml.bench import attach_market
    f = Forecast(pd.Timestamp("2021-08-14"), "2122", "Strong", "Weak", 3, 0, lam=2.0, mu=0.6, rho=-0.05)
    other = Forecast(pd.Timestamp("2021-08-14"), "2122", "MidA", "MidB", 1, 1, lam=1.2, mu=1.1, rho=-0.05)
    odds = pd.DataFrame([{"date": pd.Timestamp("2021-08-14"), "home": "Strong", "away": "Weak",
                          "odds_home": 1.3, "odds_draw": 5.5, "odds_away": 9.0, "odds_over": 1.7, "odds_under": 2.2}])
    with_odds, without = attach_market((f, other), odds)
    assert with_odds.market_lam > with_odds.market_mu
    assert without.market_lam is None


def test_tune_blend_reports_each_weight():
    from football_ml.bench import tune_blend
    f = Forecast(pd.Timestamp("2021-08-14"), "2122", "A", "B", 2, 0, lam=1.0, mu=1.0, rho=0.0,
                 market_lam=2.0, market_mu=0.5)
    table = tune_blend((f, f, f), weights=(0.0, 0.5, 1.0))
    assert [r["model_weight"] for r in table["rows"]] == [0.0, 0.5, 1.0]
    assert table["best"]["model_weight"] == 0.0  # the market saw the 2-0 coming
    assert table["with_odds"] == 3
