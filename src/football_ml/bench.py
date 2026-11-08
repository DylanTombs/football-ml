"""Plan 06 multi-season bench: refit Dixon-Coles before every week of past
seasons and score the forecasts with proper scoring rules.

Each week's matches (Monday to Sunday) are predicted by a fit on everything
before that Monday, so no forecast sees its own week or later."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from football_ml.config import BENCH_LEAGUE, MAX_GOALS, POINTS_CORRECT_RESULT, POINTS_EXACT_SCORE
from football_ml.evaluate.forecast import exact_log_loss, outcome_log_loss, outcome_of, outcome_probabilities, rps
from football_ml.model.dixon_coles import fit, matrix_from_rates
from football_ml.model.market import rates_from_odds
from football_ml.predict.scoreline import summarise
from football_ml.sources.football_data import odds_for

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Forecast:
    date: pd.Timestamp
    season: str
    home: str
    away: str
    home_goals: int
    away_goals: int
    lam: float
    mu: float
    rho: float
    market_lam: float | None = None
    market_mu: float | None = None

    def blended(self, model_weight: float) -> tuple[float, float]:
        """λ = w × model + (1 − w) × market; the model alone without market rates."""
        if self.market_lam is None or self.market_mu is None:
            return self.lam, self.mu
        w = model_weight
        return w * self.lam + (1 - w) * self.market_lam, w * self.mu + (1 - w) * self.market_mu


def week_start(dates: pd.Series) -> pd.Series:
    days = dates.dt.normalize()
    return days - pd.to_timedelta(days.dt.day_of_week, unit="D")


def walk_forward(
    results: pd.DataFrame, seasons: Sequence[str], xi: float, l2: float, league: str = BENCH_LEAGUE,
) -> tuple[Forecast, ...]:
    """Model rates for every `league` match of `seasons`, refitting weekly."""
    tests = results[(results["league"] == league) & results["season"].isin(seasons)]
    tests = tests.assign(cutoff=week_start(tests["date"]))
    forecasts: list[Forecast] = []
    for cutoff, week in tests.groupby("cutoff", sort=True):
        params = fit(results, reference_date=cutoff, xi=xi, l2=l2)
        for row in week.itertuples():
            try:
                lam, mu = params.expected_goals(row.home, row.away)
            except KeyError as exc:
                log.warning("Skipping %s v %s on %s: %s", row.home, row.away, row.date.date(), exc)
                continue
            forecasts.append(Forecast(row.date, row.season, row.home, row.away,
                                      int(row.home_goals), int(row.away_goals), lam, mu, params.rho))
    return tuple(forecasts)


def _scores(f: Forecast, model_weight: float) -> dict:
    lam, mu = f.blended(model_weight)
    matrix = matrix_from_rates(lam, mu, f.rho, MAX_GOALS)
    probs = outcome_probabilities(matrix)
    outcome = outcome_of(f.home_goals, f.away_goals)
    pick = summarise(matrix, 1)
    exact = (pick.home_goals, pick.away_goals) == (f.home_goals, f.away_goals)
    right = outcome_of(pick.home_goals, pick.away_goals) == outcome
    return {
        "exact_log_loss": exact_log_loss(matrix, f.home_goals, f.away_goals),
        "outcome_log_loss": outcome_log_loss(probs, outcome),
        "rps": rps(probs, outcome),
        "points": POINTS_EXACT_SCORE if exact else POINTS_CORRECT_RESULT if right else 0,
        "predicted_goals": lam + mu,
        "predicted_draw": probs[1],
    }


def per_match_log_loss(forecasts: Sequence[Forecast], model_weight: float = 1.0) -> np.ndarray:
    return np.array([_scores(f, model_weight)["exact_log_loss"] for f in forecasts])


def forecast_metrics(forecasts: Sequence[Forecast], model_weight: float = 1.0) -> dict:
    """Mean log losses and RPS, modal-pick outcome points, and calibration of
    total goals and draws (predicted vs actual)."""
    rows = [_scores(f, model_weight) for f in forecasts]
    mean = lambda key: float(np.mean([r[key] for r in rows]))  # noqa: E731
    return {
        "matches": len(rows),
        "exact_log_loss": mean("exact_log_loss"),
        "outcome_log_loss": mean("outcome_log_loss"),
        "rps": mean("rps"),
        "outcome_points": int(sum(r["points"] for r in rows)),
        "predicted_goals": float(sum(r["predicted_goals"] for r in rows)),
        "actual_goals": int(sum(f.home_goals + f.away_goals for f in forecasts)),
        "predicted_draws": float(sum(r["predicted_draw"] for r in rows)),
        "actual_draws": int(sum(f.home_goals == f.away_goals for f in forecasts)),
    }


def within_one_se(losses: Mapping[str, np.ndarray]) -> set[str]:
    """Settings whose mean per-match loss is within one standard error (of the
    paired difference) of the best."""
    best = min(losses, key=lambda name: losses[name].mean())

    def close(name: str) -> bool:
        diff = losses[name] - losses[best]
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
        return diff.mean() <= se
    return {name for name in losses if close(name)}


def run_bench(results: pd.DataFrame, seasons: Sequence[str], xi: float, l2: float) -> dict:
    forecasts = walk_forward(results, seasons, xi, l2)
    return {"label": f"xi{xi}-l2{l2}", "seasons": list(seasons), "xi": xi, "l2": l2,
            "metrics": forecast_metrics(forecasts)}


def tune(
    results: pd.DataFrame, seasons: Sequence[str], xis: Sequence[float], l2s: Sequence[float],
    current: tuple[float, float],
) -> dict:
    """Grid over (xi, l2) on exact-score log loss. Keeps `current` if it is
    within one standard error of the best, otherwise takes the best."""
    grid = [(xi, l2) for xi in xis for l2 in l2s]
    forecasts = {key: walk_forward(results, seasons, *key) for key in grid}
    losses = {key: per_match_log_loss(f) for key, f in forecasts.items()}
    close = within_one_se(losses)
    rows = [{"xi": xi, "l2": l2, **forecast_metrics(forecasts[(xi, l2)]), "within_one_se": (xi, l2) in close}
            for xi, l2 in grid]
    best = min(rows, key=lambda r: r["exact_log_loss"])
    chosen = next((r for r in rows if (r["xi"], r["l2"]) == tuple(current) and r["within_one_se"]), best)
    return {"seasons": list(seasons), "rows": rows, "best": best, "chosen": chosen}


def attach_market(forecasts: Sequence[Forecast], odds: pd.DataFrame) -> tuple[Forecast, ...]:
    """Add market rates (pre-closing odds → λ, μ) where the match has odds."""
    def with_market(f: Forecast) -> Forecast:
        match = odds_for(odds, f.home, f.away, f.date)
        if match is None:
            return f
        try:
            lam, mu = rates_from_odds(match, f.rho)
        except (ValueError, RuntimeError) as exc:
            log.warning("No market rates for %s v %s on %s: %s", f.home, f.away, f.date.date(), exc)
            return f
        return replace(f, market_lam=lam, market_mu=mu)
    return tuple(with_market(f) for f in forecasts)


def tune_blend(forecasts: Sequence[Forecast], weights: Sequence[float]) -> dict:
    """λ = w × model + (1 − w) × market for each w, on exact-score log loss."""
    losses = {w: per_match_log_loss(forecasts, w) for w in weights}
    close = within_one_se(losses)
    rows = [{"model_weight": w, **forecast_metrics(forecasts, w), "within_one_se": w in close} for w in weights]
    return {
        "rows": rows,
        "best": min(rows, key=lambda r: r["exact_log_loss"]),
        "with_odds": sum(f.market_lam is not None for f in forecasts),
    }
