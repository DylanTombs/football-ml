"""Turn a FixtureContext into an expected-points prediction (plan 04)."""

from football_ml.config import (
    MAX_GOALS,
    MIN_SCORELINE_PROBABILITY,
    REPEAT_PICK_MINUTES,
    SCORER_MODELS,
)
from football_ml.predict.picks import best_scoreline, cumulative_scorer_points, pick_scorers
from football_ml.predict.timing import elapsed_goal_time, pick_minutes
from football_ml.predictors.base import FixtureContext, MatchPrediction, PredictedGoal


def check_scorer_model(scorer_model: str) -> str:
    if scorer_model not in SCORER_MODELS:
        raise ValueError(f"Unknown scorer model {scorer_model!r}; choose from {SCORER_MODELS}")
    return scorer_model


def _scorers(ctx: FixtureContext, side: str, n: int, scorer_model: str):
    home = side == "home"
    squad = ctx.home_squad if home else ctx.away_squad
    rate = ctx.summary.expected_home_goals if home else ctx.summary.expected_away_goals
    if check_scorer_model(scorer_model) == "weights":
        return pick_scorers(squad, rate, n)
    rates = ctx.home_scoring_rates if home else ctx.away_scoring_rates
    if rates is None:
        raise ValueError("The 'chance' scorer model needs player histories (no scoring rates in the context)")
    return pick_scorers(squad, rate, n, rates=rates)


def prediction_for_score(ctx: FixtureContext, home_goals: int, away_goals: int, *, scorer_model: str) -> MatchPrediction:
    """Scorers by P(X ≥ k) and fixed minutes for a given scoreline."""
    slots = [("home", k) for k in range(home_goals)] + [("away", k) for k in range(away_goals)]
    picks = {side: _scorers(ctx, side, n, scorer_model) for side, n in (("home", home_goals), ("away", away_goals))}
    players = [picks[side][k].player if k < len(picks[side]) else None for side, k in slots]
    minutes = pick_minutes([p.id if p else None for p in players], REPEAT_PICK_MINUTES)
    goals = tuple(
        PredictedGoal(side, elapsed_goal_time(minute), player)
        for (side, _), player, minute in zip(slots, players, minutes)
    )
    return MatchPrediction(home_goals, away_goals, tuple(sorted(goals, key=lambda g: g.minute.sort_key)))


def expected_points_prediction(
    ctx: FixtureContext, *, scorer_model: str, floor: float = MIN_SCORELINE_PROBABILITY,
) -> MatchPrediction:
    """The plausible scoreline (probability ≥ floor) worth the most expected
    points, keeping the modal pick unless something beats it."""
    if ctx.matrix is None:
        raise ValueError("Expected-points picks need the score matrix")
    max_goals = min(MAX_GOALS, ctx.matrix.shape[0] - 1)
    pick = best_scoreline(
        ctx.matrix,
        cumulative_scorer_points(_scorers(ctx, "home", max_goals, scorer_model), max_goals),
        cumulative_scorer_points(_scorers(ctx, "away", max_goals, scorer_model), max_goals),
        fallback=(ctx.summary.home_goals, ctx.summary.away_goals),
        floor=floor,
    )
    return prediction_for_score(ctx, pick.home_goals, pick.away_goals, scorer_model=scorer_model)
