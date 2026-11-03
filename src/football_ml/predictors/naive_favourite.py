"""Benchmark: the bookmaker favourite wins 1-0 (2-1 if over 2.5 goals is favoured). Never a draw."""

from dataclasses import replace

from football_ml.config import NAIVE_FAVOURITE_NAME
from football_ml.predictors.base import FixtureContext, MatchPrediction
from football_ml.predictors.expected_points import prediction_for_score


class NaiveFavouritePredictor:
    name = NAIVE_FAVOURITE_NAME

    def describe(self) -> dict:
        return {"type": "naive-favourite", "odds": "pre-closing market average"}

    def predict(self, ctx: FixtureContext) -> MatchPrediction:
        if ctx.odds is None:
            return replace(ctx.baseline, fallback=True)
        over = ctx.odds.over_probability
        winner, loser = (2, 1) if over is not None and over > 0.5 else (1, 0)
        home, away = (winner, loser) if ctx.odds.favourite == "home" else (loser, winner)
        return prediction_for_score(ctx, home, away)
