"""The statistical baseline: the context already holds its prediction."""

from football_ml.config import PREDICTOR_NAME
from football_ml.predictors.base import FixtureContext, MatchPrediction


class DixonColesPredictor:
    name = PREDICTOR_NAME

    def describe(self) -> dict:
        return {"type": "dixon-coles"}

    def predict(self, ctx: FixtureContext) -> MatchPrediction:
        return ctx.baseline
