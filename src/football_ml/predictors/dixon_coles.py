"""The statistical baseline, with a choice of how to pick from its probabilities."""

from football_ml.config import DEFAULT_PICK_STRATEGY, PICK_STRATEGIES, PREDICTOR_NAME
from football_ml.predictors.base import FixtureContext, MatchPrediction
from football_ml.predictors.expected_points import expected_points_prediction


class DixonColesPredictor:
    def __init__(self, pick_strategy: str = DEFAULT_PICK_STRATEGY, name: str = PREDICTOR_NAME):
        if pick_strategy not in PICK_STRATEGIES:
            raise ValueError(f"Unknown pick strategy {pick_strategy!r}; choose from {PICK_STRATEGIES}")
        self.pick_strategy = pick_strategy
        self.name = name

    def describe(self) -> dict:
        return {"type": "dixon-coles", "pick_strategy": self.pick_strategy}

    def predict(self, ctx: FixtureContext) -> MatchPrediction:
        """The context's baseline is the modal pick."""
        if self.pick_strategy == "expected-points":
            return expected_points_prediction(ctx)
        return ctx.baseline
