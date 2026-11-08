"""The statistical baseline, with a choice of how to pick from its probabilities."""

from football_ml.config import DEFAULT_PICK_STRATEGY, DEFAULT_SCORER_MODEL, PICK_STRATEGIES, PREDICTOR_NAME
from football_ml.predictors.base import FixtureContext, MatchPrediction
from football_ml.predictors.expected_points import check_scorer_model, expected_points_prediction


class DixonColesPredictor:
    def __init__(
        self,
        pick_strategy: str = DEFAULT_PICK_STRATEGY,
        name: str = PREDICTOR_NAME,
        scorer_model: str = DEFAULT_SCORER_MODEL,
    ):
        if pick_strategy not in PICK_STRATEGIES:
            raise ValueError(f"Unknown pick strategy {pick_strategy!r}; choose from {PICK_STRATEGIES}")
        self.pick_strategy = pick_strategy
        self.scorer_model = check_scorer_model(scorer_model)
        self.name = name

    def describe(self) -> dict:
        return {"type": "dixon-coles", "pick_strategy": self.pick_strategy, "scorer_model": self.scorer_model}

    def predict(self, ctx: FixtureContext) -> MatchPrediction:
        """The context's baseline is the modal pick (its scorers always use the weights)."""
        if self.pick_strategy == "expected-points":
            return expected_points_prediction(ctx, scorer_model=self.scorer_model)
        return ctx.baseline
