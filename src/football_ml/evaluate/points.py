"""Score one prediction against one actual result."""

from collections import Counter
from dataclasses import dataclass

from football_ml.config import (
    MINUTE_TOLERANCE,
    POINTS_CORRECT_RESULT,
    POINTS_EXACT_SCORE,
    POINTS_PER_MINUTE,
    POINTS_PER_SCORER,
)
from football_ml.predict.timing import parse_clock
from football_ml.sources.results import ActualMatch


@dataclass(frozen=True)
class MatchPoints:
    fixture_id: int
    exact: bool
    correct_result: bool
    scorer_hits: int
    minute_hits: int

    @property
    def outcome_points(self) -> int:
        if self.exact:
            return POINTS_EXACT_SCORE
        return POINTS_CORRECT_RESULT if self.correct_result else 0

    @property
    def scorer_points(self) -> int:
        return self.scorer_hits * POINTS_PER_SCORER

    @property
    def minute_points(self) -> int:
        return self.minute_hits * POINTS_PER_MINUTE

    @property
    def total(self) -> int:
        return self.outcome_points + self.scorer_points + self.minute_points


def _sign(diff: int) -> int:
    return (diff > 0) - (diff < 0)


def _minute_hits(predicted_goals: list[dict], actual: ActualMatch) -> int:
    """For each predicted goal, claim the closest unclaimed actual goal by the
    same player within tolerance."""
    unclaimed = [g for g in actual.goals if not g.own_goal and g.minute is not None]
    hits = 0
    for goal in predicted_goals:
        minute = parse_clock(goal["minute"]).elapsed()
        options = [
            g for g in unclaimed
            if g.scorer_id == goal.get("scorer_id")
            and abs(g.minute.elapsed() - minute) <= MINUTE_TOLERANCE
        ]
        if options:
            best = min(options, key=lambda g: abs(g.minute.elapsed() - minute))
            unclaimed = [g for g in unclaimed if g is not best]
            hits += 1
    return hits


def score_prediction(predicted: dict, actual: ActualMatch) -> MatchPoints:
    score = predicted["prediction"]["score"]
    goals = [g for g in predicted["prediction"]["goals"] if g.get("scorer_id") is not None]

    predicted_ids = Counter(g["scorer_id"] for g in goals)
    actual_ids = Counter(g.scorer_id for g in actual.goals if not g.own_goal)

    return MatchPoints(
        fixture_id=actual.fixture_id,
        exact=(score["home"], score["away"]) == (actual.home_goals, actual.away_goals),
        correct_result=_sign(score["home"] - score["away"]) == _sign(actual.home_goals - actual.away_goals),
        scorer_hits=sum((predicted_ids & actual_ids).values()),
        minute_hits=_minute_hits(goals, actual),
    )
