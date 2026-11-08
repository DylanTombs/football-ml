"""What every predictor takes in and gives back, and how it's written to JSON."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from football_ml.predict.availability import TeamAvailability
from football_ml.predict.scoreline import ScorelineSummary
from football_ml.predict.scorers import Player
from football_ml.predict.timing import GoalTime
from football_ml.sources.fpl import Fixture

SIDES = ("home", "away")


@dataclass(frozen=True)
class PredictedGoal:
    side: str  # "home" | "away"
    minute: GoalTime
    scorer: Player | None

    def __post_init__(self):
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {self.side!r}")


@dataclass(frozen=True)
class MatchPrediction:
    home_goals: int
    away_goals: int
    goals: tuple[PredictedGoal, ...]
    reasoning: str | None = None
    fallback: bool = False  # True when an LLM failed and the baseline was used


@dataclass(frozen=True)
class FormResult:
    date: str
    opponent: str
    venue: str  # "H" | "A"
    goals_for: int
    goals_against: int

    @property
    def result(self) -> str:
        if self.goals_for == self.goals_against:
            return "D"
        return "W" if self.goals_for > self.goals_against else "L"


@dataclass(frozen=True)
class FixtureContext:
    """Everything a predictor may use for one fixture."""
    gameweek: int
    fixture: Fixture
    summary: ScorelineSummary  # Dixon-Coles probabilities, top scorelines, xG
    baseline: MatchPrediction  # the Dixon-Coles prediction
    home_squad: tuple[Player, ...]
    away_squad: tuple[Player, ...]
    home_availability: TeamAvailability
    away_availability: TeamAvailability
    home_form: tuple[FormResult, ...] = ()
    away_form: tuple[FormResult, ...] = ()
    # P(home i, away j) behind `summary`; None only in hand-built contexts
    matrix: np.ndarray | None = field(default=None, compare=False, repr=False)
    # plan 05 expected goals per player id; None when no scorer model was built
    home_scoring_rates: Mapping[int, float] | None = field(default=None, compare=False)
    away_scoring_rates: Mapping[int, float] | None = field(default=None, compare=False)


class Predictor(Protocol):
    name: str

    def describe(self) -> dict: ...

    def predict(self, ctx: FixtureContext) -> MatchPrediction: ...


def _goal_json(ctx: FixtureContext, goal: PredictedGoal) -> dict:
    return {
        "minute": goal.minute.label,
        "sort_key": goal.minute.sort_key,
        "team": ctx.fixture.home if goal.side == "home" else ctx.fixture.away,
        "scorer": goal.scorer.name if goal.scorer else None,
        "scorer_id": goal.scorer.id if goal.scorer else None,
    }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _missing_json(availability: TeamAvailability) -> list[dict]:
    return [{"name": m.name, "xg_share": round(m.xg_share, 3)} for m in availability.missing]


def prediction_json(ctx: FixtureContext, prediction: MatchPrediction) -> dict:
    goals = sorted(prediction.goals, key=lambda g: g.minute.sort_key)
    data = {
        "score": {"home": prediction.home_goals, "away": prediction.away_goals},
        "goals": [_goal_json(ctx, g) for g in goals],
        "fallback": prediction.fallback,
    }
    return {**data, "reasoning": prediction.reasoning} if prediction.reasoning else data


def scoreline_probability(ctx: FixtureContext, home_goals: int, away_goals: int) -> float | None:
    """Model probability of a scoreline (0 beyond the matrix, None without one)."""
    if ctx.matrix is None:
        return None
    rows, cols = ctx.matrix.shape
    if not (0 <= home_goals < rows and 0 <= away_goals < cols):
        return 0.0
    return float(ctx.matrix[home_goals, away_goals])


def match_json(ctx: FixtureContext, prediction: MatchPrediction) -> dict:
    """One match of a report. probabilities / expected_goals / top_scorelines are
    always the Dixon-Coles model's, whichever predictor made the prediction."""
    s = ctx.summary
    return {
        "fixture_id": ctx.fixture.id,
        "kickoff": ctx.fixture.kickoff.isoformat(),
        "home": ctx.fixture.home,
        "away": ctx.fixture.away,
        "prediction": prediction_json(ctx, prediction),
        "prediction_probability": _rounded(scoreline_probability(ctx, prediction.home_goals, prediction.away_goals)),
        "baseline_score": {"home": s.home_goals, "away": s.away_goals},
        "probabilities": {
            "home_win": round(s.home_win, 4), "draw": round(s.draw, 4), "away_win": round(s.away_win, 4),
        },
        "expected_goals": {
            "home": round(s.expected_home_goals, 2), "away": round(s.expected_away_goals, 2),
        },
        "top_scorelines": [
            {"score": f"{t.home_goals}-{t.away_goals}", "probability": round(t.probability, 4)}
            for t in s.top_scorelines
        ],
        "missing": {"home": _missing_json(ctx.home_availability), "away": _missing_json(ctx.away_availability)},
    }
