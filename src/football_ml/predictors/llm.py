"""An LLM that adjusts the Dixon-Coles baseline, with validation, one retry and a fallback."""

import logging
from dataclasses import replace

from football_ml.config import LLM_MAX_ATTEMPTS, MAX_GOALS, MAX_STOPPAGE_MINUTES, STOPPAGE_MINUTES
from football_ml.llm.base import LLMConfigError, LLMError, LLMProvider
from football_ml.llm.cache import JsonFileCache
from football_ml.predict.timing import GoalTime, parse_clock
from football_ml.predictors.base import SIDES, FixtureContext, MatchPrediction, PredictedGoal
from football_ml.predictors.llm_prompt import RESPONSE_SCHEMA, SYSTEM_PROMPT, build_prompt, scorer_choices

__all__ = ["RESPONSE_SCHEMA", "InvalidPrediction", "LLMPredictor", "build_prompt", "parse_prediction"]

log = logging.getLogger(__name__)


class InvalidPrediction(ValueError):
    """The LLM's JSON doesn't describe a usable prediction."""


def _goal_count(data: dict, field: str) -> int:
    value = data.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidPrediction(f"{field} must be an integer, got {value!r}")
    if not 0 <= value <= MAX_GOALS:
        raise InvalidPrediction(f"{field} must be between 0 and {MAX_GOALS}, got {value}")
    return value


def _clock(raw) -> GoalTime:
    try:
        clock = parse_clock(str(raw))
    except ValueError as exc:
        raise InvalidPrediction(f"invalid match clock {raw!r}") from exc
    base = int(clock.sort_key)
    stoppage = clock.elapsed() - base
    if not 1 <= base <= 90:
        raise InvalidPrediction(f"minute must be 1-90 plus stoppage, got {raw!r}")
    if stoppage and (base not in STOPPAGE_MINUTES or stoppage > MAX_STOPPAGE_MINUTES):
        raise InvalidPrediction(f"stoppage time only after 45' or 90' (max +{MAX_STOPPAGE_MINUTES}), got {raw!r}")
    return clock


def _goal(ctx: FixtureContext, raw) -> PredictedGoal:
    if not isinstance(raw, dict):
        raise InvalidPrediction(f"each goal must be an object, got {raw!r}")
    side = raw.get("team")
    if side not in SIDES:
        raise InvalidPrediction(f"goal team must be 'home' or 'away', got {side!r}")
    squad = ctx.home_squad if side == "home" else ctx.away_squad
    choices = {p.id: p for p in scorer_choices(squad)}
    scorer = choices.get(raw.get("scorer_id"))
    if scorer is None:
        raise InvalidPrediction(f"scorer_id {raw.get('scorer_id')!r} is not an available {side} player")
    return PredictedGoal(side, _clock(raw.get("minute")), scorer)


def parse_prediction(ctx: FixtureContext, data: dict) -> MatchPrediction:
    if not isinstance(data, dict):
        raise InvalidPrediction(f"response must be an object, got {type(data).__name__}")
    home, away = _goal_count(data, "home_goals"), _goal_count(data, "away_goals")
    raw_goals = data.get("goals")
    if not isinstance(raw_goals, list):
        raise InvalidPrediction("goals must be a list")
    goals = tuple(_goal(ctx, g) for g in raw_goals)
    for side, expected in (("home", home), ("away", away)):
        listed = sum(g.side == side for g in goals)
        if listed != expected:
            raise InvalidPrediction(f"{side} goals listed ({listed}) must equal {side}_goals ({expected})")
    reasoning = data.get("reasoning")
    return MatchPrediction(home, away, goals, reasoning=str(reasoning) if reasoning else None)


class LLMPredictor:
    def __init__(self, name: str, provider: LLMProvider, cache: JsonFileCache | None = None,
                 max_attempts: int = LLM_MAX_ATTEMPTS):
        self.name = name
        self._provider = provider
        self._cache = cache
        self._max_attempts = max_attempts

    def describe(self) -> dict:
        return {"type": "llm", "model": self._provider.model}

    def _cached(self, ctx: FixtureContext, key) -> MatchPrediction | None:
        data = self._cache.get(key) if self._cache else None
        if data is None:
            return None
        try:
            return parse_prediction(ctx, data)
        except InvalidPrediction as exc:
            log.warning("%s: ignoring invalid cached response for fixture %d: %s", self.name, ctx.fixture.id, exc)
            return None

    def predict(self, ctx: FixtureContext) -> MatchPrediction:
        key = (f"gw{ctx.gameweek:02d}", self.name, str(ctx.fixture.id))
        cached = self._cached(ctx, key)
        if cached:
            return cached

        prompt, error = build_prompt(ctx), None
        for attempt in range(1, self._max_attempts + 1):
            user = prompt if error is None else f"{prompt}\n\nYour previous answer was rejected: {error}. Fix it."
            try:
                data = self._provider.complete_json(SYSTEM_PROMPT, user, RESPONSE_SCHEMA)
                prediction = parse_prediction(ctx, data)
            except LLMConfigError:
                raise
            except (LLMError, InvalidPrediction) as exc:
                error = str(exc)
                log.warning("%s attempt %d for %s v %s failed: %s",
                            self.name, attempt, ctx.fixture.home, ctx.fixture.away, error)
                continue
            if self._cache:
                self._cache.put(key, data)
            return prediction

        log.warning("%s: falling back to Dixon-Coles for %s v %s", self.name, ctx.fixture.home, ctx.fixture.away)
        return replace(ctx.baseline, fallback=True, reasoning=f"Fell back to Dixon-Coles: {error}")
