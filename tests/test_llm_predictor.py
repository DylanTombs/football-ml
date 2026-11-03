import pandas as pd
import pytest

from football_ml.llm.base import LLMConfigError, LLMError
from football_ml.llm.cache import JsonFileCache
from football_ml.model.dixon_coles import DixonColesParams
from football_ml.predict.scorers import Player
from football_ml.predict.timing import GoalTime
from football_ml.predictors.context import fixture_context
from football_ml.predictors.llm import RESPONSE_SCHEMA, InvalidPrediction, LLMPredictor, build_prompt, parse_prediction
from football_ml.sources.fpl import Fixture

PARAMS = DixonColesParams(
    teams=("Arsenal", "Coventry"), attack=(0.5, -0.5), defence=(-0.4, 0.4),
    home_advantage=0.25, rho=-0.05,
)
PLAYERS = (
    Player(1, "Saka", "Arsenal", xg=4.0, goals=5, minutes=800, available=True),
    Player(2, "Havertz", "Arsenal", xg=3.0, goals=3, minutes=700, available=True),
    Player(5, "Injured", "Arsenal", xg=2.0, goals=2, minutes=500, available=False, availability=0.0),
    Player(3, "Simms", "Coventry", xg=1.0, goals=1, minutes=600, available=True),
)
FIXTURE = Fixture(id=100, gameweek=10, kickoff=pd.Timestamp("2026-11-07T15:00Z"),
                  home="Arsenal", away="Coventry")
DISTRIBUTION = tuple(GoalTime(float(m), f"{m}'") for m in range(1, 91))
CTX = fixture_context(10, FIXTURE, PARAMS, PLAYERS, DISTRIBUTION, injury_impact=0.0)

GOOD = {
    "reasoning": "Arsenal at home, Coventry poor away.",
    "home_goals": 2, "away_goals": 1,
    "goals": [
        {"team": "home", "scorer_id": 1, "minute": "23'"},
        {"team": "away", "scorer_id": 3, "minute": "61'"},
        {"team": "home", "scorer_id": 2, "minute": "90+2'"},
    ],
}


class ScriptedProvider:
    model = "fake-model"

    def __init__(self, *responses):
        self.responses, self.prompts = list(responses), []

    def complete_json(self, system, user, schema):
        self.prompts.append(user)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_prompt_includes_baseline_squads_and_rules():
    prompt = build_prompt(CTX)
    assert "Arsenal" in prompt and "Coventry" in prompt
    assert "Dixon-Coles" in prompt
    assert "id 1" in prompt and "Saka" in prompt
    assert "Injured" in prompt and "out" in prompt.lower()
    assert "3 points" in prompt  # scoring rules


def test_prompt_lists_only_available_players_as_choices():
    available_block = build_prompt(CTX).split("Available players")[1]
    assert "id 5" not in available_block


def test_schema_is_strict():
    assert RESPONSE_SCHEMA["additionalProperties"] is False
    assert set(RESPONSE_SCHEMA["required"]) == {"reasoning", "home_goals", "away_goals", "goals"}


def test_parse_good_prediction():
    pred = parse_prediction(CTX, GOOD)
    assert (pred.home_goals, pred.away_goals) == (2, 1)
    assert [g.scorer.name for g in pred.goals] == ["Saka", "Simms", "Havertz"]
    assert pred.goals[2].minute.label == "90+2'"
    assert pred.reasoning.startswith("Arsenal")
    assert pred.fallback is False


@pytest.mark.parametrize("change, message", [
    ({"home_goals": 11}, "between 0 and"),
    ({"home_goals": -1}, "between 0 and"),
    ({"home_goals": 1.5}, "integer"),
    ({"home_goals": True}, "integer"),
    ({"home_goals": 3}, "home goals"),
    ({"goals": [{"team": "home", "scorer_id": 99, "minute": "1'"}] + GOOD["goals"][1:]}, "not an available"),
    ({"goals": [{"team": "home", "scorer_id": 5, "minute": "1'"}] + GOOD["goals"][1:]}, "not an available"),
    ({"goals": [{"team": "away", "scorer_id": 1, "minute": "1'"}] + GOOD["goals"][1:]}, "not an available"),
    ({"goals": [{"team": "home", "scorer_id": 1, "minute": "ninety"}] + GOOD["goals"][1:]}, "clock"),
    ({"goals": [{"team": "home", "scorer_id": 1, "minute": "95'"}] + GOOD["goals"][1:]}, "minute"),
    ({"goals": [{"team": "home", "scorer_id": 1, "minute": "60+3'"}] + GOOD["goals"][1:]}, "stoppage"),
    ({"goals": [{"team": "middle", "scorer_id": 1, "minute": "5'"}] + GOOD["goals"][1:]}, "team"),
    ({"goals": "lots"}, "list"),
])
def test_parse_rejects_invalid_output(change, message):
    with pytest.raises(InvalidPrediction, match=message):
        parse_prediction(CTX, {**GOOD, **change})


def test_parse_rejects_missing_fields():
    with pytest.raises(InvalidPrediction):
        parse_prediction(CTX, {"home_goals": 1})


def test_predictor_returns_valid_llm_prediction():
    provider = ScriptedProvider(GOOD)
    pred = LLMPredictor("claude", provider).predict(CTX)
    assert (pred.home_goals, pred.away_goals) == (2, 1)
    assert len(provider.prompts) == 1


def test_predictor_retries_once_with_the_error():
    provider = ScriptedProvider({**GOOD, "home_goals": 7}, GOOD)
    pred = LLMPredictor("claude", provider).predict(CTX)
    assert pred.home_goals == 2
    assert "home goals" in provider.prompts[1]


def test_predictor_falls_back_to_baseline_after_two_failures():
    provider = ScriptedProvider(LLMError("timeout"), {"nonsense": True})
    pred = LLMPredictor("claude", provider).predict(CTX)
    assert pred.fallback is True
    assert (pred.home_goals, pred.away_goals) == (CTX.baseline.home_goals, CTX.baseline.away_goals)
    assert pred.goals == CTX.baseline.goals


def test_predictor_does_not_swallow_config_errors():
    provider = ScriptedProvider(LLMConfigError("bad key"))
    with pytest.raises(LLMConfigError):
        LLMPredictor("claude", provider).predict(CTX)


def test_predictor_uses_cache_on_rerun(tmp_path):
    cache = JsonFileCache(tmp_path)
    LLMPredictor("claude", ScriptedProvider(GOOD), cache).predict(CTX)
    second = ScriptedProvider()  # would raise IndexError if called
    pred = LLMPredictor("claude", second, cache).predict(CTX)
    assert pred.home_goals == 2
    assert second.prompts == []
    assert (tmp_path / "gw10" / "claude" / "100.json").exists()


def test_fallbacks_are_not_cached(tmp_path):
    cache = JsonFileCache(tmp_path)
    LLMPredictor("claude", ScriptedProvider(LLMError("x"), LLMError("y")), cache).predict(CTX)
    assert cache.get(("gw10", "claude", "100")) is None


def test_describe_names_the_model():
    assert LLMPredictor("claude", ScriptedProvider()).describe() == {"type": "llm", "model": "fake-model"}
