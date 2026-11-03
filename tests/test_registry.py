import pytest

from football_ml.llm.anthropic import AnthropicProvider
from football_ml.llm.openrouter import OpenRouterProvider
from football_ml.predictors.dixon_coles import DixonColesPredictor
from football_ml.predictors.llm import LLMPredictor
from football_ml.predictors.registry import build_predictors, parse_predictor_names

KEYS = {"ANTHROPIC_API_KEY": "a", "OPENROUTER_API_KEY": "o"}


def test_parse_predictor_names():
    assert parse_predictor_names("dixon-coles, claude,gpt") == ("dixon-coles", "claude", "gpt")


@pytest.mark.parametrize("raw", ["", "dixon-coles,crystal-ball", "claude,claude"])
def test_parse_predictor_names_rejects_bad_lists(raw):
    with pytest.raises(ValueError):
        parse_predictor_names(raw)


def test_dixon_coles_needs_no_keys(tmp_path):
    (predictor,) = build_predictors(("dixon-coles",), env={}, cache_dir=tmp_path)
    assert isinstance(predictor, DixonColesPredictor)


def test_builds_llm_predictors_with_the_right_providers(tmp_path):
    predictors = build_predictors(("dixon-coles", "claude", "gpt"), env=KEYS, cache_dir=tmp_path)
    assert [p.name for p in predictors] == ["dixon-coles", "claude", "gpt"]
    assert all(isinstance(p, LLMPredictor) for p in predictors[1:])
    assert isinstance(predictors[1]._provider, AnthropicProvider)
    assert isinstance(predictors[2]._provider, OpenRouterProvider)
    assert predictors[1].describe()["model"] == "claude-opus-5-5"


def test_missing_keys_are_reported_together_before_anything_runs(tmp_path):
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY.*OPENROUTER_API_KEY"):
        build_predictors(("claude", "gpt"), env={}, cache_dir=tmp_path)


def test_only_enabled_predictors_need_keys(tmp_path):
    (predictor,) = build_predictors(("gpt",), env={"OPENROUTER_API_KEY": "o"}, cache_dir=tmp_path)
    assert predictor.name == "gpt"


def test_blank_key_counts_as_missing(tmp_path):
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build_predictors(("gemini",), env={"OPENROUTER_API_KEY": "  "}, cache_dir=tmp_path)
