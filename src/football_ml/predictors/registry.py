"""Turn predictor names from the CLI into ready-to-run predictors."""

from collections.abc import Mapping
from pathlib import Path

from anthropic import Anthropic

from football_ml.config import LLM_CACHE_DIR, LLM_TIMEOUT_SECONDS, PREDICTOR_SPECS, PROVIDER_API_KEY_ENV
from football_ml.llm.anthropic import AnthropicProvider
from football_ml.llm.cache import JsonFileCache
from football_ml.llm.openrouter import OpenRouterProvider
from football_ml.predictors.base import Predictor
from football_ml.predictors.dixon_coles import DixonColesPredictor
from football_ml.predictors.llm import LLMPredictor


def parse_predictor_names(raw: str) -> tuple[str, ...]:
    names = tuple(n.strip() for n in raw.split(",") if n.strip())
    if not names:
        raise ValueError("No predictors given")
    unknown = [n for n in names if n not in PREDICTOR_SPECS]
    if unknown:
        raise ValueError(f"Unknown predictor(s) {unknown}; choose from {sorted(PREDICTOR_SPECS)}")
    if len(set(names)) != len(names):
        raise ValueError(f"Predictor listed twice: {raw}")
    return names


def _api_key(env: Mapping[str, str], provider: str) -> str | None:
    value = env.get(PROVIDER_API_KEY_ENV[provider], "").strip()
    return value or None


def _provider(provider: str, model: str, key: str):
    if provider == "anthropic":
        return AnthropicProvider(model, client=Anthropic(api_key=key, timeout=LLM_TIMEOUT_SECONDS))
    if provider == "openrouter":
        return OpenRouterProvider(model, api_key=key)
    raise ValueError(f"Unknown provider {provider!r}")


def build_predictors(
    names: tuple[str, ...], env: Mapping[str, str], cache_dir: Path = LLM_CACHE_DIR
) -> tuple[Predictor, ...]:
    """Check every needed API key first, so a run never half-finishes for a missing key."""
    specs = {name: PREDICTOR_SPECS[name] for name in parse_predictor_names(",".join(names))}
    missing = sorted({
        PROVIDER_API_KEY_ENV[spec[0]] for spec in specs.values() if spec and not _api_key(env, spec[0])
    })
    if missing:
        raise ValueError(f"Set {', '.join(missing)} to run {', '.join(n for n, s in specs.items() if s)}")

    cache = JsonFileCache(cache_dir)
    return tuple(
        DixonColesPredictor() if spec is None
        else LLMPredictor(name, _provider(spec[0], spec[1], _api_key(env, spec[0])), cache)
        for name, spec in specs.items()
    )
