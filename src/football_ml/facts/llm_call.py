"""One validated, cached LLM call shared by the fact extractors: ask, parse, retry once with
the error, and give up (None) rather than guess. The cache keeps the prompt next to the
response so every shadow run can be repeated and audited."""

import hashlib
import logging
from collections.abc import Callable
from datetime import date, timedelta
from typing import TypeVar

from football_ml.config import NEWS_MAX_RETURN_DAYS
from football_ml.llm.base import LLMConfigError, LLMError, LLMProvider
from football_ml.llm.cache import JsonFileCache

log = logging.getLogger(__name__)

T = TypeVar("T")


class InvalidFacts(ValueError):
    """The LLM's JSON doesn't describe valid facts for what it was given."""


def check_return_date(player_id: int, raw, since: date | None) -> date | None:
    """A YYYY-MM-DD date not before `since` and not absurdly far after it, or None."""
    if raw is None:
        return None
    try:
        parsed = date.fromisoformat(raw) if isinstance(raw, str) else None
    except ValueError:
        parsed = None
    if parsed is None:
        raise InvalidFacts(f"expected_return for id {player_id} must be a YYYY-MM-DD date or null, got {raw!r}")
    if since is not None:
        if parsed < since:
            raise InvalidFacts(f"expected_return {raw} for id {player_id} is before the news ({since})")
        if parsed > since + timedelta(days=NEWS_MAX_RETURN_DAYS):
            raise InvalidFacts(f"expected_return {raw} for id {player_id} is too far after the news")
    return parsed


def repeated_ids(ids: list[int]) -> None:
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise InvalidFacts(f"player ids listed more than once: {repeated}")


def cache_key(gameweek: int, namespace: str, model: str, system: str, prompt: str) -> tuple[str, ...]:
    digest = hashlib.sha256(f"{model}\n{system}\n{prompt}".encode()).hexdigest()[:16]
    return (f"gw{gameweek:02d}", namespace, digest)


def validated_call(provider: LLMProvider, cache: JsonFileCache | None, key: tuple[str, ...], system: str,
                   prompt: str, schema: dict, parse: Callable[[dict], T], max_attempts: int) -> T | None:
    entry = cache.get(key) if cache else None
    if entry is not None:
        try:
            return parse(entry.get("response"))
        except InvalidFacts as exc:
            log.warning("Ignoring invalid cached facts %s: %s", key, exc)

    error = None
    for attempt in range(1, max_attempts + 1):
        user = prompt if error is None else f"{prompt}\n\nYour previous answer was rejected: {error}. Fix it."
        try:
            data = provider.complete_json(system, user, schema)
            parsed = parse(data)
        except LLMConfigError:
            raise
        except (LLMError, InvalidFacts) as exc:
            error = str(exc)
            log.warning("%s attempt %d failed: %s", key[1], attempt, exc)
            continue
        if cache:
            cache.put(key, {"model": provider.model, "system": system, "prompt": prompt, "response": data})
        return parsed
    log.warning("%s: giving up after %d attempts; using no facts", "/".join(key), max_attempts)
    return None
