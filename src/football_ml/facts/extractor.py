"""Claude over FPL news: validated facts, one retry, then no facts at all (the shadow
prediction then equals the baseline). Prompt and response are cached together so every
run can be repeated and audited."""

import hashlib
import logging
from dataclasses import dataclass
from datetime import date, timedelta

from football_ml.config import LLM_MAX_ATTEMPTS, NEWS_FACT_REASONS, NEWS_MAX_RETURN_DAYS
from football_ml.facts.news import NewsItem
from football_ml.facts.prompt import RESPONSE_SCHEMA, SYSTEM_PROMPT, build_prompt
from football_ml.llm.base import LLMConfigError, LLMError, LLMProvider
from football_ml.llm.cache import JsonFileCache

log = logging.getLogger(__name__)

CACHE_NAMESPACE = "news-facts"


class InvalidFacts(ValueError):
    """The LLM's JSON doesn't describe one valid fact per news item."""


@dataclass(frozen=True)
class PlayerFact:
    player_id: int
    team: str
    reason: str
    expected_return: date | None
    news_added: date | None = None


@dataclass(frozen=True)
class NewsFacts:
    facts: tuple[PlayerFact, ...]
    model: str
    fallback: bool


def _return_date(item: NewsItem, raw) -> date | None:
    if raw is None:
        return None
    try:
        parsed = date.fromisoformat(raw) if isinstance(raw, str) else None
    except ValueError:
        parsed = None
    if parsed is None:
        raise InvalidFacts(f"expected_return for id {item.player_id} must be a YYYY-MM-DD date or null, got {raw!r}")
    if item.news_added is not None:
        if parsed < item.news_added:
            raise InvalidFacts(f"expected_return {raw} for id {item.player_id} is before the news ({item.news_added})")
        if parsed > item.news_added + timedelta(days=NEWS_MAX_RETURN_DAYS):
            raise InvalidFacts(f"expected_return {raw} for id {item.player_id} is too far after the news")
    return parsed


def _fact(by_id: dict[int, NewsItem], raw) -> PlayerFact:
    if not isinstance(raw, dict):
        raise InvalidFacts(f"each player entry must be an object, got {raw!r}")
    item = by_id.get(raw.get("player_id")) if isinstance(raw.get("player_id"), int) else None
    if item is None:
        raise InvalidFacts(f"player_id {raw.get('player_id')!r} is not in the news list")
    reason = raw.get("reason")
    if reason not in NEWS_FACT_REASONS:
        raise InvalidFacts(f"reason for id {item.player_id} must be one of {NEWS_FACT_REASONS}, got {reason!r}")
    return PlayerFact(item.player_id, item.team, reason, _return_date(item, raw.get("expected_return")), item.news_added)


def parse_facts(items: tuple[NewsItem, ...], data) -> tuple[PlayerFact, ...]:
    if not isinstance(data, dict):
        raise InvalidFacts(f"response must be an object, got {type(data).__name__}")
    raw_players = data.get("players")
    if not isinstance(raw_players, list):
        raise InvalidFacts("players must be a list")
    by_id = {i.player_id: i for i in items}
    facts = [_fact(by_id, raw) for raw in raw_players]
    ids = [f.player_id for f in facts]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise InvalidFacts(f"player ids listed more than once: {repeated}")
    missing = sorted(set(by_id) - set(ids))
    if missing:
        raise InvalidFacts(f"entries missing for player ids {missing}")
    by_fact = {f.player_id: f for f in facts}
    return tuple(by_fact[i.player_id] for i in items)


def _digest(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\n{SYSTEM_PROMPT}\n{prompt}".encode()).hexdigest()[:16]


class NewsFactExtractor:
    def __init__(self, provider: LLMProvider, cache: JsonFileCache | None = None,
                 max_attempts: int = LLM_MAX_ATTEMPTS):
        self._provider = provider
        self._cache = cache
        self._max_attempts = max_attempts

    def _cached(self, key, items) -> tuple[PlayerFact, ...] | None:
        entry = self._cache.get(key) if self._cache else None
        if entry is None:
            return None
        try:
            return parse_facts(items, entry.get("response"))
        except InvalidFacts as exc:
            log.warning("Ignoring invalid cached news facts %s: %s", key, exc)
            return None

    def _ask(self, prompt: str, items) -> tuple[tuple[PlayerFact, ...], dict] | None:
        error = None
        for attempt in range(1, self._max_attempts + 1):
            user = prompt if error is None else f"{prompt}\n\nYour previous answer was rejected: {error}. Fix it."
            try:
                data = self._provider.complete_json(SYSTEM_PROMPT, user, RESPONSE_SCHEMA)
                return parse_facts(items, data), data
            except LLMConfigError:
                raise
            except (LLMError, InvalidFacts) as exc:
                error = str(exc)
                log.warning("News facts attempt %d failed: %s", attempt, exc)
        return None

    def extract(self, gameweek: int, items: tuple[NewsItem, ...], as_of: date) -> NewsFacts:
        model = self._provider.model
        if not items:
            return NewsFacts((), model, fallback=False)
        prompt = build_prompt(items, as_of)
        key = (f"gw{gameweek:02d}", CACHE_NAMESPACE, _digest(model, prompt))
        cached = self._cached(key, items)
        if cached is not None:
            return NewsFacts(cached, model, fallback=False)

        answer = self._ask(prompt, items)
        if answer is None:
            log.warning("GW%d news facts: giving up after %d attempts; using no facts", gameweek, self._max_attempts)
            return NewsFacts((), model, fallback=True)
        facts, data = answer
        if self._cache:
            self._cache.put(key, {"model": model, "system": SYSTEM_PROMPT, "prompt": prompt, "response": data})
        return NewsFacts(facts, model, fallback=False)
