"""Claude over FPL news: validated facts, one retry, then no facts at all (the shadow
prediction then equals the baseline). Prompt and response are cached together so every
run can be repeated and audited."""

from dataclasses import dataclass
from datetime import date
from functools import partial

from football_ml.config import LLM_MAX_ATTEMPTS, NEWS_FACT_REASONS
from football_ml.facts.llm_call import InvalidFacts, cache_key, check_return_date, repeated_ids, validated_call
from football_ml.facts.news import NewsItem
from football_ml.facts.prompt import RESPONSE_SCHEMA, SYSTEM_PROMPT, build_prompt
from football_ml.llm.base import LLMProvider
from football_ml.llm.cache import JsonFileCache

__all__ = ["InvalidFacts", "NewsFactExtractor", "NewsFacts", "PlayerFact", "parse_facts"]

CACHE_NAMESPACE = "news-facts"


@dataclass(frozen=True)
class PlayerFact:
    player_id: int
    team: str
    reason: str
    expected_return: date | None
    news_added: date | None = None
    # 0..1 chance of playing from a press conference; None = use FPL's chance
    availability: float | None = None


@dataclass(frozen=True)
class NewsFacts:
    facts: tuple[PlayerFact, ...]
    model: str
    fallback: bool


def _fact(by_id: dict[int, NewsItem], raw) -> PlayerFact:
    if not isinstance(raw, dict):
        raise InvalidFacts(f"each player entry must be an object, got {raw!r}")
    item = by_id.get(raw.get("player_id")) if isinstance(raw.get("player_id"), int) else None
    if item is None:
        raise InvalidFacts(f"player_id {raw.get('player_id')!r} is not in the news list")
    reason = raw.get("reason")
    if reason not in NEWS_FACT_REASONS:
        raise InvalidFacts(f"reason for id {item.player_id} must be one of {NEWS_FACT_REASONS}, got {reason!r}")
    back = check_return_date(item.player_id, raw.get("expected_return"), item.news_added)
    return PlayerFact(item.player_id, item.team, reason, back, item.news_added)


def parse_facts(items: tuple[NewsItem, ...], data) -> tuple[PlayerFact, ...]:
    if not isinstance(data, dict):
        raise InvalidFacts(f"response must be an object, got {type(data).__name__}")
    raw_players = data.get("players")
    if not isinstance(raw_players, list):
        raise InvalidFacts("players must be a list")
    by_id = {i.player_id: i for i in items}
    facts = [_fact(by_id, raw) for raw in raw_players]
    repeated_ids([f.player_id for f in facts])
    missing = sorted(set(by_id) - {f.player_id for f in facts})
    if missing:
        raise InvalidFacts(f"entries missing for player ids {missing}")
    by_fact = {f.player_id: f for f in facts}
    return tuple(by_fact[i.player_id] for i in items)


class NewsFactExtractor:
    def __init__(self, provider: LLMProvider, cache: JsonFileCache | None = None,
                 max_attempts: int = LLM_MAX_ATTEMPTS):
        self._provider = provider
        self._cache = cache
        self._max_attempts = max_attempts

    def extract(self, gameweek: int, items: tuple[NewsItem, ...], as_of: date) -> NewsFacts:
        model = self._provider.model
        if not items:
            return NewsFacts((), model, fallback=False)
        prompt = build_prompt(items, as_of)
        key = cache_key(gameweek, CACHE_NAMESPACE, model, SYSTEM_PROMPT, prompt)
        facts = validated_call(self._provider, self._cache, key, SYSTEM_PROMPT, prompt, RESPONSE_SCHEMA,
                               partial(parse_facts, items), self._max_attempts)
        return NewsFacts(facts or (), model, fallback=facts is None)
