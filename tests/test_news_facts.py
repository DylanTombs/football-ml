from datetime import date

import pytest

from football_ml.config import NEWS_FACT_REASONS
from football_ml.facts.extractor import InvalidFacts, NewsFactExtractor, parse_facts
from football_ml.facts.news import NewsItem, news_items
from football_ml.facts.prompt import RESPONSE_SCHEMA, build_prompt
from football_ml.llm.base import LLMConfigError, LLMError
from football_ml.llm.cache import JsonFileCache

SNAPSHOT = {
    "gameweek": 12,
    "taken_at": "2026-11-20T09:00:00+00:00",
    "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Spurs"}],
    "elements": [
        {"id": 12, "web_name": "Saka", "team": 1, "status": "i", "chance_of_playing_next_round": 0,
         "news": "Thigh injury - Expected back 29 Nov", "news_added": "2026-11-15T20:30:09Z"},
        {"id": 20, "web_name": "Dowman", "team": 1, "status": "s", "chance_of_playing_next_round": 0,
         "news": "Suspended until 29 Nov", "news_added": "2026-10-27T22:30:09Z"},
        {"id": 27, "web_name": "G.Jesus", "team": 1, "status": "u", "chance_of_playing_next_round": 0,
         "news": "Has joined FC Barcelona permanently", "news_added": "2026-09-02T16:00:28Z"},
        {"id": 31, "web_name": "Fit", "team": 2, "status": "a", "chance_of_playing_next_round": None,
         "news": "", "news_added": None},
        {"id": 33, "web_name": "Maddison", "team": 2, "status": "a", "chance_of_playing_next_round": 100,
         "news": "Returned to training", "news_added": "2026-11-18T10:00:00Z"},
    ],
}
ITEMS = news_items(SNAPSHOT)
AS_OF = date(2026, 11, 20)

GOOD = {
    "players": [
        {"player_id": 12, "reason": "injury", "expected_return": "2026-11-29"},
        {"player_id": 20, "reason": "suspension", "expected_return": "2026-11-29"},
        {"player_id": 33, "reason": "returned", "expected_return": None},
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


def _with(**changes):
    return {"players": [{**GOOD["players"][0], **changes}, *GOOD["players"][1:]]}


# --- news items ----------------------------------------------------------------

def test_news_items_keep_players_with_news_still_at_the_club():
    assert [i.player_id for i in ITEMS] == [12, 20, 33]


def test_news_item_fields_come_from_the_snapshot():
    saka = ITEMS[0]
    assert saka == NewsItem(player_id=12, name="Saka", team="Arsenal", status="i", chance_next_round=0,
                            news="Thigh injury - Expected back 29 Nov", news_added=date(2026, 11, 15))


def test_news_item_without_a_date_keeps_none():
    snapshot = {**SNAPSHOT, "elements": [{**SNAPSHOT["elements"][0], "news_added": None}]}
    assert news_items(snapshot)[0].news_added is None


def test_news_items_map_fpl_team_names_to_canonical():
    assert ITEMS[2].team == "Tottenham"


# --- prompt ----------------------------------------------------------------------

def test_prompt_lists_every_item_with_its_text_and_the_date():
    prompt = build_prompt(ITEMS, AS_OF)
    assert "2026-11-20" in prompt
    for item in ITEMS:
        assert f"id {item.player_id}" in prompt and item.news in prompt
    assert "Jesus" not in prompt


def test_schema_is_strict_and_reasons_come_from_config():
    item = RESPONSE_SCHEMA["properties"]["players"]["items"]
    assert RESPONSE_SCHEMA["additionalProperties"] is False and item["additionalProperties"] is False
    assert item["properties"]["reason"]["enum"] == list(NEWS_FACT_REASONS)
    assert set(item["required"]) == {"player_id", "reason", "expected_return"}


# --- parsing -----------------------------------------------------------------------

def test_parse_good_facts():
    facts = parse_facts(ITEMS, GOOD)
    assert [(f.player_id, f.reason, f.expected_return) for f in facts] == [
        (12, "injury", date(2026, 11, 29)),
        (20, "suspension", date(2026, 11, 29)),
        (33, "returned", None),
    ]
    assert facts[0].team == "Arsenal" and facts[2].team == "Tottenham"


def test_parse_returns_facts_in_item_order_whatever_the_response_order():
    facts = parse_facts(ITEMS, {"players": list(reversed(GOOD["players"]))})
    assert [f.player_id for f in facts] == [12, 20, 33]


@pytest.mark.parametrize("data, message", [
    ("nope", "object"),
    ({}, "players must be a list"),
    ({"players": ["x"]}, "object"),
    (_with(player_id=99), "not in the news"),
    (_with(player_id="12"), "not in the news"),
    ({"players": GOOD["players"][:2]}, "missing"),
    ({"players": GOOD["players"] + [GOOD["players"][0]]}, "more than once"),
    (_with(reason="hamstring"), "reason"),
    (_with(expected_return="29 Nov"), "date"),
    (_with(expected_return=20261129), "date"),
    (_with(expected_return="2026-11-10"), "before the news"),
    (_with(expected_return="2028-01-01"), "too far"),
])
def test_parse_rejects_invalid_facts(data, message):
    with pytest.raises(InvalidFacts, match=message):
        parse_facts(ITEMS, data)


def test_parse_allows_any_date_when_news_has_no_date():
    items = (NewsItem(1, "X", "Arsenal", "d", 50, "Knock", None),)
    data = {"players": [{"player_id": 1, "reason": "injury", "expected_return": "2026-11-22"}]}
    assert parse_facts(items, data)[0].expected_return == date(2026, 11, 22)


# --- extractor --------------------------------------------------------------------

def test_extractor_returns_validated_facts_and_the_model():
    provider = ScriptedProvider(GOOD)
    result = NewsFactExtractor(provider).extract(12, ITEMS, AS_OF)
    assert result.fallback is False and result.model == "fake-model"
    assert len(result.facts) == 3 and len(provider.prompts) == 1


def test_extractor_makes_no_call_without_news():
    provider = ScriptedProvider()
    result = NewsFactExtractor(provider).extract(12, (), AS_OF)
    assert result.facts == () and result.fallback is False and provider.prompts == []


def test_extractor_retries_once_with_the_error():
    provider = ScriptedProvider(_with(reason="hamstring"), GOOD)
    result = NewsFactExtractor(provider).extract(12, ITEMS, AS_OF)
    assert result.fallback is False
    assert "reason" in provider.prompts[1] and "rejected" in provider.prompts[1]


def test_extractor_falls_back_to_no_facts_after_two_failures():
    provider = ScriptedProvider(LLMError("timeout"), {"players": []})
    result = NewsFactExtractor(provider).extract(12, ITEMS, AS_OF)
    assert result.fallback is True and result.facts == ()


def test_extractor_does_not_swallow_config_errors():
    with pytest.raises(LLMConfigError):
        NewsFactExtractor(ScriptedProvider(LLMConfigError("bad key"))).extract(12, ITEMS, AS_OF)


def test_extractor_caches_prompt_and_response_and_reuses_them(tmp_path):
    cache = JsonFileCache(tmp_path)
    NewsFactExtractor(ScriptedProvider(GOOD), cache).extract(12, ITEMS, AS_OF)
    [entry] = list(tmp_path.rglob("*.json"))
    assert entry.parent.name == "news-facts" and entry.parent.parent.name == "gw12"

    second = ScriptedProvider()  # would raise IndexError if called
    result = NewsFactExtractor(second, cache).extract(12, ITEMS, AS_OF)
    assert len(result.facts) == 3 and second.prompts == []
    stored = cache.get(("gw12", "news-facts", entry.stem))
    assert stored["model"] == "fake-model" and "Saka" in stored["prompt"] and stored["response"] == GOOD


def test_extractor_calls_again_when_the_news_changes(tmp_path):
    cache = JsonFileCache(tmp_path)
    NewsFactExtractor(ScriptedProvider(GOOD), cache).extract(12, ITEMS, AS_OF)
    changed = (ITEMS[0], ITEMS[1])
    provider = ScriptedProvider({"players": GOOD["players"][:2]})
    result = NewsFactExtractor(provider, cache).extract(12, changed, AS_OF)
    assert len(provider.prompts) == 1 and len(result.facts) == 2


def test_extractor_ignores_an_invalid_cached_response(tmp_path):
    cache = JsonFileCache(tmp_path)
    NewsFactExtractor(ScriptedProvider(GOOD), cache).extract(12, ITEMS, AS_OF)
    [entry] = list(tmp_path.rglob("*.json"))
    cache.put(("gw12", "news-facts", entry.stem), {"model": "fake-model", "prompt": "", "response": {}})
    provider = ScriptedProvider(GOOD)
    result = NewsFactExtractor(provider, cache).extract(12, ITEMS, AS_OF)
    assert len(provider.prompts) == 1 and len(result.facts) == 3
