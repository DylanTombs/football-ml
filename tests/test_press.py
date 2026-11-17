from datetime import date

import pytest

from football_ml.facts.extractor import InvalidFacts, PlayerFact
from football_ml.facts.press import (
    PRESS_SCHEMA,
    PressFactExtractor,
    Preview,
    build_press_prompt,
    load_previews,
    merge_facts,
    parse_press_facts,
    team_slug,
)
from football_ml.llm.base import LLMConfigError, LLMError
from football_ml.llm.cache import JsonFileCache
from football_ml.predict.scorers import Player

AS_OF = date(2026, 11, 20)
SQUAD = (
    Player(1, "Saka", "Arsenal", 4.0, 5, 800, True, full_name="Bukayo Saka"),
    Player(2, "Havertz", "Arsenal", 3.0, 3, 700, True, full_name="Kai Havertz"),
    Player(8, "Gone", "Arsenal", 0.0, 0, 0, False, departed=True),
)
PREVIEW = Preview("Arsenal", "Arteta: Saka is out for two weeks with a hamstring problem. Havertz is a doubt.")
GOOD = {"players": [
    {"player_id": 1, "status": "out", "reason": "injury", "expected_return": "2026-12-04"},
    {"player_id": 2, "status": "doubtful", "reason": "fitness", "expected_return": None},
]}


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


# --- files ---------------------------------------------------------------------------

@pytest.mark.parametrize("team, slug", [
    ("Arsenal", "arsenal"), ("Man United", "man-united"), ("Nott'm Forest", "nottm-forest"),
    ("Sheffield Weds", "sheffield-weds"),
])
def test_team_slug(team, slug):
    assert team_slug(team) == slug


def test_load_previews_reads_one_file_per_team(tmp_path, caplog):
    folder = tmp_path / "gw11"
    folder.mkdir()
    (folder / "arsenal.md").write_text("  Saka out.  \n", encoding="utf-8")
    (folder / "nottm-forest.txt").write_text("Full squad available.", encoding="utf-8")
    (folder / "empty.md").write_text("   ", encoding="utf-8")
    (folder / "wrexham.md").write_text("Not playing this week.", encoding="utf-8")
    (folder / "README").write_text("ignored: no .md/.txt", encoding="utf-8")
    previews = load_previews(tmp_path, 11, ("Arsenal", "Nott'm Forest", "Hull"))
    assert previews == (Preview("Arsenal", "Saka out."), Preview("Nott'm Forest", "Full squad available."))
    assert "wrexham.md" in caplog.text and "arsenal" in caplog.text  # lists the valid names


def test_load_previews_without_a_folder_is_empty(tmp_path):
    assert load_previews(tmp_path, 11, ("Arsenal",)) == ()


# --- prompt and parsing ------------------------------------------------------------

def test_press_prompt_lists_the_squad_ids_and_the_text():
    prompt = build_press_prompt(PREVIEW, SQUAD, AS_OF)
    assert "2026-11-20" in prompt and "Arsenal" in prompt and PREVIEW.text in prompt
    assert "id 1" in prompt and "Bukayo Saka" in prompt
    assert "id 8" not in prompt  # departed


def test_press_schema_is_strict():
    item = PRESS_SCHEMA["properties"]["players"]["items"]
    assert PRESS_SCHEMA["additionalProperties"] is False and item["additionalProperties"] is False
    assert item["properties"]["status"]["enum"] == ["out", "doubtful", "fit"]


def test_parse_press_facts_sets_availability_from_status():
    facts = parse_press_facts(SQUAD, GOOD, AS_OF)
    assert [(f.player_id, f.availability, f.expected_return, f.news_added) for f in facts] == [
        (1, 0.0, date(2026, 12, 4), AS_OF), (2, 0.5, None, AS_OF),
    ]
    assert all(f.team == "Arsenal" for f in facts)


def test_parse_press_facts_allows_nobody_mentioned():
    assert parse_press_facts(SQUAD, {"players": []}, AS_OF) == ()


@pytest.mark.parametrize("change, message", [
    ({"player_id": 99}, "not in the Arsenal squad"),
    ({"player_id": 8}, "not in the Arsenal squad"),
    ({"status": "injured"}, "status"),
    ({"reason": "hamstring"}, "reason"),
    ({"expected_return": "next week"}, "date"),
    ({"expected_return": "2026-11-01"}, "before the news"),
])
def test_parse_press_facts_rejects_invalid(change, message):
    data = {"players": [{**GOOD["players"][0], **change}]}
    with pytest.raises(InvalidFacts, match=message):
        parse_press_facts(SQUAD, data, AS_OF)


def test_parse_press_facts_rejects_duplicates_and_non_lists():
    with pytest.raises(InvalidFacts, match="more than once"):
        parse_press_facts(SQUAD, {"players": [GOOD["players"][0]] * 2}, AS_OF)
    with pytest.raises(InvalidFacts, match="list"):
        parse_press_facts(SQUAD, {"players": "Saka"}, AS_OF)


# --- extractor ---------------------------------------------------------------------

def test_press_extractor_returns_facts_and_caches_them(tmp_path):
    cache = JsonFileCache(tmp_path)
    result = PressFactExtractor(ScriptedProvider(GOOD), cache).extract(11, PREVIEW, SQUAD, AS_OF)
    assert len(result.facts) == 2 and result.fallback is False
    [entry] = list(tmp_path.rglob("*.json"))
    assert entry.parent.name == "press-facts" and "Arteta" in cache.get(("gw11", "press-facts", entry.stem))["prompt"]
    again = PressFactExtractor(ScriptedProvider(), cache).extract(11, PREVIEW, SQUAD, AS_OF)
    assert len(again.facts) == 2


def test_press_extractor_retries_then_falls_back():
    provider = ScriptedProvider({"players": [{**GOOD["players"][0], "player_id": 99}]}, LLMError("timeout"))
    result = PressFactExtractor(provider).extract(11, PREVIEW, SQUAD, AS_OF)
    assert result.fallback is True and result.facts == ()
    assert "not in the Arsenal squad" in provider.prompts[1]


def test_press_extractor_does_not_swallow_config_errors():
    with pytest.raises(LLMConfigError):
        PressFactExtractor(ScriptedProvider(LLMConfigError("bad key"))).extract(11, PREVIEW, SQUAD, AS_OF)


# --- merging with FPL news ------------------------------------------------------------

OLD = date(2026, 9, 1)


def test_press_fact_replaces_the_fpl_fact_for_the_same_player():
    fpl = (PlayerFact(1, "Arsenal", "injury", None, date(2026, 11, 18)),
           PlayerFact(5, "Arsenal", "suspension", None, date(2026, 11, 18)))
    press = (PlayerFact(1, "Arsenal", "injury", None, AS_OF, availability=1.0),)
    merged = merge_facts(fpl, press, SQUAD)
    assert [(f.player_id, f.availability) for f in merged] == [(5, None), (1, 1.0)]


def test_press_repeating_a_long_absence_keeps_the_old_news_date():
    hurt = (Player(1, "Saka", "Arsenal", 4.0, 5, 800, False, availability=0.0),)
    fpl = (PlayerFact(1, "Arsenal", "injury", None, OLD),)
    press = (PlayerFact(1, "Arsenal", "injury", None, AS_OF, availability=0.0),)
    [merged] = merge_facts(fpl, press, hurt)
    assert merged.news_added == OLD and merged.availability == 0.0


def test_press_news_about_a_player_fpl_has_as_fit_is_new():
    fpl = ()
    press = (PlayerFact(1, "Arsenal", "injury", None, AS_OF, availability=0.0),)
    [merged] = merge_facts(fpl, press, SQUAD)
    assert merged.news_added == AS_OF


def test_parse_press_facts_rejects_non_object_entries():
    with pytest.raises(InvalidFacts, match="object"):
        parse_press_facts(SQUAD, {"players": ["Saka"]}, AS_OF)
