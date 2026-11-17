"""Press-conference / team-news text pasted into data/previews/gwNN/<team-slug>.md.
Claude reports availability facts for players the text mentions, by FPL id from that
team's squad, so every name is checked against FPL. These facts take priority over FPL
news for the same player."""

import logging
import re
from dataclasses import dataclass, replace
from datetime import date
from functools import partial
from pathlib import Path

from football_ml.config import LLM_MAX_ATTEMPTS, NEWS_FACT_REASONS, PRESS_STATUS_AVAILABILITY, PREVIEW_SUFFIXES
from football_ml.facts.extractor import NewsFacts, PlayerFact
from football_ml.facts.llm_call import InvalidFacts, cache_key, check_return_date, repeated_ids, validated_call
from football_ml.llm.base import LLMProvider
from football_ml.llm.cache import JsonFileCache
from football_ml.predict.scorers import Player

log = logging.getLogger(__name__)

CACHE_NAMESPACE = "press-facts"
STATUSES = tuple(PRESS_STATUS_AVAILABILITY)

PRESS_SYSTEM_PROMPT = (
    "You read a Premier League manager's press conference or team news and report player availability "
    "for the next match. List only players the text says something about: 'out' if ruled out, "
    "'doubtful' if uncertain or being assessed, 'fit' if declared available or back. Give the reason "
    "and the return date (YYYY-MM-DD) only if the text states one; otherwise null. Use only ids from "
    "the squad list. Report only what the text says; don't guess and don't judge importance."
)

PRESS_SCHEMA = {
    "type": "object",
    "properties": {
        "players": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer"},
                    "status": {"type": "string", "enum": list(STATUSES)},
                    "reason": {"type": "string", "enum": list(NEWS_FACT_REASONS)},
                    "expected_return": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["player_id", "status", "reason", "expected_return"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["players"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Preview:
    team: str
    text: str


def team_slug(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", team.lower().replace("'", "")).strip("-")


def load_previews(previews_dir: Path, gameweek: int, teams: tuple[str, ...]) -> tuple[Preview, ...]:
    folder = previews_dir / f"gw{gameweek:02d}"
    if not folder.is_dir():
        return ()
    by_slug = {team_slug(t): t for t in teams}
    previews = []
    for path in sorted(p for p in folder.iterdir() if p.suffix in PREVIEW_SUFFIXES):
        team = by_slug.get(path.stem)
        if team is None:
            log.warning("Ignoring preview %s: no GW%d team called that; use one of %s",
                        path.name, gameweek, ", ".join(sorted(by_slug)))
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            previews.append(Preview(team, text))
    return tuple(previews)


def _squad(squad: tuple[Player, ...]) -> tuple[Player, ...]:
    return tuple(p for p in squad if not p.departed)


def build_press_prompt(preview: Preview, squad: tuple[Player, ...], as_of: date) -> str:
    players = "\n".join(f"- id {p.id} | {p.full_name or p.name} ({p.name})" for p in _squad(squad))
    return (f"Today is {as_of.isoformat()}.\n\nTeam: {preview.team}\n\nSquad:\n{players}\n\n"
            f"Text:\n\"\"\"\n{preview.text}\n\"\"\"")


def _press_fact(by_id: dict[int, Player], team: str, raw, as_of: date) -> PlayerFact:
    if not isinstance(raw, dict):
        raise InvalidFacts(f"each player entry must be an object, got {raw!r}")
    pid = raw.get("player_id")
    if not isinstance(pid, int) or pid not in by_id:
        raise InvalidFacts(f"player_id {pid!r} is not in the {team} squad")
    status, reason = raw.get("status"), raw.get("reason")
    if status not in STATUSES:
        raise InvalidFacts(f"status for id {pid} must be one of {STATUSES}, got {status!r}")
    if reason not in NEWS_FACT_REASONS:
        raise InvalidFacts(f"reason for id {pid} must be one of {NEWS_FACT_REASONS}, got {reason!r}")
    back = check_return_date(pid, raw.get("expected_return"), as_of)
    return PlayerFact(pid, team, reason, back, as_of, availability=PRESS_STATUS_AVAILABILITY[status])


def parse_press_facts(squad: tuple[Player, ...], data, as_of: date) -> tuple[PlayerFact, ...]:
    if not isinstance(data, dict) or not isinstance(data.get("players"), list):
        raise InvalidFacts("response must be an object with a players list")
    members = _squad(squad)
    team = members[0].team if members else "?"
    facts = tuple(_press_fact({p.id: p for p in members}, team, raw, as_of) for raw in data["players"])
    repeated_ids([f.player_id for f in facts])
    return facts


class PressFactExtractor:
    def __init__(self, provider: LLMProvider, cache: JsonFileCache | None = None,
                 max_attempts: int = LLM_MAX_ATTEMPTS):
        self._provider = provider
        self._cache = cache
        self._max_attempts = max_attempts

    def extract(self, gameweek: int, preview: Preview, squad: tuple[Player, ...], as_of: date) -> NewsFacts:
        model = self._provider.model
        prompt = build_press_prompt(preview, squad, as_of)
        key = cache_key(gameweek, CACHE_NAMESPACE, model, PRESS_SYSTEM_PROMPT, prompt)
        facts = validated_call(self._provider, self._cache, key, PRESS_SYSTEM_PROMPT, prompt, PRESS_SCHEMA,
                               partial(parse_press_facts, squad, as_of=as_of), self._max_attempts)
        return NewsFacts(facts or (), model, fallback=facts is None)


def merge_facts(fpl: tuple[PlayerFact, ...], press: tuple[PlayerFact, ...],
                players: tuple[Player, ...]) -> tuple[PlayerFact, ...]:
    """Press facts replace FPL facts for the same player. A press conference repeating an
    absence FPL already showed keeps FPL's news date, so old news stays old."""
    fpl_by_id = {f.player_id: f for f in fpl}
    availability = {p.id: p.availability for p in players}

    def dated(fact: PlayerFact) -> PlayerFact:
        earlier = fpl_by_id.get(fact.player_id)
        repeats = (earlier is not None and (fact.availability or 0.0) < 1
                   and availability.get(fact.player_id, 1.0) < 1)
        return replace(fact, news_added=earlier.news_added) if repeats else fact

    pressed = {f.player_id for f in press}
    return tuple(f for f in fpl if f.player_id not in pressed) + tuple(dated(f) for f in press)
