"""Prompt and JSON schema for turning FPL news text into facts. Claude reports facts only:
no multipliers, no scores, no opinions on how much a player matters."""

from datetime import date

from football_ml.config import NEWS_FACT_REASONS
from football_ml.facts.news import NewsItem

SYSTEM_PROMPT = (
    "You read Fantasy Premier League injury and availability news and turn each item into facts. "
    "For every player listed, give exactly one entry with the reason for the news and the date the "
    "news says they are expected back (YYYY-MM-DD). Use 'returned' when the news says the player is "
    "back in training or fit again. Use null for expected_return when the news gives no date or only "
    "says 'unknown'. Resolve dates like '29 Nov' against today's date. Report only what the text "
    "says: don't guess return dates and don't judge how important the player is."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "players": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer"},
                    "reason": {"type": "string", "enum": list(NEWS_FACT_REASONS)},
                    "expected_return": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["player_id", "reason", "expected_return"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["players"],
    "additionalProperties": False,
}


def _line(item: NewsItem) -> str:
    chance = "not given" if item.chance_next_round is None else f"{item.chance_next_round}%"
    added = item.news_added.isoformat() if item.news_added else "unknown"
    return (f"- id {item.player_id} | {item.name} ({item.team}) | FPL status {item.status}, "
            f"chance next round {chance} | news added {added} | \"{item.news}\"")


def build_prompt(items: tuple[NewsItem, ...], as_of: date) -> str:
    lines = "\n".join(_line(i) for i in items)
    return f"Today is {as_of.isoformat()}.\n\nPlayers with news ({len(items)}):\n{lines}"
