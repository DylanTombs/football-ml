"""The plan 08 shadow prediction: news facts (Claude) + fixture facts (ESPN, config) →
fixed-prior adjustments → the same Dixon-Coles pick on the adjusted rates. Written as
`pipeline-shadow.json` and scored alongside the main prediction; never shown as the pick."""

import logging
from collections.abc import Callable
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from football_ml.config import MANAGER_CHANGES, PREVIEWS_DIR
from football_ml.facts.adjust import FixtureAdjustment, gameweek_adjustments
from football_ml.facts.extractor import NewsFactExtractor, PlayerFact
from football_ml.facts.fixture import ManagerChange, fixture_facts, manager_changes
from football_ml.facts.news import news_items
from football_ml.facts.press import PressFactExtractor, load_previews, merge_facts
from football_ml.pipeline import GameweekContext, SeasonInputs, gameweek_context, predictor_report
from football_ml.predictors.base import Predictor
from football_ml.sources.espn_schedule import ScheduledMatch, fetch_team_schedules

log = logging.getLogger(__name__)

ScheduleFetcher = Callable[[tuple[str, ...]], tuple[ScheduledMatch, ...]]


def _positions(snapshot: dict) -> dict[int, int]:
    return {e["id"]: int(e["element_type"]) for e in snapshot["elements"] if e.get("element_type")}


def _adjustment_json(adj: FixtureAdjustment) -> dict:
    return {
        "home_multiplier": round(adj.home_multiplier, 4),
        "away_multiplier": round(adj.away_multiplier, 4),
        "items": [{**asdict(a), "effect": round(a.effect, 4)} for a in adj.adjustments],
    }


def _teams(base: GameweekContext) -> tuple[str, ...]:
    return tuple(sorted({team for c in base.fixtures for team in (c.fixture.home, c.fixture.away)}))


def _press_facts(gameweek: int, inputs: SeasonInputs, base: GameweekContext, as_of: date,
                 press_extractor: PressFactExtractor | None, previews_dir: Path) -> tuple[list, tuple[PlayerFact, ...], int]:
    """(teams with text, facts, fallbacks) from the pasted previews; nothing without an extractor."""
    if press_extractor is None:
        return [], (), 0
    previews = load_previews(previews_dir, gameweek, _teams(base))
    results = [press_extractor.extract(gameweek, p, tuple(x for x in inputs.players if x.team == p.team), as_of)
               for p in previews]
    facts = tuple(f for r in results for f in r.facts)
    return [p.team for p in previews], facts, sum(r.fallback for r in results)


def run_shadow(gameweek: int, inputs: SeasonInputs, base: GameweekContext, snapshot: dict, predictor: Predictor,
               extractor: NewsFactExtractor, fetch_schedules: ScheduleFetcher = fetch_team_schedules,
               changes: tuple[ManagerChange, ...] | None = None, press_extractor: PressFactExtractor | None = None,
               previews_dir: Path = PREVIEWS_DIR) -> dict:
    as_of: date = pd.Timestamp(snapshot["taken_at"]).date()
    items = news_items(snapshot)
    news = extractor.extract(gameweek, items, as_of)
    press_teams, press, press_fallbacks = _press_facts(gameweek, inputs, base, as_of, press_extractor, previews_dir)
    player_facts = merge_facts(news.facts, press, inputs.players)
    schedule = fetch_schedules(_teams(base))
    changes = manager_changes(MANAGER_CHANGES) if changes is None else changes
    facts = {c.fixture.id: fixture_facts(c.fixture, schedule, changes) for c in base.fixtures}
    adjustments = gameweek_adjustments(base.fixtures, player_facts, facts, _positions(snapshot), as_of)
    log.info("GW%d shadow: %d news facts (%s%s), %d schedule matches", gameweek, len(news.facts), news.model,
             ", fallback" if news.fallback else "", len(schedule))

    multipliers = {fid: (a.home_multiplier, a.away_multiplier) for fid, a in adjustments.items()}
    report = predictor_report(gameweek_context(gameweek, inputs, multipliers=multipliers), predictor)
    summary = {"news_model": news.model, "news_fallback": news.fallback, "news_items": len(items),
               "news_facts": len(news.facts), "press_teams": press_teams, "press_facts": len(press),
               "press_fallbacks": press_fallbacks, "schedule_matches": len(schedule), "as_of": as_of.isoformat()}
    return {
        **report,
        "model": {**report["model"], "shadow": summary},
        "matches": [{**m, "adjustments": _adjustment_json(adjustments[m["fixture_id"]])} for m in report["matches"]],
    }
