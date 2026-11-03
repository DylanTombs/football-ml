"""Replay past gameweeks as if predicting them the week before, then score them."""

import json
import logging
from dataclasses import replace
from pathlib import Path

from football_ml.config import (
    BACKTEST_DIR,
    DEFAULT_BACKTEST_PROXY,
    DEFAULT_PICK_STRATEGY,
    INJURY_IMPACT,
    PREDICTOR_NAME,
)
from football_ml.evaluate.compare import paired_comparison
from football_ml.evaluate.season import score_gameweek, season_table, write_gameweek_results
from football_ml.historical_availability import Minutes, players_as_of, team_gameweeks
from football_ml.pipeline import SeasonInputs, gameweek_context, predictor_report, write_report
from football_ml.predictors.base import Predictor
from football_ml.predictors.dixon_coles import DixonColesPredictor
from football_ml.predictors.naive_favourite import NaiveFavouritePredictor
from football_ml.sources.results import build_actual_matches

log = logging.getLogger(__name__)


def backtest_predictors(pick_strategy: str = DEFAULT_PICK_STRATEGY) -> tuple[Predictor, ...]:
    """Modal Dixon-Coles first (the reference), `pick_strategy` if different,
    then the naive-favourite benchmark."""
    strategies = tuple(dict.fromkeys(("modal", pick_strategy)))
    return (
        *(DixonColesPredictor(s, name=f"{PREDICTOR_NAME}-{s}") for s in strategies),
        NaiveFavouritePredictor(),
    )


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_backtest(
    first: int,
    last: int,
    inputs: SeasonInputs,
    out_dir: Path = BACKTEST_DIR,
    injury_impact: float = INJURY_IMPACT,
    gameweek_minutes: Minutes | None = None,
    proxy: str = DEFAULT_BACKTEST_PROXY,
    pick_strategy: str = DEFAULT_PICK_STRATEGY,
) -> dict:
    """Results only use data from before each gameweek. Player stats come from
    the live FPL feed (season-to-date), so scorer points are optimistic
    until weekly snapshots exist — flagged as `scorer_leak`. Who was missing
    each week is rebuilt from FPL minutes using `proxy`. Every predictor is
    compared with the first (modal Dixon-Coles) by paired bootstrap."""
    if first > last:
        raise ValueError(f"--from ({first}) must be <= --to ({last})")
    if proxy != "live" and gameweek_minutes is None:
        raise ValueError(f"Availability proxy {proxy!r} needs per-gameweek minutes")

    predictors = backtest_predictors(pick_strategy)
    teams = team_gameweeks(inputs.raw_fixtures, inputs.bootstrap)
    results: dict[str, list[dict]] = {p.name: [] for p in predictors}
    for gw in range(first, last + 1):
        actual = build_actual_matches(inputs.raw_fixtures, inputs.bootstrap, inputs.goal_events, gw)
        if not actual:
            log.warning("GW%d has no finished matches; skipping", gw)
            continue
        players = players_as_of(inputs.players, gameweek_minutes or {}, teams, gw, proxy)
        context = gameweek_context(gw, replace(inputs, players=players), injury_impact)
        for predictor in predictors:
            report = predictor_report(context, predictor)
            write_report(report, out_dir / "predictions")
            result = score_gameweek(report, actual)
            write_gameweek_results(result, out_dir / "results")
            results = {**results, predictor.name: [*results[predictor.name], result]}
            log.info("GW%d %s: %d pts", gw, predictor.name, result["totals"]["points"])

    names = [p.name for p in predictors]
    gameweeks = [r for name in names for r in results[name]]
    comparisons = [paired_comparison(results[names[0]], results[name]) for name in names[1:]] if gameweeks else []
    _write_json(out_dir / "comparisons.json", comparisons)
    return {
        "scorer_leak": True,
        "availability_proxy": proxy,
        "injury_impact": injury_impact,
        "pick_strategy": pick_strategy,
        "predictors": names,
        "gameweeks": gameweeks,
        "season": season_table(gameweeks),
        "comparisons": comparisons,
    }
