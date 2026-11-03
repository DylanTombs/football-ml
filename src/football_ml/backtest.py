"""Replay past gameweeks as if predicting them the week before, then score them."""

import logging
from dataclasses import replace
from pathlib import Path

from football_ml.config import BACKTEST_DIR, DEFAULT_BACKTEST_PROXY, INJURY_IMPACT
from football_ml.evaluate.season import score_gameweek, season_table, write_gameweek_results
from football_ml.historical_availability import Minutes, players_as_of, team_gameweeks
from football_ml.pipeline import SeasonInputs, predict_gameweek, write_report
from football_ml.sources.results import build_actual_matches

log = logging.getLogger(__name__)


def run_backtest(
    first: int,
    last: int,
    inputs: SeasonInputs,
    out_dir: Path = BACKTEST_DIR,
    injury_impact: float = INJURY_IMPACT,
    gameweek_minutes: Minutes | None = None,
    proxy: str = DEFAULT_BACKTEST_PROXY,
) -> dict:
    """Results only use data from before each gameweek. Player stats come from
    the live FPL feed (season-to-date), so scorer points are optimistic
    until weekly snapshots exist — flagged as `scorer_leak`. Who was missing
    each week is rebuilt from FPL minutes using `proxy`."""
    if first > last:
        raise ValueError(f"--from ({first}) must be <= --to ({last})")
    if proxy != "live" and gameweek_minutes is None:
        raise ValueError(f"Availability proxy {proxy!r} needs per-gameweek minutes")

    teams = team_gameweeks(inputs.raw_fixtures, inputs.bootstrap)
    gameweeks = []
    for gw in range(first, last + 1):
        actual = build_actual_matches(inputs.raw_fixtures, inputs.bootstrap, inputs.goal_events, gw)
        if not actual:
            log.warning("GW%d has no finished matches; skipping", gw)
            continue
        players = players_as_of(inputs.players, gameweek_minutes or {}, teams, gw, proxy)
        report = predict_gameweek(gw, replace(inputs, players=players), injury_impact)
        write_report(report, out_dir / "predictions")
        result = score_gameweek(report, actual)
        write_gameweek_results(result, out_dir / "results")
        gameweeks.append(result)
        log.info("GW%d: %d pts", gw, result["totals"]["points"])

    return {
        "scorer_leak": True,
        "availability_proxy": proxy,
        "injury_impact": injury_impact,
        "gameweeks": gameweeks,
        "season": season_table(gameweeks),
    }
