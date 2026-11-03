"""Replay past gameweeks as if predicting them the week before, then score them."""

import logging
from pathlib import Path

from football_ml.config import BACKTEST_DIR
from football_ml.evaluate.season import score_gameweek, season_table, write_gameweek_results
from football_ml.pipeline import SeasonInputs, predict_gameweek, write_report
from football_ml.sources.results import build_actual_matches

log = logging.getLogger(__name__)


def run_backtest(first: int, last: int, inputs: SeasonInputs, out_dir: Path = BACKTEST_DIR) -> dict:
    """Results only use data from before each gameweek. Player stats come from
    the live FPL feed (season-to-date), so scorer points are optimistic
    until weekly snapshots exist — flagged as `scorer_leak`."""
    if first > last:
        raise ValueError(f"--from ({first}) must be <= --to ({last})")

    gameweeks = []
    for gw in range(first, last + 1):
        actual = build_actual_matches(inputs.raw_fixtures, inputs.bootstrap, inputs.goal_events, gw)
        if not actual:
            log.warning("GW%d has no finished matches; skipping", gw)
            continue
        report = predict_gameweek(gw, inputs)
        write_report(report, out_dir / "predictions")
        result = score_gameweek(report, actual)
        write_gameweek_results(result, out_dir / "results")
        gameweeks.append(result)
        log.info("GW%d: %d pts", gw, result["totals"]["points"])

    return {
        "scorer_leak": True,
        "gameweeks": gameweeks,
        "season": season_table(gameweeks),
    }
