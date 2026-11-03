"""Predict a gameweek: fit → context per fixture → each predictor → report JSON."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from football_ml.config import INJURY_IMPACT, PREDICTIONS_DIR, TRAINING_SEASONS
from football_ml.model.dixon_coles import DixonColesParams, fit
from football_ml.predict.scorers import Player
from football_ml.predict.timing import GoalTime, parse_clock
from football_ml.predictors.base import FixtureContext, Predictor, match_json
from football_ml.predictors.context import fixture_context
from football_ml.predictors.dixon_coles import DixonColesPredictor
from football_ml.sources import espn, fpl
from football_ml.sources.espn import GoalEvent
from football_ml.sources.football_data import fetch_results

log = logging.getLogger(__name__)

_FALLBACK_DISTRIBUTION = tuple(GoalTime(float(m), f"{m}'") for m in range(1, 91))


@dataclass(frozen=True)
class SeasonInputs:
    """Everything fetched once per run; prediction itself does no IO."""
    bootstrap: dict
    raw_fixtures: list[dict]
    results: pd.DataFrame
    players: tuple[Player, ...]
    goal_events: tuple[GoalEvent, ...]


def load_season_inputs() -> SeasonInputs:
    bootstrap = fpl.fetch_bootstrap()
    now = pd.Timestamp.now(tz="UTC")
    return SeasonInputs(
        bootstrap=bootstrap,
        raw_fixtures=fpl.fetch_raw_fixtures(),
        results=fetch_results(TRAINING_SEASONS),
        players=fpl.parse_players(bootstrap),
        goal_events=espn.fetch_goal_events(fpl.season_start(bootstrap), now + pd.Timedelta(days=1)),
    )


def predict_fixture(
    fixture: fpl.Fixture,
    params: DixonColesParams,
    players: tuple[Player, ...],
    distribution: tuple[GoalTime, ...],
    injury_impact: float = INJURY_IMPACT,
) -> dict:
    """Dixon-Coles prediction for one fixture as report JSON."""
    ctx = fixture_context(fixture.gameweek, fixture, params, players, distribution, injury_impact)
    return match_json(ctx, ctx.baseline)


def goal_distribution(events: tuple[GoalEvent, ...], before: pd.Timestamp) -> tuple[GoalTime, ...]:
    """Goal times from matches that kicked off before `before` (no peeking)."""
    parsed = []
    for event in events:
        if event.kickoff >= before:
            continue
        try:
            parsed.append(parse_clock(event.clock))
        except ValueError:
            log.warning("Ignoring unparseable goal clock %r", event.clock)
    if not parsed:
        log.warning("No goal times before %s; using uniform 1'-90'", before.date())
        return _FALLBACK_DISTRIBUTION
    return tuple(parsed)


@dataclass(frozen=True)
class GameweekContext:
    gameweek: int
    model: dict  # Dixon-Coles fit details, shared by every predictor's report
    fixtures: tuple[FixtureContext, ...]


def gameweek_context(gameweek: int, inputs: SeasonInputs, injury_impact: float = INJURY_IMPACT) -> GameweekContext:
    fixtures = fpl.parse_fixtures(inputs.raw_fixtures, inputs.bootstrap, gameweek)
    if not fixtures:
        raise ValueError(f"No fixtures with kickoff times found for gameweek {gameweek}")

    first_kickoff = min(f.kickoff for f in fixtures)
    cutoff = first_kickoff.tz_convert(None).normalize()
    params = fit(inputs.results, reference_date=cutoff)
    distribution = goal_distribution(inputs.goal_events, before=first_kickoff)
    model = {
        "name": "dixon-coles",
        "trained_until": cutoff.date().isoformat(),
        "home_advantage": round(params.home_advantage, 3),
        "rho": round(params.rho, 3),
        "goal_time_samples": len(distribution),
        "injury_impact": injury_impact,
    }
    contexts = tuple(
        fixture_context(gameweek, f, params, inputs.players, distribution, injury_impact, inputs.results)
        for f in sorted(fixtures, key=lambda f: f.kickoff)
    )
    return GameweekContext(gameweek, model, contexts)


def predictor_report(gw: GameweekContext, predictor: Predictor) -> dict:
    return {
        "gameweek": gw.gameweek,
        "predictor": predictor.name,
        "predictor_config": predictor.describe(),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model": gw.model,
        "matches": [match_json(ctx, predictor.predict(ctx)) for ctx in gw.fixtures],
    }


def predict_gameweek(
    gameweek: int,
    inputs: SeasonInputs,
    injury_impact: float = INJURY_IMPACT,
    predictor: Predictor | None = None,
) -> dict:
    """One predictor's report (Dixon-Coles unless given)."""
    return predictor_report(gameweek_context(gameweek, inputs, injury_impact), predictor or DixonColesPredictor())


def prediction_path(gameweek: int, predictor: str, predictions_dir: Path = PREDICTIONS_DIR) -> Path:
    return predictions_dir / f"gw{gameweek:02d}" / f"{predictor}.json"


def write_report(report: dict, predictions_dir: Path = PREDICTIONS_DIR) -> Path:
    path = prediction_path(report["gameweek"], report["predictor"], predictions_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_gameweek(gameweek: int | None, predictors: tuple[Predictor, ...]) -> list[dict]:
    """Predict a gameweek with every predictor, building the shared context once."""
    inputs = load_season_inputs()
    gw = gameweek_context(gameweek or fpl.next_gameweek(inputs.bootstrap), inputs)
    reports = []
    for predictor in predictors:
        report = predictor_report(gw, predictor)
        path = write_report(report)
        log.info("Wrote %s", path)
        reports.append({**report, "path": str(path)})
    return reports
