"""Predict a gameweek: fit → context per fixture → each predictor → report JSON."""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from football_ml.config import CURRENT_SEASON, INJURY_IMPACT, PREDICTIONS_DIR, SNAPSHOTS_DIR, TRAINING_SEASONS
from football_ml.model.dixon_coles import DixonColesParams, fit
from football_ml.predict.scorers import Player
from football_ml.predict.scoring_chance import ScorerModel, build_scorer_model, penalty_goals
from football_ml.predict.timing import GoalTime, parse_clock
from football_ml.predictors.base import FixtureContext, Predictor, match_json
from football_ml.predictors.context import fixture_context
from football_ml.predictors.dixon_coles import DixonColesPredictor
from football_ml.sources import espn, fpl
from football_ml.sources.espn import GoalEvent
from football_ml.snapshots import write_snapshot
from football_ml.sources.football_data import fetch_odds, fetch_results, odds_for
from football_ml.sources.player_history import Appearance, fetch_player_histories
from football_ml.sources.results import build_actual_matches

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
    odds: pd.DataFrame | None = None  # pre-closing odds of played matches (none for upcoming yet)
    # plan 05: per-player match history (None = no scorer model) and FPL's
    # current penalty order (None = infer takers from past penalties)
    histories: Mapping[int, tuple[Appearance, ...]] | None = None
    penalty_order: Mapping[int, int] | None = None


def penalty_order(bootstrap: dict) -> dict[int, int]:
    return {e["id"]: int(e["penalties_order"]) for e in bootstrap["elements"] if e.get("penalties_order")}


def load_season_inputs(with_histories: bool = False) -> SeasonInputs:
    """`with_histories` adds one FPL request per player (cached for a few hours)."""
    bootstrap = fpl.fetch_bootstrap()
    raw_fixtures = fpl.fetch_raw_fixtures()
    now = pd.Timestamp.now(tz="UTC")
    return SeasonInputs(
        bootstrap=bootstrap,
        raw_fixtures=raw_fixtures,
        results=fetch_results(TRAINING_SEASONS),
        players=fpl.parse_players(bootstrap),
        goal_events=espn.fetch_goal_events(fpl.season_start(bootstrap), now + pd.Timedelta(days=1)),
        odds=fetch_odds((CURRENT_SEASON,)),
        histories=fetch_player_histories(bootstrap, raw_fixtures) if with_histories else None,
        penalty_order=penalty_order(bootstrap),
    )


def scorer_model(gameweek: int, inputs: SeasonInputs, before: pd.Timestamp) -> ScorerModel | None:
    """Plan 05 scorer model from matches before `before`; None without histories."""
    if inputs.histories is None:
        return None
    past = tuple(m for gw in range(1, gameweek)
                 for m in build_actual_matches(inputs.raw_fixtures, inputs.bootstrap, inputs.goal_events, gw))
    positions = {e["id"]: int(e["element_type"]) for e in inputs.bootstrap["elements"] if e.get("element_type")}
    return build_scorer_model(inputs.histories, positions, penalty_goals(past), before, inputs.penalty_order)


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
    scorers = scorer_model(gameweek, inputs, before=first_kickoff)
    if scorers is not None:
        model = {**model, "scorer_profiles": len(scorers.profiles)}
    contexts = tuple(
        fixture_context(
            gameweek, f, params, inputs.players, distribution, injury_impact, inputs.results,
            odds=odds_for(inputs.odds, f.home, f.away, f.kickoff), scorer_model=scorers,
        )
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


def run_gameweek(
    gameweek: int | None,
    predictors: tuple[Predictor, ...],
    predictions_dir: Path = PREDICTIONS_DIR,
    snapshots_dir: Path = SNAPSHOTS_DIR,
    now: pd.Timestamp | None = None,
) -> list[dict]:
    """Snapshot FPL player state (before the deadline only), then predict the
    gameweek with every predictor, building the shared context once."""
    needs_histories = any(getattr(p, "scorer_model", None) == "chance" for p in predictors)
    inputs = load_season_inputs(with_histories=needs_histories)
    number = gameweek or fpl.next_gameweek(inputs.bootstrap)
    snapshot = write_snapshot(inputs.bootstrap, number, now or pd.Timestamp.now(tz="UTC"), snapshots_dir)
    if snapshot:
        log.info("Wrote %s", snapshot)
    gw = gameweek_context(number, inputs)
    reports = []
    for predictor in predictors:
        report = predictor_report(gw, predictor)
        path = write_report(report, predictions_dir)
        log.info("Wrote %s", path)
        reports.append({**report, "path": str(path), "snapshot": str(snapshot) if snapshot else None})
    return reports
