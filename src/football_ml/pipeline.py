"""Predict a gameweek: fit → predict every fixture → report dict / JSON."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from football_ml.config import (
    INJURY_IMPACT,
    MAX_GOALS,
    PREDICTIONS_DIR,
    PREDICTOR_NAME,
    TOP_SCORELINES,
    TRAINING_SEASONS,
)
from football_ml.model.dixon_coles import DixonColesParams, fit, matrix_from_rates
from football_ml.predict.availability import TeamAvailability, adjust_rate, team_availability
from football_ml.predict.scoreline import summarise
from football_ml.predict.scorers import Player, assign_scorers
from football_ml.predict.timing import GoalTime, goal_times, order_goals, parse_clock
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


def _goal_timeline(fixture, summary, home_scorers, away_scorers, distribution) -> list[dict]:
    sides = order_goals(summary.home_goals, summary.away_goals)
    times = goal_times(len(sides), distribution, seed=fixture.id)
    scorers = {"home": home_scorers, "away": away_scorers}
    team = {"home": fixture.home, "away": fixture.away}

    def goal(k: int, side: str) -> dict:
        nth = sides[:k].count(side)
        scorer = scorers[side][nth] if nth < len(scorers[side]) else None
        return {
            "minute": times[k].label,
            "sort_key": times[k].sort_key,
            "team": team[side],
            "scorer": scorer.name if scorer else None,
            "scorer_id": scorer.id if scorer else None,
        }

    return [goal(k, side) for k, side in enumerate(sides)]


def _missing_json(availability: TeamAvailability) -> list[dict]:
    return [{"name": m.name, "xg_share": round(m.xg_share, 3)} for m in availability.missing]


def predict_fixture(
    fixture: fpl.Fixture,
    params: DixonColesParams,
    players: tuple[Player, ...],
    distribution: tuple[GoalTime, ...],
    injury_impact: float = INJURY_IMPACT,
) -> dict:
    squad = lambda team: tuple(p for p in players if p.team == team)  # noqa: E731
    home_squad, away_squad = squad(fixture.home), squad(fixture.away)
    home_avail, away_avail = team_availability(home_squad), team_availability(away_squad)

    lam, mu = params.expected_goals(fixture.home, fixture.away)
    matrix = matrix_from_rates(
        adjust_rate(lam, home_avail.missing_share, injury_impact),
        adjust_rate(mu, away_avail.missing_share, injury_impact),
        params.rho,
        MAX_GOALS,
    )
    summary = summarise(matrix, TOP_SCORELINES)
    home_scorers = assign_scorers(home_squad, summary.home_goals)
    away_scorers = assign_scorers(away_squad, summary.away_goals)

    return {
        "fixture_id": fixture.id,
        "kickoff": fixture.kickoff.isoformat(),
        "home": fixture.home,
        "away": fixture.away,
        "prediction": {
            "score": {"home": summary.home_goals, "away": summary.away_goals},
            "probability": round(summary.probability, 4),
            "goals": _goal_timeline(fixture, summary, home_scorers, away_scorers, distribution),
        },
        "probabilities": {
            "home_win": round(summary.home_win, 4),
            "draw": round(summary.draw, 4),
            "away_win": round(summary.away_win, 4),
        },
        "expected_goals": {
            "home": round(summary.expected_home_goals, 2),
            "away": round(summary.expected_away_goals, 2),
        },
        "top_scorelines": [
            {"score": f"{s.home_goals}-{s.away_goals}", "probability": round(s.probability, 4)}
            for s in summary.top_scorelines
        ],
        "missing": {"home": _missing_json(home_avail), "away": _missing_json(away_avail)},
    }


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


def predict_gameweek(gameweek: int, inputs: SeasonInputs, injury_impact: float = INJURY_IMPACT) -> dict:
    fixtures = fpl.parse_fixtures(inputs.raw_fixtures, inputs.bootstrap, gameweek)
    if not fixtures:
        raise ValueError(f"No fixtures with kickoff times found for gameweek {gameweek}")

    first_kickoff = min(f.kickoff for f in fixtures)
    cutoff = first_kickoff.tz_convert(None).normalize()
    params = fit(inputs.results, reference_date=cutoff)
    distribution = goal_distribution(inputs.goal_events, before=first_kickoff)

    return {
        "gameweek": gameweek,
        "predictor": PREDICTOR_NAME,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model": {
            "name": "dixon-coles",
            "trained_until": cutoff.date().isoformat(),
            "home_advantage": round(params.home_advantage, 3),
            "rho": round(params.rho, 3),
            "goal_time_samples": len(distribution),
            "injury_impact": injury_impact,
        },
        "matches": [
            predict_fixture(f, params, inputs.players, distribution, injury_impact)
            for f in sorted(fixtures, key=lambda f: f.kickoff)
        ],
    }


def prediction_path(gameweek: int, predictions_dir: Path = PREDICTIONS_DIR) -> Path:
    return predictions_dir / f"gw{gameweek:02d}.json"


def write_report(report: dict, predictions_dir: Path = PREDICTIONS_DIR) -> Path:
    path = prediction_path(report["gameweek"], predictions_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_gameweek(gameweek: int | None = None) -> dict:
    inputs = load_season_inputs()
    gw = gameweek or fpl.next_gameweek(inputs.bootstrap)
    report = predict_gameweek(gw, inputs)
    path = write_report(report)
    log.info("Wrote %s", path)
    return {**report, "path": str(path)}
