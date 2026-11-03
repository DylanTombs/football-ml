"""Weekly pipeline: fetch → fit → predict every fixture in a gameweek → JSON."""

import json
import logging

import pandas as pd

from football_ml.config import MAX_GOALS, PREDICTIONS_DIR, TOP_SCORELINES, TRAINING_SEASONS
from football_ml.model.dixon_coles import DixonColesParams, fit, score_matrix
from football_ml.predict.scoreline import summarise
from football_ml.predict.scorers import Player, assign_scorers
from football_ml.predict.timing import GoalTime, goal_times, order_goals, parse_clock
from football_ml.sources import espn, fpl
from football_ml.sources.football_data import fetch_results

log = logging.getLogger(__name__)

_FALLBACK_DISTRIBUTION = tuple(GoalTime(float(m), f"{m}'") for m in range(1, 91))


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
        }

    return [goal(k, side) for k, side in enumerate(sides)]


def predict_fixture(
    fixture: fpl.Fixture,
    params: DixonColesParams,
    players: tuple[Player, ...],
    distribution: tuple[GoalTime, ...],
) -> dict:
    summary = summarise(score_matrix(params, fixture.home, fixture.away, MAX_GOALS), TOP_SCORELINES)
    squad = lambda team: tuple(p for p in players if p.team == team)  # noqa: E731
    home_scorers = assign_scorers(squad(fixture.home), summary.home_goals)
    away_scorers = assign_scorers(squad(fixture.away), summary.away_goals)

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
    }


def _goal_distribution(bootstrap: dict, until: pd.Timestamp) -> tuple[GoalTime, ...]:
    clocks = espn.fetch_goal_clocks(fpl.season_start(bootstrap), until)
    parsed = []
    for raw in clocks:
        try:
            parsed.append(parse_clock(raw))
        except ValueError:
            log.warning("Ignoring unparseable goal clock %r", raw)
    if not parsed:
        log.warning("No ESPN goal times available; falling back to uniform 1'-90'")
        return _FALLBACK_DISTRIBUTION
    return tuple(parsed)


def run_gameweek(gameweek: int | None = None) -> dict:
    bootstrap = fpl.fetch_bootstrap()
    gw = gameweek or fpl.next_gameweek(bootstrap)
    fixtures = fpl.parse_fixtures(fpl.fetch_raw_fixtures(), bootstrap, gw)
    if not fixtures:
        raise ValueError(f"No fixtures with kickoff times found for gameweek {gw}")

    first_kickoff = min(f.kickoff for f in fixtures)
    cutoff = first_kickoff.tz_convert(None).normalize()
    params = fit(fetch_results(TRAINING_SEASONS), reference_date=cutoff)
    players = fpl.parse_players(bootstrap)
    distribution = _goal_distribution(bootstrap, until=first_kickoff)

    report = {
        "gameweek": gw,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model": {
            "name": "dixon-coles",
            "trained_until": cutoff.date().isoformat(),
            "home_advantage": round(params.home_advantage, 3),
            "rho": round(params.rho, 3),
            "goal_time_samples": len(distribution),
        },
        "matches": [
            predict_fixture(f, params, players, distribution)
            for f in sorted(fixtures, key=lambda f: f.kickoff)
        ],
    }
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"gw{gw:02d}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", path)
    return {**report, "path": str(path)}
