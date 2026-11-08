"""Build the FixtureContext for each fixture: Dixon-Coles baseline, squads, form."""

import pandas as pd

from football_ml.config import FORM_MATCHES, MAX_GOALS, TOP_SCORELINES
from football_ml.model.dixon_coles import DixonColesParams, matrix_from_rates
from football_ml.predict.availability import adjust_rate, team_availability
from football_ml.predict.scoreline import summarise
from football_ml.predict.scorers import Player, assign_scorers
from football_ml.predict.scoring_chance import ScorerModel, goal_rates
from football_ml.predict.timing import GoalTime, goal_times, order_goals
from football_ml.predictors.base import FixtureContext, FormResult, MatchPrediction, PredictedGoal
from football_ml.sources.football_data import MatchOdds
from football_ml.sources.fpl import Fixture


def recent_form(results: pd.DataFrame, team: str, before: pd.Timestamp, n: int = FORM_MATCHES) -> tuple[FormResult, ...]:
    """Last `n` results for `team` before `before`, newest first."""
    played = results[(results["date"] < before) & ((results["home"] == team) | (results["away"] == team))]
    latest = played.sort_values("date", ascending=False).head(n)

    def row_to_form(row) -> FormResult:
        at_home = row["home"] == team
        return FormResult(
            date=row["date"].date().isoformat(),
            opponent=row["away"] if at_home else row["home"],
            venue="H" if at_home else "A",
            goals_for=int(row["home_goals"] if at_home else row["away_goals"]),
            goals_against=int(row["away_goals"] if at_home else row["home_goals"]),
        )

    return tuple(row_to_form(row) for _, row in latest.iterrows())


def _baseline(fixture, summary, home_squad, away_squad, distribution) -> MatchPrediction:
    sides = order_goals(summary.home_goals, summary.away_goals)
    times = goal_times(len(sides), distribution, seed=fixture.id)
    scorers = {
        "home": assign_scorers(home_squad, summary.home_goals),
        "away": assign_scorers(away_squad, summary.away_goals),
    }

    def goal(k: int, side: str) -> PredictedGoal:
        nth = sides[:k].count(side)
        scorer = scorers[side][nth] if nth < len(scorers[side]) else None
        return PredictedGoal(side, times[k], scorer)

    return MatchPrediction(
        home_goals=summary.home_goals,
        away_goals=summary.away_goals,
        goals=tuple(goal(k, side) for k, side in enumerate(sides)),
    )


def fixture_context(
    gameweek: int,
    fixture: Fixture,
    params: DixonColesParams,
    players: tuple[Player, ...],
    distribution: tuple[GoalTime, ...],
    injury_impact: float,
    results: pd.DataFrame | None = None,
    odds: MatchOdds | None = None,
    scorer_model: ScorerModel | None = None,
) -> FixtureContext:
    home_squad = tuple(p for p in players if p.team == fixture.home)
    away_squad = tuple(p for p in players if p.team == fixture.away)
    home_avail, away_avail = team_availability(home_squad), team_availability(away_squad)

    lam, mu = params.expected_goals(fixture.home, fixture.away)
    matrix = matrix_from_rates(
        adjust_rate(lam, home_avail.missing_share, injury_impact),
        adjust_rate(mu, away_avail.missing_share, injury_impact),
        params.rho,
        MAX_GOALS,
    )
    summary = summarise(matrix, TOP_SCORELINES)
    before = fixture.kickoff.tz_convert(None).normalize()
    form = (lambda team: recent_form(results, team, before)) if results is not None else (lambda _: ())
    rates = (lambda squad, team, rate: goal_rates(scorer_model, squad, team, rate)) if scorer_model else (lambda *_: None)

    return FixtureContext(
        gameweek=gameweek,
        fixture=fixture,
        summary=summary,
        baseline=_baseline(fixture, summary, home_squad, away_squad, distribution),
        home_squad=home_squad,
        away_squad=away_squad,
        home_availability=home_avail,
        away_availability=away_avail,
        home_form=form(fixture.home),
        away_form=form(fixture.away),
        matrix=matrix,
        odds=odds,
        home_scoring_rates=rates(home_squad, fixture.home, summary.expected_home_goals),
        away_scoring_rates=rates(away_squad, fixture.away, summary.expected_away_goals),
    )
