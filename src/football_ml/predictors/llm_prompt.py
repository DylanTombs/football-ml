"""The prompt and JSON schema given to every LLM predictor."""

from football_ml.config import (
    MAX_GOALS,
    MINUTE_TOLERANCE,
    POINTS_CORRECT_RESULT,
    POINTS_EXACT_SCORE,
    POINTS_PER_MINUTE,
    POINTS_PER_SCORER,
)
from football_ml.predict.scorers import Player
from football_ml.predictors.base import FixtureContext, FormResult

SYSTEM_PROMPT = (
    "You predict Premier League matches for a weekly prediction league. "
    "You are given a statistical baseline from a Dixon-Coles model. Start from it and adjust only "
    "where you have a concrete reason the model can't see: form, injuries and suspensions, likely "
    "rotation, a manager change, a derby, fixture congestion. If you have no such reason, keep the "
    "baseline. Pick scorers only from the listed available players, using their ids. "
    "Give minutes as match clocks like \"23'\", \"45+2'\" or \"90+3'\". "
    "Keep the reasoning to one or two sentences."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "home_goals": {"type": "integer"},
        "away_goals": {"type": "integer"},
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "enum": ["home", "away"]},
                    "scorer_id": {"type": "integer"},
                    "minute": {"type": "string"},
                },
                "required": ["team", "scorer_id", "minute"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reasoning", "home_goals", "away_goals", "goals"],
    "additionalProperties": False,
}


def scorer_choices(squad: tuple[Player, ...]) -> tuple[Player, ...]:
    """Players an LLM may pick: available, still at the club, and has played."""
    return tuple(p for p in squad if p.available and not p.departed and p.minutes > 0)


def _rules() -> str:
    return (
        f"Scoring: exact score {POINTS_EXACT_SCORE} points, otherwise correct result {POINTS_CORRECT_RESULT} point. "
        f"+{POINTS_PER_SCORER} for each predicted scorer who scores, and +{POINTS_PER_MINUTE} more if that "
        f"goal's minute is within {MINUTE_TOLERANCE} of the real one. Each team scores 0-{MAX_GOALS}; "
        "list exactly one goal entry per goal."
    )


def _form_line(form: tuple[FormResult, ...]) -> str:
    if not form:
        return "no recent results"
    return ", ".join(f"{f.result} {f.goals_for}-{f.goals_against} {f.venue} v {f.opponent}" for f in form)


def _players_block(squad: tuple[Player, ...]) -> str:
    rows = sorted(scorer_choices(squad), key=lambda p: (-p.xg, -p.goals))
    lines = [
        f"  id {p.id}: {p.name} (xG {p.xg:.2f}, goals {p.goals}, minutes {p.minutes}"
        + (f", {p.availability:.0%} chance of playing" if p.availability < 1 else "") + ")"
        for p in rows
    ]
    return "\n".join(lines) or "  (none listed)"


def _team_block(label: str, team: str, squad, availability, form) -> str:
    missing = ", ".join(f"{m.name} ({m.xg_share:.0%} of team xG)" for m in availability.missing) or "none"
    return (
        f"{label}: {team}\n"
        f"Last results (newest first): {_form_line(form)}\n"
        f"Out or doubtful: {missing}\n"
        f"Available players (season stats):\n{_players_block(squad)}"
    )


def _baseline_block(ctx: FixtureContext) -> str:
    s, b = ctx.summary, ctx.baseline
    goals = ", ".join(
        f"{(g.scorer.name if g.scorer else 'unknown')} {g.minute.label} ({g.side})"
        for g in sorted(b.goals, key=lambda g: g.minute.sort_key)
    ) or "no goals"
    top = ", ".join(f"{t.home_goals}-{t.away_goals} ({t.probability:.0%})" for t in s.top_scorelines)
    return (
        "Dixon-Coles baseline:\n"
        f"  home win {s.home_win:.0%}, draw {s.draw:.0%}, away win {s.away_win:.0%}\n"
        f"  expected goals {s.expected_home_goals:.2f} - {s.expected_away_goals:.2f}\n"
        f"  most likely scorelines: {top}\n"
        f"  its prediction: {b.home_goals}-{b.away_goals}; goals: {goals}"
    )


def build_prompt(ctx: FixtureContext) -> str:
    f = ctx.fixture
    return "\n\n".join([
        f"Gameweek {ctx.gameweek}: {f.home} (home) v {f.away} (away), kickoff {f.kickoff.isoformat()}.",
        _rules(),
        _baseline_block(ctx),
        _team_block("Home", f.home, ctx.home_squad, ctx.home_availability, ctx.home_form),
        _team_block("Away", f.away, ctx.away_squad, ctx.away_availability, ctx.away_form),
        "Predict the final score, who scores each goal and when.",
    ])
