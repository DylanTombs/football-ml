"""football-ml predict | score | backtest"""

import argparse
import json
import logging
import os
import sys

from football_ml.backtest import run_backtest
from football_ml.bench import run_bench, tune
from football_ml.model.structure import Structure
from football_ml.config import (
    AVAILABILITY_PROXIES,
    BENCH_DIR,
    BENCH_L2_GRID,
    BENCH_SEASON_SETS,
    BENCH_XI_GRID,
    DEFAULT_BACKTEST_PROXY,
    DEFAULT_PICK_STRATEGY,
    DEFAULT_PREDICTORS,
    DEFAULT_SCORER_MODEL,
    INJURY_IMPACT,
    PICK_STRATEGIES,
    PREDICTIONS_DIR,
    RATING_L2_PENALTY,
    RESULTS_DIR,
    SCORER_MODELS,
    SEASON_HOME_L2,
    SEASON_LEVEL_L2,
    TIME_DECAY_XI,
    TRAINING_SEASONS,
)
from football_ml.evaluate.season import leaderboard, score_gameweek, season_table, write_gameweek_results
from football_ml.llm.base import LLMError
from football_ml.pipeline import load_season_inputs, run_gameweek
from football_ml.predictors.registry import build_predictors, parse_predictor_names
from football_ml.sources import fpl
from football_ml.sources.football_data import fetch_results
from football_ml.sources.http import DataSourceError
from football_ml.sources.results import build_actual_matches


def _format_match(m: dict) -> str:
    score = m["prediction"]["score"]
    p = m["probabilities"]
    goals = ", ".join(
        f"{g['scorer'] or '?'} {g['minute']}" for g in m["prediction"]["goals"]
    ) or "no goals"
    return (
        f"{m['home']:>15} {score['home']}-{score['away']} {m['away']:<15}"
        f" H{p['home_win']:.0%} D{p['draw']:.0%} A{p['away_win']:.0%}  | {goals}"
    )


def _format_result(r: dict) -> str:
    pts = r["points"]
    return (
        f"{r['home']:>15} {r['predicted']} → {r['actual']} {r['away']:<15}"
        f" {pts['total']} pts (result {pts['outcome']}, scorers {pts['scorers']}, minutes {pts['minutes']})"
    )


def _totals_line(t: dict) -> str:
    return (
        f"{t['points']} pts | {t['exact_scores']} exact, {t['correct_results']}/{t['matches']} results,"
        f" {t['scorer_hits']} scorers, {t['minute_hits']} minutes"
        + (f", {t['fallbacks']} fallbacks" if t.get("fallbacks") else "")
        + (f", log loss {t['log_loss_sum'] / t['log_loss_matches']:.3f}" if t.get("log_loss_matches") else "")
    )


def _leaderboard_lines(table: dict) -> list[str]:
    return [f"{rank}. {name:<12} {_totals_line(row)}" for rank, (name, row) in enumerate(leaderboard(table), 1)]


def _variant_lines(main: dict, variant: dict) -> list[str]:
    """Plan 07 shadow variants: only the matches whose scoreline differs from the main prediction."""
    name = variant["predictor"].removeprefix(f"{variant['variant_of']}-")
    main_scores = {m["fixture_id"]: m["prediction"]["score"] for m in main["matches"]}
    lines = []
    for m in variant["matches"]:
        score, base = m["prediction"]["score"], main_scores.get(m["fixture_id"])
        if base and score != base:
            lines.append(f"  {name}: {m['home']} {score['home']}-{score['away']} {m['away']}"
                         f" (main {base['home']}-{base['away']})")
    return lines or [f"  {name}: same scorelines as the main prediction"]


def _predict(args) -> int:
    predictors = build_predictors(args.predictors, env=os.environ, pick_strategy=args.pick_strategy,
                                  scorer_model=args.scorer_model)
    reports = run_gameweek(args.gameweek, predictors)
    by_name = {r["predictor"]: r for r in reports}
    for report in (r for r in reports if not r.get("variant_of")):
        print(f"\nGameweek {report['gameweek']}: {report['predictor']}\n")
        for match in report["matches"]:
            print(_format_match(match))
        print(f"\nSaved to {report['path']}")
        if report.get("snapshot"):
            print(f"Player snapshot: {report['snapshot']}")
    variants = [r for r in reports if r.get("variant_of")]
    if variants:
        print("\nModel variants (shadow, plan 07), where they differ:")
        for variant in variants:
            print("\n".join(_variant_lines(by_name[variant["variant_of"]], variant)))
    return 0


def _score(args) -> int:
    paths = sorted((PREDICTIONS_DIR / f"gw{args.gameweek:02d}").glob("*.json"))
    if not paths:
        print(f"No predictions saved for GW{args.gameweek} in {PREDICTIONS_DIR}", file=sys.stderr)
        return 1
    inputs = load_season_inputs()
    actual = build_actual_matches(inputs.raw_fixtures, inputs.bootstrap, inputs.goal_events, args.gameweek)

    results = []
    for path in paths:
        result = score_gameweek(json.loads(path.read_text(encoding="utf-8")), actual)
        write_gameweek_results(result, RESULTS_DIR)
        results.append(result)
        status = "complete" if result["complete"] else "partial — some matches not finished"
        print(f"\nGameweek {args.gameweek}: {result['predictor']} ({status})\n")
        for row in result["matches"]:
            print(_format_result(row))

    print(f"\nGW{args.gameweek} leaderboard")
    print("\n".join(_leaderboard_lines(season_table(results))))
    season = json.loads((RESULTS_DIR / "season.json").read_text(encoding="utf-8"))
    print("\nSeason leaderboard")
    print("\n".join(_leaderboard_lines(season)))
    return 0


def _comparison_lines(c: dict) -> list[str]:
    def part(name: str) -> str:
        d = c["components"][name]
        lo, hi = d["ci95"]
        return f"{name} {d['difference']:+d} ({d['per_match']:+.3f}/match, 95% [{lo:+.2f}, {hi:+.2f}])"

    return [
        f"{c['challenger']} vs {c['reference']}: {c['matches']} matches, {c['changed_scorelines']} scorelines changed",
        "  " + " | ".join(part(name) for name in ("total", "outcome", "scorers", "minutes")),
    ]


def _backtest(args) -> int:
    inputs = load_season_inputs(with_histories=True)
    minutes = None
    if args.availability != "live":
        finished = tuple(gw for gw in fpl.finished_gameweeks(inputs.bootstrap) if gw <= args.last)
        minutes = fpl.fetch_gameweek_minutes(finished)
    summary = run_backtest(
        args.first, args.last, inputs,
        injury_impact=args.injury_impact, gameweek_minutes=minutes, proxy=args.availability,
        pick_strategy=args.pick_strategy,
    )
    scorers = "optimistic: live FPL stats" if summary.get("scorer_leak", True) else "leak-free: FPL history"
    print(
        f"\nBacktest (scorer stats {scorers};"
        f" injury impact {summary['injury_impact']}, availability {summary['availability_proxy']})\n"
    )
    for gw in summary["gameweeks"]:
        print(f"GW{gw['gameweek']:>2} {gw['predictor']:<28} {_totals_line(gw['totals'])}")
    print("\n" + "\n".join(_leaderboard_lines(summary["season"])))
    for comparison in summary.get("comparisons", []):
        print("\n" + "\n".join(_comparison_lines(comparison)))
    return 0


def _metrics_line(m: dict) -> str:
    return (
        f"{m['matches']} matches | exact log loss {m['exact_log_loss']:.4f} | 1X2 log loss {m['outcome_log_loss']:.4f}"
        f" | RPS {m['rps']:.4f} | outcome pts {m['outcome_points']} | goals {m['predicted_goals']:.0f}"
        f" vs {m['actual_goals']} | draws {m['predicted_draws']:.0f} vs {m['actual_draws']}"
    )


def _write_bench(name: str, data: dict) -> str:
    path = BENCH_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def _bench(args) -> int:
    seasons = BENCH_SEASON_SETS[args.seasons]
    results = fetch_results(TRAINING_SEASONS)
    if args.grid:
        table = tune(results, seasons, BENCH_XI_GRID, BENCH_L2_GRID, current=(args.xi, args.l2))
        for row in table["rows"]:
            mark = "*" if row["within_one_se"] else " "
            print(f"{mark} xi {row['xi']:<7} l2 {row['l2']:<5} {_metrics_line(row)}")
        print(f"\nbest xi {table['best']['xi']} l2 {table['best']['l2']}; chosen xi {table['chosen']['xi']} l2 {table['chosen']['l2']}"
              f" (* = within one SE of the best)")
        print(f"Saved to {_write_bench(f'grid-{args.seasons}', table)}")
        return 0
    structure = Structure(per_league=args.per_league, season_home=args.season_home, season_home_l2=args.season_home_l2,
                          season_level=args.season_level, season_level_l2=args.season_level_l2)
    summary = run_bench(results, seasons, xi=args.xi, l2=args.l2, structure=structure)
    print(f"\nBench {args.seasons} {', '.join(seasons)} ({summary['label']})")
    print(_metrics_line(summary["metrics"]))
    path = _write_bench(f"{args.seasons}-{summary['label']}", summary)
    print(f"Saved to {path}")
    return 0


def _gameweek_arg(value: str) -> int:
    gw = int(value)
    if not 1 <= gw <= 38:
        raise argparse.ArgumentTypeError("gameweek must be between 1 and 38")
    return gw


def _predictors_arg(value: str) -> tuple[str, ...]:
    try:
        return parse_predictor_names(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _impact_arg(value: str) -> float:
    impact = float(value)
    if not 0 <= impact <= 1:
        raise argparse.ArgumentTypeError("injury impact must be between 0 and 1")
    return impact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="football-ml")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="Predict every fixture in a gameweek")
    predict.add_argument("--gameweek", type=_gameweek_arg, help="defaults to the next gameweek")
    predict.add_argument("--predictors", type=_predictors_arg, default=DEFAULT_PREDICTORS,
                         help=f"comma-separated, e.g. dixon-coles,claude,gpt (default {','.join(DEFAULT_PREDICTORS)})")
    predict.add_argument("--pick-strategy", choices=PICK_STRATEGIES, default=DEFAULT_PICK_STRATEGY,
                         help=f"how Dixon-Coles picks its scoreline (default {DEFAULT_PICK_STRATEGY})")
    predict.add_argument("--scorer-model", choices=SCORER_MODELS, default=DEFAULT_SCORER_MODEL,
                         help=f"how expected-points picks choose scorers (default {DEFAULT_SCORER_MODEL})")
    predict.set_defaults(handler=_predict)

    score = sub.add_parser("score", help="Score saved predictions against actual results")
    score.add_argument("--gameweek", type=_gameweek_arg, required=True)
    score.set_defaults(handler=_score)

    backtest = sub.add_parser("backtest", help="Re-predict and score past gameweeks")
    backtest.add_argument("--from", dest="first", type=_gameweek_arg, required=True)
    backtest.add_argument("--to", dest="last", type=_gameweek_arg, required=True)
    backtest.add_argument("--injury-impact", type=_impact_arg, default=INJURY_IMPACT,
                          help=f"0 = ignore injuries (default {INJURY_IMPACT})")
    backtest.add_argument("--availability", choices=AVAILABILITY_PROXIES, default=DEFAULT_BACKTEST_PROXY,
                          help="how to tell who was missing in past gameweeks")
    backtest.add_argument("--pick-strategy", choices=PICK_STRATEGIES, default=DEFAULT_PICK_STRATEGY,
                          help="strategy compared against modal picks")
    backtest.set_defaults(handler=_backtest)

    bench = sub.add_parser("bench", help="Walk-forward over past seasons, scored by log loss (plan 06)")
    bench.add_argument("--seasons", choices=sorted(BENCH_SEASON_SETS), default="tune",
                       help="tune = 2021-22 to 2024-25; holdout = 2025-26 (test once)")
    bench.add_argument("--xi", type=float, default=TIME_DECAY_XI, help="time decay per day")
    bench.add_argument("--l2", type=float, default=RATING_L2_PENALTY, help="ridge penalty on ratings")
    bench.add_argument("--grid", action="store_true", help="tune xi and l2 over the grid in config")
    bench.add_argument("--per-league", action="store_true", help="plan 07: own scoring level + home edge per league")
    bench.add_argument("--season-home", action="store_true", help="plan 07: home advantage per season")
    bench.add_argument("--season-home-l2", type=float, default=SEASON_HOME_L2,
                       help=f"ridge on each season's home advantage (default {SEASON_HOME_L2})")
    bench.add_argument("--season-level", action="store_true", help="plan 07: scoring level per season")
    bench.add_argument("--season-level-l2", type=float, default=SEASON_LEVEL_L2,
                       help=f"ridge on each season's scoring level (default {SEASON_LEVEL_L2})")
    bench.set_defaults(handler=_bench)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return args.handler(args)
    except (DataSourceError, LLMError, KeyError, ValueError, RuntimeError) as exc:
        print(f"{args.command} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
