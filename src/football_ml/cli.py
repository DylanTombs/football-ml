"""football-ml predict | score | backtest"""

import argparse
import json
import logging
import sys

from football_ml.backtest import run_backtest
from football_ml.config import (
    AVAILABILITY_PROXIES,
    DEFAULT_BACKTEST_PROXY,
    INJURY_IMPACT,
    RESULTS_DIR,
)
from football_ml.evaluate.season import score_gameweek, write_gameweek_results
from football_ml.pipeline import load_season_inputs, prediction_path, run_gameweek
from football_ml.sources import fpl
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
    )


def _predict(args) -> int:
    report = run_gameweek(args.gameweek)
    print(f"\nGameweek {report['gameweek']} predictions\n")
    for match in report["matches"]:
        print(_format_match(match))
    print(f"\nSaved to {report['path']}")
    return 0


def _score(args) -> int:
    path = prediction_path(args.gameweek)
    if not path.exists():
        print(f"No predictions saved for GW{args.gameweek} ({path})", file=sys.stderr)
        return 1
    report = json.loads(path.read_text(encoding="utf-8"))
    inputs = load_season_inputs()
    actual = build_actual_matches(inputs.raw_fixtures, inputs.bootstrap, inputs.goal_events, args.gameweek)
    result = score_gameweek(report, actual)
    out = write_gameweek_results(result, RESULTS_DIR)

    status = "complete" if result["complete"] else "partial — some matches not finished"
    print(f"\nGameweek {args.gameweek} results ({status})\n")
    for row in result["matches"]:
        print(_format_result(row))
    print(f"\n{_totals_line(result['totals'])}\nSaved to {out}")
    return 0


def _backtest(args) -> int:
    inputs = load_season_inputs()
    minutes = None
    if args.availability != "live":
        finished = tuple(gw for gw in fpl.finished_gameweeks(inputs.bootstrap) if gw <= args.last)
        minutes = fpl.fetch_gameweek_minutes(finished)
    summary = run_backtest(
        args.first, args.last, inputs,
        injury_impact=args.injury_impact, gameweek_minutes=minutes, proxy=args.availability,
    )
    print(
        f"\nBacktest (scorer points optimistic: live FPL stats;"
        f" injury impact {summary['injury_impact']}, availability {summary['availability_proxy']})\n"
    )
    for gw in summary["gameweeks"]:
        print(f"GW{gw['gameweek']:>2}: {_totals_line(gw['totals'])}")
    for name, row in summary["season"].items():
        print(f"\n{name}: {_totals_line(row)}")
    return 0


def _gameweek_arg(value: str) -> int:
    gw = int(value)
    if not 1 <= gw <= 38:
        raise argparse.ArgumentTypeError("gameweek must be between 1 and 38")
    return gw


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
    backtest.set_defaults(handler=_backtest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return args.handler(args)
    except (DataSourceError, KeyError, ValueError, RuntimeError) as exc:
        print(f"{args.command} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
