"""`uv run football-ml predict [--gameweek N]`"""

import argparse
import logging
import sys

from football_ml.pipeline import run_gameweek
from football_ml.sources.http import DataSourceError


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="football-ml")
    sub = parser.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict", help="Predict every fixture in a gameweek")
    predict.add_argument("--gameweek", type=int, help="defaults to the next gameweek")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.gameweek is not None and not 1 <= args.gameweek <= 38:
        parser.error("--gameweek must be between 1 and 38")

    try:
        report = run_gameweek(args.gameweek)
    except (DataSourceError, ValueError, RuntimeError) as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nGameweek {report['gameweek']} predictions\n")
    for match in report["matches"]:
        print(_format_match(match))
    print(f"\nSaved to {report['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
