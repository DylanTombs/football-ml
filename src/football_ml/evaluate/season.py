"""Gameweek results and the running season table."""

import json
from pathlib import Path

from football_ml.evaluate.points import score_prediction
from football_ml.sources.results import ActualMatch

_TOTAL_KEYS = (
    "points", "matches", "exact_scores", "correct_results", "scorer_hits", "minute_hits", "fallbacks",
)


def _match_row(predicted: dict, actual: ActualMatch) -> dict:
    pts = score_prediction(predicted, actual)
    score = predicted["prediction"]["score"]
    return {
        "fixture_id": actual.fixture_id,
        "home": actual.home,
        "away": actual.away,
        "predicted": f"{score['home']}-{score['away']}",
        "actual": f"{actual.home_goals}-{actual.away_goals}",
        "predicted_scorers": [g.get("scorer") for g in predicted["prediction"]["goals"]],
        "actual_scorers": [
            f"{g.scorer_name}{' (og)' if g.own_goal else ''} {g.minute.label if g.minute else '?'}"
            for g in actual.goals
        ],
        "points": {
            "outcome": pts.outcome_points,
            "scorers": pts.scorer_points,
            "minutes": pts.minute_points,
            "total": pts.total,
        },
        "_hits": {"exact": pts.exact, "result": pts.correct_result,
                  "scorers": pts.scorer_hits, "minutes": pts.minute_hits,
                  "fallback": bool(predicted["prediction"].get("fallback", False))},
    }


def score_gameweek(report: dict, actual: tuple[ActualMatch, ...]) -> dict:
    """Score every finished match of a prediction report."""
    if not report.get("matches"):
        raise ValueError(f"Prediction report for GW{report.get('gameweek')} has no matches")

    by_id = {m.fixture_id: m for m in actual}
    rows = [_match_row(p, by_id[p["fixture_id"]]) for p in report["matches"] if p["fixture_id"] in by_id]
    totals = {
        "points": sum(r["points"]["total"] for r in rows),
        "matches": len(rows),
        "exact_scores": sum(r["_hits"]["exact"] for r in rows),
        "correct_results": sum(r["_hits"]["result"] for r in rows),
        "scorer_hits": sum(r["_hits"]["scorers"] for r in rows),
        "minute_hits": sum(r["_hits"]["minutes"] for r in rows),
        "fallbacks": sum(r["_hits"]["fallback"] for r in rows),
    }
    return {
        "gameweek": report["gameweek"],
        "predictor": report["predictor"],
        "complete": len(rows) == len(report["matches"]),
        "totals": totals,
        "matches": [{k: v for k, v in r.items() if k != "_hits"} for r in rows],
    }


def season_table(gameweek_results: list[dict]) -> dict:
    """Totals per predictor across gameweeks."""
    table: dict[str, dict] = {}
    for result in sorted(gameweek_results, key=lambda r: r["gameweek"]):
        name = result["predictor"]
        previous = table.get(name, {**{k: 0 for k in _TOTAL_KEYS}, "by_gameweek": {}})
        table = {
            **table,
            name: {
                **{k: previous[k] + result["totals"].get(k, 0) for k in _TOTAL_KEYS},
                "by_gameweek": {**previous["by_gameweek"], str(result["gameweek"]): result["totals"]["points"]},
            },
        }
    return table


def leaderboard(table: dict) -> list[tuple[str, dict]]:
    """Predictors best first: points, then exact scores as the tie-break."""
    return sorted(table.items(), key=lambda item: (-item[1]["points"], -item[1]["exact_scores"]))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_gameweek_results(result: dict, results_dir: Path) -> Path:
    """Write gwNN/<predictor>.json and rebuild season.json from every result on disk."""
    path = results_dir / f"gw{result['gameweek']:02d}" / f"{result['predictor']}.json"
    _write_json(path, result)
    all_results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(results_dir.glob("gw*/*.json"))]
    _write_json(results_dir / "season.json", season_table(all_results))
    return path
