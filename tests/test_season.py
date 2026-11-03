import json

import pytest

from football_ml.evaluate.season import leaderboard, score_gameweek, season_table, write_gameweek_results
from football_ml.sources.results import ActualMatch

REPORT = {
    "gameweek": 3,
    "predictor": "dixon-coles",
    "matches": [
        {"fixture_id": 1, "home": "Arsenal", "away": "Hull",
         "prediction": {"score": {"home": 2, "away": 0}, "goals": []}},
        {"fixture_id": 2, "home": "Leeds", "away": "Spurs",
         "prediction": {"score": {"home": 1, "away": 1}, "goals": []}},
    ],
}
ACTUAL = (
    ActualMatch(1, 3, "Arsenal", "Hull", 2, 0, ()),
)


def test_score_gameweek_only_scores_finished_matches():
    result = score_gameweek(REPORT, ACTUAL)
    assert result["complete"] is False
    assert len(result["matches"]) == 1
    assert result["matches"][0]["actual"] == "2-0"
    assert result["totals"]["points"] == 3
    assert result["totals"]["exact_scores"] == 1


def test_score_gameweek_complete_when_all_finished():
    actual = (*ACTUAL, ActualMatch(2, 3, "Leeds", "Spurs", 0, 2, ()))
    result = score_gameweek(REPORT, actual)
    assert result["complete"] is True
    assert result["totals"]["points"] == 3
    assert result["totals"]["matches"] == 2


def test_season_table_aggregates_by_predictor():
    gw = lambda n, pts: {"gameweek": n, "predictor": "dixon-coles", "totals": {  # noqa: E731
        "points": pts, "matches": 10, "exact_scores": 1, "correct_results": 4,
        "scorer_hits": 3, "minute_hits": 1}}
    table = season_table([gw(1, 10), gw(2, 7)])
    row = table["dixon-coles"]
    assert row["points"] == 17
    assert row["matches"] == 20
    assert row["by_gameweek"] == {"1": 10, "2": 7}


def test_write_gameweek_results_rebuilds_season(tmp_path):
    result = score_gameweek(REPORT, ACTUAL)
    write_gameweek_results(result, tmp_path)
    assert (tmp_path / "gw03" / "dixon-coles.json").exists()
    season = json.loads((tmp_path / "season.json").read_text())
    assert season["dixon-coles"]["points"] == 3


def _claude_report(home, away, fallback=False):
    return {**REPORT, "predictor": "claude", "matches": [
        {**REPORT["matches"][0], "prediction": {"score": {"home": home, "away": away}, "goals": [],
                                                "fallback": fallback}},
    ]}


def test_multiple_predictors_share_one_season_table(tmp_path):
    write_gameweek_results(score_gameweek(REPORT, ACTUAL), tmp_path)
    write_gameweek_results(score_gameweek(_claude_report(1, 0), ACTUAL), tmp_path)
    season = json.loads((tmp_path / "season.json").read_text())
    assert season["dixon-coles"]["points"] == 3
    assert season["claude"]["points"] == 1


def test_fallbacks_are_counted():
    result = score_gameweek(_claude_report(2, 0, fallback=True), ACTUAL)
    assert result["totals"]["fallbacks"] == 1
    assert season_table([result])["claude"]["fallbacks"] == 1


def test_season_table_tolerates_results_written_before_fallbacks_existed():
    old = {"gameweek": 1, "predictor": "dixon-coles", "totals": {
        "points": 5, "matches": 10, "exact_scores": 1, "correct_results": 4, "scorer_hits": 0, "minute_hits": 0}}
    assert season_table([old])["dixon-coles"]["fallbacks"] == 0


def test_leaderboard_ranks_by_points_then_exact_scores():
    row = lambda pts, exact: {"points": pts, "exact_scores": exact}  # noqa: E731
    table = {"a": row(10, 1), "b": row(12, 0), "c": row(10, 2)}
    assert [name for name, _ in leaderboard(table)] == ["b", "c", "a"]


def test_score_gameweek_rejects_report_without_matches():
    with pytest.raises(ValueError):
        score_gameweek({"gameweek": 1, "predictor": "x"}, ())
