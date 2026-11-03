from datetime import timedelta

import pandas as pd

from football_ml.backtest import run_backtest
from football_ml.pipeline import SeasonInputs, predict_gameweek
from football_ml.predict.scorers import Player
from football_ml.sources.espn import GoalEvent

START = pd.Timestamp("2026-08-01T15:00Z")
TEAMS = ["Arsenal", "Hull", "Leeds", "Tottenham"]
BOOTSTRAP = {
    "teams": [{"id": i + 1, "name": t} for i, t in enumerate(TEAMS)],
    "elements": [],
    "events": [{"id": gw, "deadline_time": (START + timedelta(weeks=gw - 1)).isoformat()}
               for gw in (1, 2, 3)],
}


def _raw_fixtures():
    rows = []
    for gw in (1, 2, 3):
        kickoff = START + timedelta(weeks=gw - 1, days=1)
        for fid, (h, a) in enumerate([(1, 2), (3, 4)]):
            rows.append({
                "id": gw * 10 + fid, "event": gw, "team_h": h, "team_a": a,
                "kickoff_time": kickoff.isoformat(), "finished": True,
                "team_h_score": 2, "team_a_score": 0,
                "stats": [{"identifier": "goals_scored", "h": [], "a": []},
                          {"identifier": "own_goals", "h": [], "a": []}],
            })
    return rows


def _history():
    """Two seasons of results where home always wins 2-0."""
    dates = pd.date_range("2024-08-01", "2026-07-01", freq="7D")
    return pd.DataFrame([
        {"date": d, "home": TEAMS[i % 4], "away": TEAMS[(i + 1) % 4], "home_goals": 2, "away_goals": 0}
        for i, d in enumerate(dates)
    ])


def _inputs(events=()):
    return SeasonInputs(
        bootstrap=BOOTSTRAP,
        raw_fixtures=_raw_fixtures(),
        results=_history(),
        players=(Player(1, "Saka", "Arsenal", 3.0, 3, 500, True),),
        goal_events=tuple(events),
    )


def test_predict_gameweek_ignores_goal_events_after_kickoff():
    late = GoalEvent(START + timedelta(weeks=5), "Arsenal", "Hull", "Arsenal", "Saka", "88'", False)
    early = GoalEvent(START - timedelta(days=3), "Arsenal", "Hull", "Arsenal", "Saka", "12'", False)
    report = predict_gameweek(2, _inputs([late, early]))
    assert report["model"]["goal_time_samples"] == 1
    assert report["predictor"] == "dixon-coles"
    assert len(report["matches"]) == 2


def test_predict_gameweek_falls_back_to_uniform_goal_times():
    report = predict_gameweek(1, _inputs())
    assert report["model"]["goal_time_samples"] == 90


def test_run_backtest_scores_each_gameweek(tmp_path):
    summary = run_backtest(1, 3, _inputs(), out_dir=tmp_path)
    assert [g["gameweek"] for g in summary["gameweeks"]] == [1, 2, 3]
    assert (tmp_path / "predictions" / "gw02.json").exists()
    assert (tmp_path / "results" / "season.json").exists()
    assert summary["scorer_leak"] is True
    # home always wins 2-0 in history and in "actual" results → model should get results right
    assert all(g["totals"]["correct_results"] == 2 for g in summary["gameweeks"])
