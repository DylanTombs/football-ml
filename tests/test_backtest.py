import json

import pytest
from datetime import timedelta

import pandas as pd

from football_ml.backtest import run_backtest
from dataclasses import replace

from football_ml.pipeline import SeasonInputs, gameweek_context, predict_gameweek, run_gameweek
from football_ml.predictors.dixon_coles import DixonColesPredictor
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
    summary = run_backtest(1, 3, _inputs(), out_dir=tmp_path, proxy="live")
    modal = [g for g in summary["gameweeks"] if g["predictor"] == "dixon-coles-modal"]
    assert [g["gameweek"] for g in modal] == [1, 2, 3]
    assert (tmp_path / "predictions" / "gw02" / "dixon-coles-modal.json").exists()
    assert (tmp_path / "predictions" / "gw02" / "naive-favourite.json").exists()
    assert (tmp_path / "results" / "season.json").exists()
    assert summary["scorer_leak"] is True
    # home always wins 2-0 in history and in "actual" results → model should get results right
    assert all(g["totals"]["correct_results"] == 2 for g in modal)


def test_backtest_records_proxy_and_impact(tmp_path):
    minutes = {1: {1: 90}, 2: {1: 0}, 3: {1: 90}}
    summary = run_backtest(1, 3, _inputs(), out_dir=tmp_path, injury_impact=0.5,
                           gameweek_minutes=minutes, proxy="lagged")
    assert summary["availability_proxy"] == "lagged"
    assert summary["injury_impact"] == 0.5
    # Saka played GW1 then missed GW2 → lagged proxy has him out for GW3 only
    gw3 = json.loads((tmp_path / "predictions" / "gw03" / "dixon-coles-modal.json").read_text())
    gw2 = json.loads((tmp_path / "predictions" / "gw02" / "dixon-coles-modal.json").read_text())
    assert gw3["matches"][0]["missing"]["home"] == [{"name": "Saka", "xg_share": 1.0}]
    assert gw2["matches"][0]["missing"]["home"] == []


def test_backtest_needs_minutes_for_historical_proxies(tmp_path):
    with pytest.raises(ValueError, match="minutes"):
        run_backtest(1, 3, _inputs(), out_dir=tmp_path, proxy="lagged")


def _odds(gameweek, home, away, odds_home, odds_away):
    day = (START + timedelta(weeks=gameweek - 1, days=1)).tz_convert(None).normalize()
    return {"date": day, "home": home, "away": away, "odds_home": odds_home, "odds_draw": 3.5,
            "odds_away": odds_away, "odds_over": 2.0, "odds_under": 1.8}


def test_gameweek_context_attaches_pre_closing_odds():
    odds = pd.DataFrame([_odds(2, "Arsenal", "Hull", 1.5, 6.0)])
    gw = gameweek_context(2, replace(_inputs(), odds=odds))
    by_home = {ctx.fixture.home: ctx for ctx in gw.fixtures}
    assert by_home["Arsenal"].odds.home == 1.5
    assert by_home["Leeds"].odds is None


def test_backtest_compares_strategies_and_benchmark(tmp_path):
    odds = pd.DataFrame([_odds(gw, h, a, 4.0, 1.8) for gw in (1, 2, 3)
                         for h, a in (("Arsenal", "Hull"), ("Leeds", "Tottenham"))])
    summary = run_backtest(1, 3, replace(_inputs(), odds=odds), out_dir=tmp_path, proxy="live",
                           pick_strategy="expected-points")
    names = ["dixon-coles-modal", "dixon-coles-expected-points", "naive-favourite"]
    assert summary["predictors"] == names
    assert set(summary["season"]) == set(names)
    assert [c["challenger"] for c in summary["comparisons"]] == names[1:]
    assert all(c["reference"] == "dixon-coles-modal" for c in summary["comparisons"])
    assert json.loads((tmp_path / "comparisons.json").read_text()) == summary["comparisons"]
    # the market (wrongly) backs the away side every time; home always wins 2-0
    assert summary["season"]["naive-favourite"]["correct_results"] == 0


def test_run_gameweek_writes_a_snapshot_before_the_deadline(monkeypatch, tmp_path):
    from football_ml import pipeline
    monkeypatch.setattr(pipeline, "load_season_inputs", _inputs)
    reports = run_gameweek(2, (DixonColesPredictor(),), predictions_dir=tmp_path / "predictions",
                           snapshots_dir=tmp_path / "snapshots", now=START)
    assert (tmp_path / "predictions" / "gw02" / "dixon-coles.json").exists()
    assert (tmp_path / "snapshots" / "gw02.json").exists()
    assert reports[0]["snapshot"] == str(tmp_path / "snapshots" / "gw02.json")


def test_run_gameweek_skips_the_snapshot_after_the_deadline(monkeypatch, tmp_path):
    from football_ml import pipeline
    monkeypatch.setattr(pipeline, "load_season_inputs", _inputs)
    reports = run_gameweek(2, (DixonColesPredictor(),), predictions_dir=tmp_path / "predictions",
                           snapshots_dir=tmp_path / "snapshots", now=START + timedelta(weeks=2))
    assert not (tmp_path / "snapshots").exists()
    assert reports[0]["snapshot"] is None
