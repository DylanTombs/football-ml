import json

import pytest
import requests

from football_ml import cli
from football_ml.sources import http


class _Response:
    def __init__(self, text: str, status: int = 200):
        self.text, self.status_code, self.encoding = text, status, "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(http, "CACHE_DIR", tmp_path)
    return tmp_path


def test_fetch_writes_then_serves_cache(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: calls.append(1) or _Response("hi"))
    assert http.fetch_text("u", "x/a.txt", max_age_hours=None) == "hi"
    assert http.fetch_text("u", "x/a.txt", max_age_hours=None) == "hi"
    assert len(calls) == 1
    assert (cache / "x" / "a.txt").read_text() == "hi"


def test_fetch_refetches_when_stale(cache, monkeypatch):
    (cache / "a.txt").write_text("old")
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: _Response("new"))
    assert http.fetch_text("u", "a.txt", max_age_hours=0) == "new"


def test_fetch_falls_back_to_stale_cache_on_error(cache, monkeypatch):
    (cache / "a.txt").write_text("old")
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: _Response("", status=500))
    assert http.fetch_text("u", "a.txt", max_age_hours=0) == "old"


def test_fetch_raises_without_cache(cache, monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(http.requests, "get", boom)
    with pytest.raises(http.DataSourceError, match="offline"):
        http.fetch_text("u", "missing.txt", max_age_hours=None)


MATCH = {
    "home": "Arsenal", "away": "Hull",
    "prediction": {"score": {"home": 2, "away": 0},
                   "goals": [{"scorer": "Saka", "minute": "12'"}, {"scorer": None, "minute": "80'"}]},
    "probabilities": {"home_win": 0.6, "draw": 0.25, "away_win": 0.15},
}


def test_format_match():
    line = cli._format_match(MATCH)
    assert "2-0" in line and "Saka 12'" in line and "? 80'" in line and "H60%" in line


def test_cli_rejects_bad_gameweek():
    with pytest.raises(SystemExit):
        cli.main(["predict", "--gameweek", "50"])


def test_cli_reports_failure(monkeypatch, capsys):
    def fail(*_):
        raise http.DataSourceError("FPL down")
    monkeypatch.setattr(cli, "run_gameweek", fail)
    assert cli.main(["predict"]) == 1
    assert "FPL down" in capsys.readouterr().err


def test_cli_prints_report(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_gameweek", lambda gw, predictors: [
        {"gameweek": 10, "predictor": p.name, "matches": [MATCH], "path": "p"} for p in predictors])
    assert cli.main(["predict", "--gameweek", "10"]) == 0
    assert "Gameweek 10: dixon-coles" in capsys.readouterr().out


def test_cli_predict_checks_keys_before_running(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "run_gameweek", lambda *a: pytest.fail("ran without a key"))
    assert cli.main(["predict", "--predictors", "dixon-coles,claude"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_cli_rejects_unknown_predictor():
    with pytest.raises(SystemExit):
        cli.main(["predict", "--predictors", "dixon-coles,oracle"])


def _report(predictor, home, away):
    return {"gameweek": 3, "predictor": predictor, "matches": [
        {"fixture_id": 1, "home": "Arsenal", "away": "Hull",
         "prediction": {"score": {"home": home, "away": away}, "goals": []}}]}


def test_cli_scores_every_predictor_into_one_leaderboard(monkeypatch, tmp_path, capsys):
    from football_ml.sources.results import ActualMatch
    preds, results = tmp_path / "predictions", tmp_path / "results"
    (preds / "gw03").mkdir(parents=True)
    (preds / "gw03" / "dixon-coles.json").write_text(json.dumps(_report("dixon-coles", 1, 0)))
    (preds / "gw03" / "claude.json").write_text(json.dumps(_report("claude", 2, 0)))
    monkeypatch.setattr(cli, "PREDICTIONS_DIR", preds)
    monkeypatch.setattr(cli, "RESULTS_DIR", results)
    monkeypatch.setattr(cli, "load_season_inputs", lambda: type("I", (), {
        "raw_fixtures": [], "bootstrap": {}, "goal_events": ()})())
    monkeypatch.setattr(cli, "build_actual_matches",
                        lambda *a: (ActualMatch(1, 3, "Arsenal", "Hull", 2, 0, ()),))
    assert cli.main(["score", "--gameweek", "3"]) == 0
    out = capsys.readouterr().out
    assert out.index("1. claude") < out.index("2. dixon-coles")
    assert (results / "gw03" / "claude.json").exists()


def test_cli_score_without_predictions(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "PREDICTIONS_DIR", tmp_path)
    assert cli.main(["score", "--gameweek", "4"]) == 1
    assert "No predictions" in capsys.readouterr().err


def _fake_inputs(**_):
    return type("Inputs", (), {"bootstrap": {"events": [{"id": 1, "finished": True},
                                                         {"id": 2, "finished": False}]}})()


def test_cli_backtest_passes_flags_and_fetches_minutes(monkeypatch, capsys):
    seen = {}

    def fake_backtest(first, last, inputs, **kwargs):
        seen.update(kwargs, first=first, last=last)
        return {"gameweeks": [], "season": {}, "comparisons": [], "availability_proxy": kwargs["proxy"],
                "injury_impact": kwargs["injury_impact"]}

    monkeypatch.setattr(cli, "load_season_inputs", _fake_inputs)
    monkeypatch.setattr(cli.fpl, "fetch_gameweek_minutes", lambda gws: {gw: {} for gw in gws})
    monkeypatch.setattr(cli, "run_backtest", fake_backtest)
    argv = ["backtest", "--from", "1", "--to", "1", "--injury-impact", "0.25", "--availability", "oracle",
            "--pick-strategy", "expected-points"]
    assert cli.main(argv) == 0
    assert seen["pick_strategy"] == "expected-points"
    assert seen["injury_impact"] == 0.25
    assert seen["proxy"] == "oracle"
    assert seen["gameweek_minutes"] == {1: {}}
    assert "oracle" in capsys.readouterr().out


def test_cli_backtest_live_proxy_skips_minutes(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "load_season_inputs", _fake_inputs)
    monkeypatch.setattr(cli.fpl, "fetch_gameweek_minutes", lambda gws: pytest.fail("fetched"))
    monkeypatch.setattr(cli, "run_backtest", lambda f, l, i, **kw: seen.update(kw) or
                        {"gameweeks": [], "season": {}, "availability_proxy": "live", "injury_impact": 0.5})
    assert cli.main(["backtest", "--from", "1", "--to", "1", "--availability", "live"]) == 0
    assert seen["gameweek_minutes"] is None


def test_cli_rejects_bad_injury_impact():
    with pytest.raises(SystemExit):
        cli.main(["backtest", "--from", "1", "--to", "2", "--injury-impact", "1.5"])


def test_cli_predict_passes_pick_strategy(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "run_gameweek", lambda gw, predictors: seen.update(p=predictors) or [])
    assert cli.main(["predict", "--pick-strategy", "expected-points"]) == 0
    assert seen["p"][0].pick_strategy == "expected-points"


def test_cli_rejects_unknown_pick_strategy():
    with pytest.raises(SystemExit):
        cli.main(["predict", "--pick-strategy", "vibes"])


def test_cli_backtest_prints_comparisons(monkeypatch, capsys):
    comparison = {"reference": "dixon-coles-modal", "challenger": "naive-favourite", "matches": 90,
                  "gameweeks": 9, "changed_scorelines": 30,
                  "components": {c: {"difference": 5, "per_match": 0.0556, "ci95": [-0.1, 0.2]}
                                 for c in ("total", "outcome", "scorers", "minutes")}}
    monkeypatch.setattr(cli, "load_season_inputs", _fake_inputs)
    monkeypatch.setattr(cli, "run_backtest", lambda f, l, i, **kw: {
        "gameweeks": [], "season": {}, "comparisons": [comparison],
        "availability_proxy": "live", "injury_impact": 0.0})
    assert cli.main(["backtest", "--from", "1", "--to", "1", "--availability", "live"]) == 0
    out = capsys.readouterr().out
    assert "naive-favourite vs dixon-coles-modal" in out and "+5" in out and "[-0.10, +0.20]" in out


def test_cli_predict_passes_scorer_model(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "run_gameweek", lambda gw, predictors: seen.update(p=predictors) or [])
    assert cli.main(["predict", "--scorer-model", "chance"]) == 0
    assert seen["p"][0].scorer_model == "chance"


def test_cli_backtest_loads_player_histories(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "load_season_inputs", lambda **kw: seen.update(kw) or _fake_inputs())
    monkeypatch.setattr(cli, "run_backtest", lambda f, l, i, **kw: {
        "gameweeks": [], "season": {}, "availability_proxy": "live", "injury_impact": 0.0, "scorer_leak": False})
    assert cli.main(["backtest", "--from", "1", "--to", "1", "--availability", "live"]) == 0
    assert seen == {"with_histories": True}
