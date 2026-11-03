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
    def fail(_):
        raise http.DataSourceError("FPL down")
    monkeypatch.setattr(cli, "run_gameweek", fail)
    assert cli.main(["predict"]) == 1
    assert "FPL down" in capsys.readouterr().err


def test_cli_prints_report(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_gameweek", lambda _: {"gameweek": 10, "matches": [MATCH], "path": "p"})
    assert cli.main(["predict", "--gameweek", "10"]) == 0
    assert "Gameweek 10" in capsys.readouterr().out
