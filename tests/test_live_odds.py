import pandas as pd

from football_ml.sources import odds as live_odds
from football_ml.sources.fpl import Fixture
from football_ml.sources.http import DataSourceError

FIXTURES_CSV = (
    "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5\n"
    "E0,21/11/2026,12:30,Man City,Fulham,1.40,5.00,7.50,1.55,2.45\n"
    "E1,21/11/2026,15:00,Leicester,Hull,2.10,3.40,3.50,1.95,1.85\n"
)
BOOTSTRAP = {"events": [{"id": 11, "deadline_time": "2026-11-21T11:00:00Z"}]}
FIXTURES = (Fixture(1, 11, pd.Timestamp("2026-11-21T12:30Z"), "Man City", "Fulham"),)


def test_fetch_upcoming_odds_keeps_the_premier_league(monkeypatch):
    seen = {}
    monkeypatch.setattr(live_odds, "fetch_text", lambda url, cache_name, max_age_hours: seen.update(url=url) or FIXTURES_CSV)
    df = live_odds.fetch_upcoming_odds()
    assert list(df["home"]) == ["Man City"] and df.iloc[0]["odds_home"] == 1.40
    assert seen["url"].endswith("fixtures.csv")


def test_fetch_upcoming_odds_failure_gives_empty_frame(monkeypatch, caplog):
    def fail(*_, **__):
        raise DataSourceError("down")
    monkeypatch.setattr(live_odds, "fetch_text", fail)
    assert live_odds.fetch_upcoming_odds().empty
    assert "down" in caplog.text


def test_odds_snapshot_only_before_the_deadline(tmp_path):
    odds = pd.DataFrame([{"date": pd.Timestamp("2026-11-21"), "home": "Man City", "away": "Fulham",
                          "odds_home": 1.4, "odds_draw": 5.0, "odds_away": 7.5, "odds_over": 1.55, "odds_under": 2.45}])
    path = live_odds.write_odds_snapshot(odds, BOOTSTRAP, FIXTURES, pd.Timestamp("2026-11-20T18:00Z"), tmp_path)
    assert path == tmp_path / "gw11.csv"
    assert pd.read_csv(path)["home"].tolist() == ["Man City"]
    assert live_odds.write_odds_snapshot(odds, BOOTSTRAP, FIXTURES, pd.Timestamp("2026-11-21T12:00Z"), tmp_path / "late") is None
    assert live_odds.write_odds_snapshot(odds.iloc[0:0], BOOTSTRAP, FIXTURES, pd.Timestamp("2026-11-20T18:00Z"), tmp_path / "none") is None


def test_fetch_upcoming_odds_survives_a_latin1_decoded_bom(monkeypatch):
    """football-data serves a UTF-8 BOM but labels the file Latin-1, so the
    first header arrives as 'ï»¿Div'."""
    mangled = "ï»¿" + FIXTURES_CSV
    monkeypatch.setattr(live_odds, "fetch_text", lambda *a, **k: mangled)
    assert list(live_odds.fetch_upcoming_odds()["home"]) == ["Man City"]


def test_odds_snapshot_skips_last_rounds_rows(tmp_path):
    """fixtures.csv still holds the previous round until football-data updates it."""
    stale = pd.DataFrame([{"date": pd.Timestamp("2026-11-08"), "home": "Man City", "away": "Fulham",
                           "odds_home": 1.4, "odds_draw": 5.0, "odds_away": 7.5, "odds_over": 1.55, "odds_under": 2.45}])
    assert live_odds.write_odds_snapshot(stale, BOOTSTRAP, FIXTURES, pd.Timestamp("2026-11-20T18:00Z"), tmp_path) is None
