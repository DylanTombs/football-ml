import json

import pandas as pd
import pytest

from football_ml.snapshots import build_snapshot, load_snapshot, snapshot_players, write_snapshot
from football_ml.sources.fpl import parse_players

BOOTSTRAP = {
    "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"}, {"id": 2, "name": "Spurs"}],
    "events": [{"id": 10, "deadline_time": "2026-11-07T11:00:00Z"}],
    "elements": [
        {"id": 7, "web_name": "Saka", "first_name": "Bukayo", "second_name": "Saka", "team": 1,
         "element_type": 3, "expected_goals": "3.40", "goals_scored": 4, "minutes": 780, "starts": 9,
         "status": "d", "chance_of_playing_next_round": 75, "news": "Knock - 75% chance of playing",
         "penalties_order": 1, "total_points": 61, "photo": "x.jpg"},
        {"id": 9, "web_name": "Kane", "first_name": "Harry", "second_name": "Kane", "team": 2,
         "element_type": 4, "expected_goals": "0.0", "goals_scored": 0, "minutes": 0, "starts": 0,
         "status": "u", "chance_of_playing_next_round": None, "news": "Joined Bayern", "penalties_order": None},
    ],
}
BEFORE = pd.Timestamp("2026-11-06T09:00Z")
AFTER = pd.Timestamp("2026-11-07T11:00Z")


def test_snapshot_keeps_the_fields_we_cannot_recover():
    snap = build_snapshot(BOOTSTRAP, 10, taken_at=BEFORE)
    saka = snap["elements"][0]
    assert snap["gameweek"] == 10
    assert snap["deadline"] == "2026-11-07T11:00:00+00:00"
    assert saka["news"] == "Knock - 75% chance of playing"
    assert saka["penalties_order"] == 1 and saka["starts"] == 9
    assert "photo" not in saka and "total_points" not in saka


def test_snapshot_round_trips_to_the_same_players(tmp_path):
    path = write_snapshot(BOOTSTRAP, 10, now=BEFORE, snapshots_dir=tmp_path)
    assert path == tmp_path / "gw10.json"
    assert snapshot_players(load_snapshot(path)) == parse_players(BOOTSTRAP)
    assert json.loads(path.read_text())["taken_at"] == BEFORE.isoformat()


def test_snapshot_is_not_written_after_the_deadline(tmp_path):
    assert write_snapshot(BOOTSTRAP, 10, now=AFTER, snapshots_dir=tmp_path) is None
    assert not (tmp_path / "gw10.json").exists()


def test_snapshot_refreshes_before_the_deadline(tmp_path):
    write_snapshot(BOOTSTRAP, 10, now=BEFORE, snapshots_dir=tmp_path)
    later = BEFORE + pd.Timedelta(hours=12)
    path = write_snapshot(BOOTSTRAP, 10, now=later, snapshots_dir=tmp_path)
    assert load_snapshot(path)["taken_at"] == later.isoformat()


def test_snapshot_needs_a_known_gameweek():
    with pytest.raises(ValueError, match="gameweek 11"):
        build_snapshot(BOOTSTRAP, 11, taken_at=BEFORE)
