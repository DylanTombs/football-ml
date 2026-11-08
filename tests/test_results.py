import pandas as pd
import pytest

from football_ml.sources.espn import GoalEvent, parse_goal_events
from football_ml.sources.results import build_actual_matches, names_match
from football_ml.predict.scorers import Player

BOOTSTRAP = {
    "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Coventry City"}],
    "elements": [
        {"id": 10, "web_name": "Havertz", "first_name": "Kai", "second_name": "Havertz", "team": 1},
        {"id": 11, "web_name": "Ødegaard", "first_name": "Martin", "second_name": "Ødegaard", "team": 1},
        {"id": 20, "web_name": "Simms", "first_name": "Ellis", "second_name": "Simms", "team": 2},
        {"id": 21, "web_name": "Thomas", "first_name": "Bobby", "second_name": "Thomas", "team": 2},
    ],
}


def _fixture(fid=1, event=1, finished=True, hs=3, as_=1, goals_h=(), goals_a=(), og_h=(), og_a=()):
    stat = lambda pairs: [{"element": e, "value": v} for e, v in pairs]  # noqa: E731
    return {
        "id": fid, "event": event, "finished": finished, "team_h": 1, "team_a": 2,
        "team_h_score": hs, "team_a_score": as_, "kickoff_time": "2026-08-21T19:00:00Z",
        "stats": [
            {"identifier": "goals_scored", "h": stat(goals_h), "a": stat(goals_a)},
            {"identifier": "own_goals", "h": stat(og_h), "a": stat(og_a)},
        ],
    }


ESPN_EVENTS = (
    GoalEvent(pd.Timestamp("2026-08-21T19:00Z"), "Arsenal", "Coventry", "Arsenal", "Kai Havertz", "15'", False),
    GoalEvent(pd.Timestamp("2026-08-21T19:00Z"), "Arsenal", "Coventry", "Arsenal", "Kai Havertz", "60'", False),
    GoalEvent(pd.Timestamp("2026-08-21T19:00Z"), "Arsenal", "Coventry", "Coventry", "Ellis Simms", "88'", False),
    GoalEvent(pd.Timestamp("2026-08-21T19:00Z"), "Arsenal", "Coventry", "Arsenal", "Bobby Thomas", "70'", True),
)


def test_build_actual_match_scorers_and_minutes():
    raw = [_fixture(goals_h=[(10, 2)], goals_a=[(20, 1)], og_a=[(21, 1)], hs=3, as_=1)]
    (match,) = build_actual_matches(raw, BOOTSTRAP, ESPN_EVENTS, gameweek=1)
    assert (match.home, match.away, match.home_goals, match.away_goals) == ("Arsenal", "Coventry", 3, 1)

    scored = [g for g in match.goals if not g.own_goal]
    assert sorted(g.scorer_id for g in scored) == [10, 10, 20]
    havertz_minutes = sorted(g.minute.elapsed() for g in scored if g.scorer_id == 10)
    assert havertz_minutes == [15, 60]

    (og,) = [g for g in match.goals if g.own_goal]
    assert og.team == "Arsenal" and og.scorer_id == 21


def test_unfinished_and_other_gameweeks_are_skipped():
    raw = [_fixture(fid=1, finished=False), _fixture(fid=2, event=2)]
    assert build_actual_matches(raw, BOOTSTRAP, (), gameweek=1) == ()


def test_provisionally_finished_matches_count():
    """FPL sets `finished` only once bonus points are confirmed, often the next
    morning; the score and scorers are final at `finished_provisional`."""
    provisional = {**_fixture(fid=1, finished=False, goals_h=[(10, 1)], hs=1, as_=0), "finished_provisional": True}
    in_play = {**_fixture(fid=2, finished=False), "finished_provisional": False}
    (match,) = build_actual_matches([provisional, in_play], BOOTSTRAP, (), gameweek=1)
    assert match.fixture_id == 1
    assert match.goals[0].scorer_id == 10


def test_missing_espn_data_leaves_minutes_empty():
    raw = [_fixture(goals_h=[(10, 1)], hs=1, as_=0)]
    (match,) = build_actual_matches(raw, BOOTSTRAP, (), gameweek=1)
    assert match.goals[0].scorer_id == 10
    assert match.goals[0].minute is None


def test_names_match_handles_accents_and_short_names():
    player = Player(11, "Ødegaard", "Arsenal", 0, 0, 0, True, full_name="Martin Ødegaard")
    assert names_match("Martin Odegaard", player)
    assert names_match("M. Ødegaard", player)
    assert not names_match("Kai Havertz", player)


ESPN_PAYLOAD = {"events": [{
    "date": "2026-08-21T19:00Z",
    "competitions": [{
        "competitors": [
            {"homeAway": "home", "team": {"id": "359", "displayName": "Arsenal"}},
            {"homeAway": "away", "team": {"id": "388", "displayName": "Coventry City"}},
        ],
        "details": [
            {"scoringPlay": True, "team": {"id": "359"}, "clock": {"displayValue": "15'"},
             "ownGoal": False, "athletesInvolved": [{"displayName": "Kai Havertz"}]},
            {"scoringPlay": False, "team": {"id": "388"}, "clock": {"displayValue": "20'"}},
            {"scoringPlay": True, "team": {"id": "388"}, "clock": {"displayValue": "90'+2'"},
             "ownGoal": True, "athletesInvolved": []},
        ],
    }],
}]}


def test_parse_goal_events_maps_teams_and_flags():
    events = parse_goal_events(ESPN_PAYLOAD)
    assert len(events) == 2
    assert events[0].home == "Arsenal" and events[0].away == "Coventry"
    assert events[0].scorer == "Kai Havertz" and events[0].clock == "15'"
    assert events[1].team == "Coventry" and events[1].own_goal and events[1].scorer is None


def test_parse_goal_events_empty_payload():
    assert parse_goal_events({}) == ()


@pytest.mark.parametrize("espn,web,full", [
    ("Pascal Gross", "Groß", "Pascal Groß"),
    ("Bruno Guimarães", "Bruno G.", "Bruno Guimarães Rodriguez Moura"),
    ("Yeremy Pino", "Yeremy", "Yeremy Jesús Pino Santos"),
    ("Jay Da Silva", "Dasilva", "Jay Dasilva"),
])
def test_names_match_real_world_variants(espn, web, full):
    assert names_match(espn, Player(1, web, "X", 0, 0, 0, True, full_name=full))
