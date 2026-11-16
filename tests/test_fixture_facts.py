from datetime import date

import pandas as pd

from football_ml.facts.fixture import (
    Congestion,
    ManagerChange,
    congestion,
    fixture_facts,
    is_derby,
    new_manager,
)
from football_ml.sources.espn_schedule import ScheduledMatch
from football_ml.sources.fpl import Fixture

KICKOFF = pd.Timestamp("2026-11-21T17:30Z")
FIXTURE = Fixture(id=200, gameweek=12, kickoff=KICKOFF, home="Newcastle", away="Arsenal")


def _m(team, when, competition="eng.1"):
    return ScheduledMatch(team, pd.Timestamp(when), competition)


SCHEDULE = (
    _m("Arsenal", "2026-11-04T20:00Z", "uefa.champions"),
    _m("Arsenal", "2026-11-07T15:00Z"),
    _m("Arsenal", "2026-11-21T17:30Z"),  # this fixture itself
    _m("Arsenal", "2026-11-24T20:00Z", "uefa.champions"),
    _m("Newcastle", "2026-11-08T14:00Z"),
    _m("Newcastle", "2026-11-18T19:45Z", "eng.league_cup"),
    _m("Newcastle", "2026-11-21T17:15Z"),  # same match, ESPN time a little off
)


# --- congestion ---------------------------------------------------------------

def test_congestion_finds_rest_and_next_gap_across_competitions():
    arsenal = congestion(SCHEDULE, "Arsenal", KICKOFF)
    assert arsenal == Congestion(rest_days=14.1, previous_competition="eng.1",
                                 days_to_next=3.1, next_competition="uefa.champions")


def test_congestion_ignores_the_fixture_itself_even_when_times_differ():
    newcastle = congestion(SCHEDULE, "Newcastle", KICKOFF)
    assert newcastle.rest_days == 2.9 and newcastle.previous_competition == "eng.league_cup"
    assert newcastle.days_to_next is None and newcastle.next_competition is None


def test_congestion_without_a_schedule_is_unknown():
    assert congestion((), "Arsenal", KICKOFF) == Congestion(None, None, None, None)


# --- derbies -------------------------------------------------------------------

def test_derbies_work_either_way_round():
    assert is_derby("Arsenal", "Tottenham") and is_derby("Tottenham", "Arsenal")
    assert is_derby("Newcastle", "Sunderland")
    assert not is_derby("Arsenal", "Newcastle")


# --- new manager -----------------------------------------------------------------

CHANGES = (
    ManagerChange("Arsenal", date(2026, 10, 30), "New Boss"),
    ManagerChange("Arsenal", date(2025, 1, 1), "Old Boss"),
    ManagerChange("Newcastle", date(2026, 11, 25), "Future Boss"),
    ManagerChange("Chelsea", date(2026, 6, 1), "Long Ago"),
)


def test_new_manager_reports_the_latest_appointment_within_the_window():
    fact = new_manager(CHANGES, "Arsenal", KICKOFF)
    assert fact.manager == "New Boss" and fact.days_in_charge == 22


def test_new_manager_ignores_future_and_old_appointments():
    assert new_manager(CHANGES, "Newcastle", KICKOFF) is None
    assert new_manager(CHANGES, "Chelsea", KICKOFF) is None
    assert new_manager((), "Arsenal", KICKOFF) is None


# --- per fixture -------------------------------------------------------------------

def test_fixture_facts_combine_both_sides():
    facts = fixture_facts(FIXTURE, SCHEDULE, CHANGES)
    assert facts.fixture_id == 200 and facts.derby is False
    assert facts.home_congestion.rest_days == 2.9
    assert facts.away_congestion.next_competition == "uefa.champions"
    assert facts.home_new_manager is None
    assert facts.away_new_manager.manager == "New Boss"


def test_manager_table_in_config_is_well_formed():
    from football_ml.config import MANAGER_CHANGES
    from football_ml.facts.fixture import manager_changes
    changes = manager_changes(MANAGER_CHANGES)
    assert all(isinstance(c.appointed, date) and c.team and c.manager for c in changes)
