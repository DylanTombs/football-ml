from datetime import date

import pandas as pd
import pytest

from football_ml.config import FACT_EFFECTS, MAX_ADJUSTMENT
from football_ml.facts.adjust import fixture_adjustments, gameweek_adjustments, side_multiplier
from football_ml.facts.adjust import Adjustment
from football_ml.facts.extractor import PlayerFact
from football_ml.facts.fixture import Congestion, FixtureFacts, NewManager
from football_ml.model.dixon_coles import DixonColesParams
from football_ml.predict.scorers import Player
from football_ml.predict.timing import GoalTime
from football_ml.predictors.context import fixture_context
from football_ml.sources.fpl import Fixture

PARAMS = DixonColesParams(
    teams=("Arsenal", "Coventry", "Leeds", "Hull"), attack=(0.5, -0.5, 0.0, -0.2),
    defence=(-0.4, 0.4, 0.0, 0.2), home_advantage=0.25, rho=-0.05,
)
PLAYERS = (
    Player(1, "Saka", "Arsenal", xg=4.0, goals=5, minutes=800, available=False, availability=0.0),
    Player(2, "Havertz", "Arsenal", xg=4.0, goals=3, minutes=900, available=True),
    Player(4, "Saliba", "Arsenal", xg=0.2, goals=0, minutes=900, available=False, availability=0.0),
    Player(6, "Bench", "Arsenal", xg=0.1, goals=0, minutes=90, available=False, availability=0.0),
    Player(7, "Doubt", "Arsenal", xg=2.0, goals=1, minutes=500, available=True, availability=0.5),
    Player(3, "Simms", "Coventry", xg=2.0, goals=1, minutes=600, available=True),
    Player(5, "Wright", "Coventry", xg=1.0, goals=1, minutes=600, available=True),
)
POSITIONS = {1: 3, 2: 4, 4: 2, 6: 2, 7: 3, 3: 4, 5: 3}
DISTRIBUTION = tuple(GoalTime(float(m), f"{m}'") for m in range(1, 91))
KICKOFF = pd.Timestamp("2026-11-21T15:00Z")
AS_OF = date(2026, 11, 20)
FIXTURE = Fixture(id=100, gameweek=12, kickoff=KICKOFF, home="Arsenal", away="Coventry")
OTHER = Fixture(id=101, gameweek=12, kickoff=KICKOFF, home="Leeds", away="Hull")
CTX = fixture_context(12, FIXTURE, PARAMS, PLAYERS, DISTRIBUTION, injury_impact=0.0)
OTHER_CTX = fixture_context(12, OTHER, PARAMS, PLAYERS, DISTRIBUTION, injury_impact=0.0)

NO_CONGESTION = Congestion(7.0, "eng.1", 7.0, "eng.1")
PLAIN = FixtureFacts(100, False, NO_CONGESTION, NO_CONGESTION, None, None)
RECENT = date(2026, 11, 15)


def _fact(pid, reason="injury", back=None, added=RECENT, team="Arsenal"):
    return PlayerFact(pid, team, reason, back, added)


def _kinds(adjustments):
    return sorted((a.side, a.kind) for a in adjustments)


def test_new_absence_of_a_key_attacker_lowers_own_rate():
    [adj] = fixture_adjustments(CTX, (_fact(1),), PLAIN, POSITIONS, AS_OF)
    assert (adj.side, adj.kind, adj.effect) == ("home", "key_attacker_out", FACT_EFFECTS["key_attacker_out"])
    assert "Saka" in adj.detail


def test_old_news_is_already_in_the_ratings():
    old = _fact(1, added=date(2026, 9, 1))
    assert fixture_adjustments(CTX, (old,), PLAIN, POSITIONS, AS_OF) == ()
    assert fixture_adjustments(CTX, (_fact(1, added=None),), PLAIN, POSITIONS, AS_OF) == ()


def test_player_due_back_before_kickoff_is_not_missing():
    assert fixture_adjustments(CTX, (_fact(1, back=date(2026, 11, 21)),), PLAIN, POSITIONS, AS_OF) == ()


def test_doubtful_player_counts_by_chance_of_missing():
    [adj] = fixture_adjustments(CTX, (_fact(7),), PLAIN, POSITIONS, AS_OF)
    assert adj.effect == pytest.approx(FACT_EFFECTS["key_attacker_out"] * 0.5)


def test_regular_defender_out_raises_the_opponent_rate_but_a_fringe_one_does_not():
    adjustments = fixture_adjustments(CTX, (_fact(4), _fact(6)), PLAIN, POSITIONS, AS_OF)
    assert _kinds(adjustments) == [("away", "defender_out")]


def test_key_attacker_back_raises_own_rate():
    [adj] = fixture_adjustments(CTX, (_fact(2, reason="returned"),), PLAIN, POSITIONS, AS_OF)
    assert (adj.side, adj.kind) == ("home", "key_attacker_back")


def test_facts_for_other_teams_or_unknown_players_are_ignored():
    facts = (_fact(3, team="Leeds"), _fact(99))
    assert fixture_adjustments(CTX, facts, PLAIN, POSITIONS, AS_OF) == ()


def test_fixture_facts_become_adjustments():
    tired = Congestion(2.5, "uefa.champions", 3.0, "uefa.europa")
    facts = FixtureFacts(100, True, tired, NO_CONGESTION, None, NewManager("Boss", 10))
    assert _kinds(fixture_adjustments(CTX, (), facts, POSITIONS, AS_OF)) == [
        ("away", "derby"), ("away", "new_manager"),
        ("home", "derby"), ("home", "europe_next"), ("home", "short_rest"),
    ]


def test_league_game_next_is_not_rotation():
    soon = Congestion(7.0, "eng.1", 3.0, "eng.1")
    facts = FixtureFacts(100, False, soon, NO_CONGESTION, None, None)
    assert fixture_adjustments(CTX, (), facts, POSITIONS, AS_OF) == ()


def test_side_multiplier_is_capped():
    many = tuple(Adjustment("home", "key_attacker_out", -0.1, "x") for _ in range(5))
    assert side_multiplier(many, "home") == pytest.approx(1 - MAX_ADJUSTMENT)
    assert side_multiplier(many, "away") == 1.0


def test_gameweek_adjustments_are_recentred_so_total_goals_stay_the_same():
    facts = {100: PLAIN, 101: FixtureFacts(101, False, NO_CONGESTION, NO_CONGESTION, None, None)}
    result = gameweek_adjustments((CTX, OTHER_CTX), (_fact(1),), facts, POSITIONS, AS_OF)
    before = sum(c.rates[0] + c.rates[1] for c in (CTX, OTHER_CTX))
    after = sum(c.rates[0] * result[c.fixture.id].home_multiplier + c.rates[1] * result[c.fixture.id].away_multiplier
                for c in (CTX, OTHER_CTX))
    assert after == pytest.approx(before)
    assert result[100].home_multiplier < 1 < result[101].home_multiplier
    assert result[100].adjustments[0].kind == "key_attacker_out" and result[101].adjustments == ()


def test_gameweek_without_facts_leaves_every_rate_alone():
    result = gameweek_adjustments((CTX, OTHER_CTX), (), {}, POSITIONS, AS_OF)
    assert {(a.home_multiplier, a.away_multiplier) for a in result.values()} == {(1.0, 1.0)}


def test_press_availability_overrides_fpl():
    fit = PlayerFact(1, "Arsenal", "injury", None, RECENT, availability=1.0)
    assert fixture_adjustments(CTX, (fit,), PLAIN, POSITIONS, AS_OF) == ()
    doubt = PlayerFact(2, "Arsenal", "fitness", None, RECENT, availability=0.5)
    [adj] = fixture_adjustments(CTX, (doubt,), PLAIN, POSITIONS, AS_OF)
    assert adj.effect == pytest.approx(FACT_EFFECTS["key_attacker_out"] * 0.5)
