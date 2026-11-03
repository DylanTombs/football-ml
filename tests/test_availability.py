import pytest

from football_ml.predict.availability import adjust_rate, team_availability
from football_ml.predict.scorers import Player


def _player(pid, name, xg, availability=1.0, departed=False):
    return Player(pid, name, "Man City", xg=xg, goals=0, minutes=900,
                  available=availability >= 0.75, availability=availability, departed=departed)


FULL_SQUAD = (
    _player(1, "Haaland", 6.0),
    _player(2, "Foden", 3.0),
    _player(3, "Doku", 1.0),
)


def test_full_squad_has_nothing_missing():
    result = team_availability(FULL_SQUAD)
    assert result.missing_share == 0.0
    assert result.missing == ()


def test_star_out_removes_his_share_of_team_xg():
    squad = (_player(1, "Haaland", 6.0, availability=0.0), *FULL_SQUAD[1:])
    result = team_availability(squad)
    assert result.missing_share == pytest.approx(0.6)
    assert [(m.name, m.xg_share) for m in result.missing] == [("Haaland", pytest.approx(0.6))]


def test_doubtful_player_counts_half():
    squad = (*FULL_SQUAD[:2], _player(3, "Doku", 1.0, availability=0.5))
    assert team_availability(squad).missing_share == pytest.approx(0.05)


def test_missing_players_sorted_by_share():
    squad = (
        _player(1, "Haaland", 6.0, availability=0.0),
        _player(2, "Foden", 3.0, availability=0.0),
        _player(3, "Doku", 1.0),
    )
    assert [m.name for m in team_availability(squad).missing] == ["Haaland", "Foden"]


def test_departed_players_are_not_part_of_the_team():
    squad = (*FULL_SQUAD, _player(4, "Loanee", 10.0, availability=0.0, departed=True))
    assert team_availability(squad).missing_share == 0.0


def test_squad_without_xg_has_nothing_missing():
    squad = (_player(1, "Keeper", 0.0, availability=0.0),)
    assert team_availability(squad).missing_share == 0.0
    assert team_availability(()).missing_share == 0.0


@pytest.mark.parametrize("rate, share, impact, expected", [
    (2.0, 0.0, 0.5, 2.0),
    (2.0, 0.6, 0.0, 2.0),
    (2.0, 0.6, 0.5, 1.4),
    (1.0, 1.0, 1.0, 0.0),
])
def test_adjust_rate(rate, share, impact, expected):
    assert adjust_rate(rate, share, impact) == pytest.approx(expected)


def test_adjust_rate_rejects_out_of_range_inputs():
    with pytest.raises(ValueError):
        adjust_rate(1.0, 1.2, 0.5)
    with pytest.raises(ValueError):
        adjust_rate(1.0, 0.5, -0.1)
