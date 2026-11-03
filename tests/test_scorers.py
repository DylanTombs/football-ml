from football_ml.predict.scorers import Player, assign_scorers


def _player(pid, name, xg, goals=0, available=True, minutes=900):
    return Player(id=pid, name=name, team="Arsenal", xg=xg, goals=goals,
                  minutes=minutes, available=available)


SQUAD = (
    _player(1, "Striker", xg=6.0, goals=7),
    _player(2, "Winger", xg=3.0, goals=3),
    _player(3, "Midfielder", xg=1.0, goals=1),
    _player(4, "Injured", xg=9.0, goals=9, available=False),
    _player(5, "Unused", xg=0.0, goals=0, minutes=0),
)


def test_zero_goals_means_no_scorers():
    assert assign_scorers(SQUAD, 0) == ()


def test_first_goal_goes_to_best_available_scorer():
    assert assign_scorers(SQUAD, 1)[0].name == "Striker"


def test_unavailable_and_unused_players_never_score():
    names = {p.name for p in assign_scorers(SQUAD, 6)}
    assert "Injured" not in names
    assert "Unused" not in names


def test_goals_are_spread_with_repeat_decay():
    names = [p.name for p in assign_scorers(SQUAD, 2, repeat_decay=1.0)]
    # Striker 6/(1+1)=3 ties Winger 3 -> second goal not automatically Striker again
    assert names[0] == "Striker"
    assert len(names) == 2


def test_hat_trick_possible_for_dominant_scorer():
    squad = (_player(1, "Star", xg=10.0, goals=10), _player(2, "Other", xg=1.0))
    names = [p.name for p in assign_scorers(squad, 3, repeat_decay=1.0)]
    assert names.count("Star") == 3


def test_no_eligible_players_returns_empty():
    squad = (_player(1, "Out", xg=5.0, available=False),)
    assert assign_scorers(squad, 2) == ()


def test_does_not_mutate_input():
    squad = tuple(SQUAD)
    assign_scorers(squad, 3)
    assert squad == SQUAD
