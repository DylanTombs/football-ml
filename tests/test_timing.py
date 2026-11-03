import pytest

from football_ml.predict.timing import GoalTime, goal_times, order_goals, parse_clock


@pytest.mark.parametrize("raw,label,key", [
    ("15'", "15'", 15.0),
    ("45'+2'", "45+2'", 45.02),
    ("90'+4'", "90+4'", 90.04),
])
def test_parse_clock(raw, label, key):
    t = parse_clock(raw)
    assert t.label == label
    assert t.sort_key == pytest.approx(key)


def test_parse_clock_rejects_garbage():
    with pytest.raises(ValueError):
        parse_clock("half time")


DISTRIBUTION = tuple(GoalTime(float(m), f"{m}'") for m in range(1, 91))


def test_goal_times_count_and_order():
    times = goal_times(3, DISTRIBUTION)
    assert len(times) == 3
    keys = [t.sort_key for t in times]
    assert keys == sorted(keys)


def test_goal_times_spread_over_match():
    times = goal_times(2, DISTRIBUTION)
    assert times[0].sort_key < 45 < times[1].sort_key


def test_goal_times_seed_is_reproducible_and_varies():
    a = goal_times(3, DISTRIBUTION, seed=1)
    assert a == goal_times(3, DISTRIBUTION, seed=1)
    variants = {goal_times(3, DISTRIBUTION, seed=s) for s in range(10)}
    assert len(variants) > 1


def test_goal_times_zero():
    assert goal_times(0, DISTRIBUTION) == ()


def test_goal_times_empty_distribution_raises():
    with pytest.raises(ValueError):
        goal_times(1, ())


def test_order_goals_interleaves_teams():
    assert order_goals(2, 1) == ("home", "away", "home")
    assert order_goals(0, 2) == ("away", "away")
    assert order_goals(1, 1) == ("home", "away")
    assert order_goals(0, 0) == ()
