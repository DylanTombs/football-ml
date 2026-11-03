import pytest

from football_ml.predict.timing import GoalTime, goal_times, order_goals, parse_clock


@pytest.mark.parametrize("raw,label,key", [
    ("15'", "15'", 15.0),
    ("45'+2'", "45+2'", 45.02),
    ("90'+4'", "90+4'", 90.04),
    ("90+4'", "90+4'", 90.04),
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


def test_elapsed_adds_stoppage_time():
    assert parse_clock("90'+4'").elapsed() == 94
    assert parse_clock("45+2'").elapsed() == 47
    assert parse_clock("15'").elapsed() == 15


# --- fixed minutes (plan 04) -------------------------------------------------

from football_ml.predict.timing import best_minutes, elapsed_goal_time, pick_minutes, window_coverage  # noqa: E402

CLUSTERED = (50,) * 10 + (80,) * 5 + (20,) * 3


def test_window_coverage():
    assert window_coverage((50, 55, 70), minute=52, tolerance=10) == pytest.approx(2 / 3)
    assert window_coverage((), minute=52, tolerance=10) == 0.0


def test_best_single_minute_covers_the_densest_cluster():
    (minute,) = best_minutes(CLUSTERED, k=1, tolerance=10)
    assert 40 <= minute <= 60


def test_best_minutes_tile_non_overlapping_windows():
    first, second = best_minutes(CLUSTERED, k=2, tolerance=10)
    assert second - first > 20
    assert window_coverage(CLUSTERED, first, 10) + window_coverage(CLUSTERED, second, 10) == pytest.approx(15 / 18)


def test_best_minutes_rejects_bad_k():
    with pytest.raises(ValueError):
        best_minutes(CLUSTERED, k=0, tolerance=10)


TABLE = {1: (52,), 2: (39, 61)}


@pytest.mark.parametrize("scorers,minutes", [
    ((1,), (52,)),
    ((1, 1), (39, 61)),
    ((1, 2), (52, 53)),          # different players: second nudged off the same minute
    ((1, 2, 1), (39, 52, 61)),   # player 1 tiles its two windows, player 2 keeps the best
    ((None, None), (52, 53)),    # unknown scorers count as different players
    ((7, 7, 7), (15, 45, 75)),   # no tuned windows for three: spread evenly
])
def test_pick_minutes(scorers, minutes):
    assert pick_minutes(scorers, TABLE) == minutes


def test_elapsed_goal_time_labels():
    assert elapsed_goal_time(52) == GoalTime(52.0, "52'")
    stoppage = elapsed_goal_time(93)
    assert stoppage.label == "90+3'"
    assert stoppage.elapsed() == 93
