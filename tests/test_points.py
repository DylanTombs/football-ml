from football_ml.evaluate.points import score_prediction
from football_ml.predict.timing import parse_clock
from football_ml.sources.results import ActualGoal, ActualMatch


def _actual(hg, ag, goals=()):
    return ActualMatch(fixture_id=1, gameweek=1, home="Arsenal", away="Hull",
                       home_goals=hg, away_goals=ag, goals=tuple(goals))


def _goal(pid, minute, team="Arsenal", own_goal=False):
    return ActualGoal(team=team, scorer_id=pid, scorer_name=str(pid),
                      minute=parse_clock(minute) if minute else None, own_goal=own_goal)


def _predicted(hg, ag, goals=()):
    return {
        "fixture_id": 1,
        "prediction": {
            "score": {"home": hg, "away": ag},
            "goals": [{"scorer_id": pid, "scorer": str(pid), "minute": m, "team": "Arsenal"}
                      for pid, m in goals],
        },
    }


def test_exact_score_gets_exact_points_only():
    pts = score_prediction(_predicted(2, 1), _actual(2, 1))
    assert pts.outcome_points == 3


def test_correct_result_wrong_score():
    assert score_prediction(_predicted(1, 0), _actual(3, 1)).outcome_points == 1


def test_wrong_result():
    assert score_prediction(_predicted(1, 0), _actual(1, 1)).outcome_points == 0


def test_draws_compare_as_draws():
    assert score_prediction(_predicted(1, 1), _actual(0, 0)).outcome_points == 1


def test_scorer_points_use_multiset_matching():
    predicted = _predicted(2, 0, [(10, "20'"), (10, "70'")])
    actual = _actual(2, 0, [_goal(10, "15'"), _goal(11, "80'")])
    pts = score_prediction(predicted, actual)
    assert pts.scorer_points == 1  # Havertz once, not twice


def test_own_goals_never_count_as_scorer_hits():
    predicted = _predicted(1, 0, [(21, "70'")])
    actual = _actual(1, 0, [_goal(21, "70'", own_goal=True)])
    pts = score_prediction(predicted, actual)
    assert pts.scorer_points == 0 and pts.minute_points == 0


def test_minute_points_within_tolerance_only_for_matched_scorers():
    predicted = _predicted(2, 0, [(10, "20'"), (11, "45+2'")])
    actual = _actual(2, 0, [_goal(10, "29'"), _goal(11, "60'")])
    pts = score_prediction(predicted, actual)
    assert pts.scorer_points == 2
    assert pts.minute_points == 1  # 20 vs 29 within 10; 47 vs 60 not


def test_minute_points_skip_when_actual_minute_unknown():
    pts = score_prediction(_predicted(1, 0, [(10, "20'")]), _actual(1, 0, [_goal(10, None)]))
    assert pts.scorer_points == 1 and pts.minute_points == 0


def test_total():
    predicted = _predicted(1, 0, [(10, "20'")])
    pts = score_prediction(predicted, _actual(1, 0, [_goal(10, "25'")]))
    assert pts.total == 3 + 1 + 1


def test_missing_scorer_id_is_not_a_hit():
    predicted = _predicted(1, 0, [(None, "20'")])
    assert score_prediction(predicted, _actual(1, 0, [_goal(10, "20'")])).scorer_points == 0
