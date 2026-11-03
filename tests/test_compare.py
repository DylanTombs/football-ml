import pytest

from football_ml.evaluate.compare import paired_comparison


def _row(fid, predicted, outcome, scorers=0, minutes=0):
    return {"fixture_id": fid, "predicted": predicted,
            "points": {"outcome": outcome, "scorers": scorers, "minutes": minutes,
                       "total": outcome + scorers + minutes}}


def _gw(gameweek, predictor, rows):
    return {"gameweek": gameweek, "predictor": predictor, "matches": rows}


REFERENCE = [
    _gw(1, "modal", [_row(1, "1-0", 1), _row(2, "1-1", 0)]),
    _gw(2, "modal", [_row(3, "2-0", 3), _row(4, "0-1", 0, scorers=1)]),
]


def test_identical_predictors_differ_by_zero():
    result = paired_comparison(REFERENCE, REFERENCE, resamples=200, seed=1)
    assert result["matches"] == 4
    assert result["changed_scorelines"] == 0
    assert result["components"]["total"] == {"difference": 0, "per_match": 0.0, "ci95": [0.0, 0.0]}


def test_constant_gain_has_a_tight_interval():
    better = [
        _gw(1, "ep", [_row(1, "2-1", 1, scorers=1), _row(2, "2-2", 0, scorers=1)]),
        _gw(2, "ep", [_row(3, "2-0", 3, scorers=1), _row(4, "1-2", 0, scorers=2)]),
    ]
    result = paired_comparison(REFERENCE, better, resamples=200, seed=1)
    assert (result["reference"], result["challenger"]) == ("modal", "ep")
    assert result["changed_scorelines"] == 3
    assert result["components"]["scorers"]["difference"] == 4
    assert result["components"]["scorers"]["per_match"] == pytest.approx(1.0)
    assert result["components"]["scorers"]["ci95"] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert result["components"]["outcome"]["difference"] == 0


def test_interval_resamples_whole_gameweeks_and_is_seeded():
    mixed = [
        _gw(1, "ep", [_row(1, "2-1", 3), _row(2, "1-1", 0)]),   # +2 in GW1
        _gw(2, "ep", [_row(3, "2-1", 1), _row(4, "0-1", 0)]),   # -3 in GW2
    ]
    first = paired_comparison(REFERENCE, mixed, resamples=500, seed=7)
    again = paired_comparison(REFERENCE, mixed, resamples=500, seed=7)
    total = first["components"]["total"]
    assert first == again
    assert total["per_match"] == pytest.approx(-0.25)
    lo, hi = total["ci95"]
    assert lo == pytest.approx(-1.5) and hi == pytest.approx(1.0)  # all-GW2 and all-GW1 resamples


def test_only_shared_fixtures_are_paired():
    partial = [_gw(1, "ep", [_row(1, "1-0", 1)])]
    assert paired_comparison(REFERENCE, partial, resamples=50, seed=0)["matches"] == 1


def test_no_shared_fixtures_is_an_error():
    with pytest.raises(ValueError, match="shared"):
        paired_comparison(REFERENCE, [_gw(9, "ep", [_row(99, "1-0", 1)])], resamples=50, seed=0)
