import math

import numpy as np
import pandas as pd
import pytest

from football_ml.model.dixon_coles import Structure
from football_ml.model.shots import fit_team_model


def _results():
    """Strong outshoots everyone; goals are noisier than shots."""
    teams = ["Strong", "MidA", "MidB", "Weak"]
    rng = np.random.default_rng(0)
    rows = []
    for week in range(30):
        for i, h in enumerate(teams):
            for a in teams[i + 1:]:
                home, away = (h, a) if week % 2 else (a, h)
                sot_h = 9 if home == "Strong" else 5 if away != "Strong" else 3
                sot_a = 8 if away == "Strong" else 4 if home != "Strong" else 2
                rows.append({"date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=7 * week), "home": home,
                             "away": away, "home_goals": int(rng.poisson(sot_h * 0.3)),
                             "away_goals": int(rng.poisson(sot_a * 0.3)), "home_sot": sot_h, "away_sot": sot_a,
                             "league": "E0", "season": "2526"})
    return pd.DataFrame(rows)


CUTOFF = pd.Timestamp("2026-06-01")


def test_without_shots_the_team_model_is_plain_dixon_coles():
    from football_ml.model.dixon_coles import fit
    model = fit_team_model(_results(), CUTOFF, xi=0.0019, l2=0.5, structure=Structure())
    plain = fit(_results(), CUTOFF, xi=0.0019, l2=0.5)
    assert model.expected_goals("Strong", "Weak") == pytest.approx(plain.expected_goals("Strong", "Weak"))
    assert model.rho == plain.rho and model.home_advantage == plain.home_advantage


def test_shots_weight_blends_goal_and_shot_rates_geometrically():
    results = _results()
    goals = fit_team_model(results, CUTOFF, xi=0.0019, l2=0.5, structure=Structure())
    shots = fit_team_model(results, CUTOFF, xi=0.0019, l2=0.5, structure=Structure(shots_weight=1.0))
    half = fit_team_model(results, CUTOFF, xi=0.0019, l2=0.5, structure=Structure(shots_weight=0.5))
    g, s, h = (m.expected_goals("Strong", "Weak") for m in (goals, shots, half))
    assert h[0] == pytest.approx(math.sqrt(g[0] * s[0]))
    assert h[1] == pytest.approx(math.sqrt(g[1] * s[1]))
    # shot-based rates: expected shots on target × goals per shot on target
    assert 0.2 < shots.conversion < 0.4
    assert s[0] > s[1]
    assert shots.rho == goals.rho  # the low-score correction stays the goals model's


def test_shots_label_and_missing_shot_columns():
    from football_ml.model.structure import label
    assert label(Structure(shots_weight=0.5)) == "shots0.5"
    no_shots = _results().drop(columns=["home_sot", "away_sot"])
    with pytest.raises(ValueError, match="sot"):
        fit_team_model(no_shots, CUTOFF, xi=0.0019, l2=0.5, structure=Structure(shots_weight=0.5))
