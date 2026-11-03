# 06 — Multi-season test bench + bookmaker odds

**Goal:** judge changes to the score model on five seasons instead of 90 matches, then blend bookmaker odds into λ and μ, which is the strongest information we can get.

## Decisions

- **Walk-forward over past seasons:** refit before every gameweek using only earlier matches. Tune on 2021-22 to 2024-25 (1,520 matches). Hold out 2025-26 and test on it once. Add 2020-21 so the first tuning season has a warm-up year.
- **Metric for tuning:** out-of-sample log-likelihood of the exact score (log loss). Report 1X2 log loss, RPS and outcome points alongside. Points are too noisy to tune on: even near-identical variants need about a 16-point gap over 90 matches to be told apart.
- **Choosing settings:** a small grid (time decay, ridge penalty). Pick the simplest setting within one standard error of the best.
- **Odds to rates:**
  - remove the bookmaker margin with Shin's method
  - fit λ and μ to the 1X2 and over/under 2.5 probabilities on a Dixon-Coles grid (as `penaltyblog`'s `goal_expectancy` does)
  - blend: `λ = w × model + (1 − w) × market`, with `w` tuned on the walk-forward. Expect a small weight on the model.
- **Odds data:**
  - Use pre-closing odds only (columns without the `C` suffix), never closing odds. They're collected on Friday afternoon (by about 17:00 UK) for weekend games and on Tuesday for midweek games.
  - The live source is football-data's `fixtures.csv`. It's overwritten every round, so save a copy with each frozen prediction.
  - If a match has no odds, fall back to the model alone.
- **Paired comparison against the plan 01 / 04 baseline** for every change, as the CLAUDE.md rule requires.

## Tasks

- [ ] Add 2020-21 to the training data; walk-forward runner over past seasons (score part only)
- [ ] Log loss, RPS and calibration output (predicted vs actual draw rate and total goals)
- [ ] Tune time decay and the ridge penalty, with results recorded here
- [ ] `sources/odds.py`: pre-closing odds from season CSVs and `fixtures.csv`, Shin margin removal, with tests
- [ ] Odds → λ, μ; blend weight tuned on the walk-forward
- [ ] Save `fixtures.csv` odds alongside each gameweek's frozen prediction

## Done when

- The model's settings are chosen on the multi-season walk-forward. The odds blend is on by default only if it beats the model alone on held-out 2025-26 log loss without losing outcome points.
